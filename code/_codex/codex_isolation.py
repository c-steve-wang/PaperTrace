"""One fail-closed file-access policy for all model-facing Codex calls."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

PROFILE = 'isolated_files'
POLICY_VERSION = 'codex-file-isolation-v1'
_TEMP_ROOTS = ('/tmp', '/private/tmp', '/var/tmp', '/private/var/tmp')
_DISABLED_FEATURES = ('apps', 'plugins', 'browser_use', 'computer_use', 'multi_agent',
                      'memories', 'workspace_dependencies', 'hooks', 'shell_snapshot')
_ORIGIN_KEYS = ('CODEX_THREAD_ID', 'CODEX_SESSION_ID',
                'CODEX_INTERNAL_ORIGINATOR_OVERRIDE', 'CODEX_APP_TOOLS_PIPE_PATH')


def identity():
    return {'version': POLICY_VERSION,
            'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _file(value, label):
    try:
        path = Path(value).resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f'Missing or invalid {label}: {value}') from exc
    if not path.is_file():
        raise ValueError(f'{label} must be a regular file: {path}')
    return path


def material_paths(read_files, image_files=()):
    if not isinstance(read_files, (list, tuple, set)) or not read_files:
        raise ValueError('Isolation requires an explicit nonempty file allowlist')
    paths = sorted({_file(path, 'input') for path in read_files})
    images = {_file(path, 'image attachment') for path in image_files}
    if not images.issubset(paths):
        raise ValueError('Image attachment is not in this call\'s file allowlist')
    return paths


def _workdir(value):
    path = Path(value).resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f'Call work directory is missing: {path}')
    if path in (Path('/'), Path.home().resolve()):
        raise ValueError('Isolation requires a private call work directory')
    if any(path == root or root in path.parents for root in
           {Path(name).resolve() for name in _TEMP_ROOTS}):
        raise ValueError('Call work must be outside shared temporary directories')
    return path


def resolve_pdf_skill(explicit=None):
    explicit = explicit or os.environ.get('CHECK_DIFF_PDF_SKILL')
    if explicit:
        path = Path(explicit)
        return _file(path / 'SKILL.md' if path.is_dir() else path, 'PDF skill')
    home = Path(os.environ.get('CODEX_HOME', str(Path.home() / '.codex')))
    direct = home / 'skills/pdf/SKILL.md'
    if direct.is_file():
        return _file(direct, 'PDF skill')
    runtime = default_tool_runtime()
    if runtime and (runtime / 'pdf/SKILL.md').is_file():
        return _file(runtime / 'pdf/SKILL.md', 'PDF skill')
    choices = [path for path in home.glob('plugins/cache/*/pdf/*/skills/pdf/SKILL.md')
               if path.is_file()]
    if not choices:
        raise ValueError('No PDF skill found; provide CHECK_DIFF_PDF_SKILL')
    return _file(max(choices, key=lambda path: path.stat().st_mtime), 'PDF skill')


def default_tool_runtime(explicit=None):
    if explicit is not None:
        root = Path(explicit).resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f'Invalid tool runtime: {root}')
        return root
    utility = Path(__file__).resolve().parent
    if utility.name == 'codex':
        utility = utility.parent
    adjacent = utility / '.runtime'
    if adjacent.is_dir():
        return adjacent.resolve()
    # Frozen callers receive the project runtime through their validated launcher.
    for name in os.environ.get('PYTHONPATH', '').split(os.pathsep):
        path = Path(name) if name else None
        if (path and path.name == 'python_vendor' and path.parent.name == '.runtime'
                and path.is_dir()):
            return path.parent.resolve()
    return None


def _runtime_paths(codex, pdf_skill, tool_runtime, ripgrep):
    executable = shutil.which(str(codex or 'codex'))
    if executable is None:
        raise ValueError(f'Codex executable not found: {codex}')
    skill = _file(pdf_skill, 'PDF skill')
    paths = {_file(executable, 'Codex executable'), Path(executable).absolute(),
             _file(sys.executable, 'Python executable'),
             Path(sys.base_prefix).resolve(strict=True), skill.parent}
    bundled = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies'
    if bundled.is_dir():
        paths.add(bundled.resolve())
    runtime = default_tool_runtime(tool_runtime)
    if runtime:
        for name in ('python_vendor', 'ghostscript-9.55.0'):
            path = runtime / name
            if path.is_dir():
                paths.add(path.resolve())
    if ripgrep:
        paths.add(_file(ripgrep, 'ripgrep executable'))
    if paths & {Path('/'), Path.home().resolve()}:
        raise ValueError('Runtime grants cannot expose the filesystem root or home directory')
    return sorted(paths)


def _table(values):
    return '{' + ', '.join(f'{json.dumps(str(k))}={json.dumps(v)}'
                            for k, v in values.items()) + '}'


def permission_options(workdir, read_files, pdf_skill, codex=None,
                       tool_runtime=None, ripgrep=None):
    work = _workdir(workdir)
    files = material_paths(read_files)
    filesystem = {':root': 'deny', ':minimal': 'read'}
    for name in _TEMP_ROOTS:
        filesystem[name] = 'deny'
        if sys.platform == 'darwin':
            filesystem[name + '/**'] = 'deny'
            filesystem[name[:-1] + '[p]'] = 'deny'
    for path in _runtime_paths(codex, pdf_skill, tool_runtime, ripgrep):
        if path.is_dir() and (work == path or path in work.parents
                              or any(path == file or path in file.parents for file in files)):
            raise ValueError(f'Runtime grant overlaps call material or work: {path}')
        filesystem[str(path)] = 'read'
    filesystem[str(work)] = 'write'
    for path in files:
        filesystem[str(path)] = 'read'
    return ['-c', f'default_permissions="{PROFILE}"',
            '-c', f'permissions.{PROFILE}.network.enabled=false',
            '-c', f'permissions.{PROFILE}.filesystem=' + _table(filesystem)]


def child_environment(workdir, base_env=None):
    work = _workdir(workdir)
    env = dict(os.environ if base_env is None else base_env)
    for key in _ORIGIN_KEYS:
        env.pop(key, None)
    env.update(TMPDIR=str(work), TMPPREFIX=str(work / 'zsh'), PYTHONNOUSERSITE='1')
    return env


def isolation_options(read_files, workdir, *, codex, pdf_skill, tool_runtime=None,
                      ripgrep=None, image_files=(), base_env=None):
    work = _workdir(workdir)
    files = material_paths(read_files, image_files)
    permissions = permission_options(work, files, pdf_skill, codex, tool_runtime, ripgrep)
    runtime = default_tool_runtime(tool_runtime)
    tool_env = {'PATH': os.pathsep.join(dict.fromkeys((str(Path(sys.executable).resolve().parent),
                                                     '/usr/bin', '/bin'))),
                'TMPDIR': str(work), 'TMPPREFIX': str(work / 'zsh'), 'PYTHONNOUSERSITE': '1'}
    bundled = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies'
    for path in (bundled / 'python/bin', bundled / 'bin/override'):
        if path.is_dir():
            tool_env['PATH'] = str(path) + os.pathsep + tool_env['PATH']
    if runtime:
        if (runtime / 'python_vendor').is_dir():
            tool_env['PYTHONPATH'] = str(runtime / 'python_vendor')
        gs = runtime / 'ghostscript-9.55.0'
        if gs.is_dir():
            tool_env['PATH'] = str(gs / 'bin') + os.pathsep + tool_env['PATH']
            tool_env['GS_LIB'] = os.pathsep.join(str(gs / name) for name in
                ('share/ghostscript/9.55.0/Resource/Init', 'share/ghostscript/9.55.0/lib',
                 'share/ghostscript/9.55.0/Resource/Font'))
    if ripgrep:
        tool_env['PATH'] = str(Path(ripgrep).resolve().parent) + os.pathsep + tool_env['PATH']
    flags = ['--strict-config', '--ignore-user-config', '--ephemeral', '--skip-git-repo-check',
             *permissions, '-c', 'web_search="disabled"', '-c', 'mcp_servers={}',
             *[part for feature in _DISABLED_FEATURES for part in ('-c', f'features.{feature}=false')],
             '-c', 'features.skip_host_skill_discovery=true',
             '-c', 'shell_environment_policy.inherit="none"',
             '-c', 'shell_environment_policy.set=' + _table(tool_env),
             '-c', 'allow_login_shell=false', '-c', 'approval_policy="never"',
             '-c', 'project_doc_max_bytes=0']
    return flags, child_environment(work, base_env)
