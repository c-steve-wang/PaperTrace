"""PDF input and result persistence shared by the OpenRouter review entry point."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                     prefix='.' + path.stem, delete=False) as f:
        tmp = Path(f.name)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


@dataclass(frozen=True)
class Version:
    paper_id: str
    number: int
    sources: tuple

    def read(self):
        """One read per unique inode; verify non-hardlinked duplicate copies too."""
        seen = set()
        data = None
        checksum = None
        for path in self.sources:
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            current = path.read_bytes()
            current_hash = sha(current)
            if data is None:
                if not current.startswith(b'%PDF-'):
                    raise ValueError(f'Not a PDF: {path}')
                data, checksum = current, current_hash
            elif current_hash != checksum:
                raise ValueError(f'Conflicting copies of {self.paper_id}/v{self.number}: {self.sources}')
        return data, checksum


def read_document(path, aid, source_id=None):
    if not path.exists():
        document = {'format_version': 1, 'paper_id': aid, 'versions': {}}
    else:
        document = json.loads(path.read_text())
        if document.get('format_version') != 1 or document.get('paper_id') != aid:
            raise ValueError(f'Incompatible result file: {path}')
    if source_id is not None:
        saved_source = document.get('source_id')
        has_reviews = any(v.get('reviews') for v in document['versions'].values())
        if saved_source not in (None, source_id) or (saved_source is None and has_reviews):
            raise ValueError(f'Result source is missing or differs for {aid}; use a fresh output directory')
        document['source_id'] = source_id
    return document


def validate_output(input_root, output, papers=()):
    """Keep resumable result paths separate from inputs and protected project files."""
    output = output.absolute()
    for path in (output, *output.parents):
        if path.is_symlink():
            raise ValueError(f'Output path must not traverse a symlink: {path}')
        if path.exists() and not path.is_dir():
            raise ValueError(f'Output path must be a directory: {path}')
    output = output.resolve()
    if sys.platform == 'darwin' and any(
            output == root or root in output.parents
            for root in (Path('/private/tmp'), Path('/private/var/tmp'))):
        raise ValueError('Output must be outside shared temporary directories on macOS; '
                         'use a dedicated project res/ directory')
    input_root = input_root.resolve()
    if output == input_root or output in input_root.parents or input_root in output.parents:
        raise ValueError('Input and output directories must not overlap')
    project = ROOT.resolve()
    protected = [project / name for name in
                 ('review', 'code', 'data', 'downloads_papers', 'clean_data', '.git')]
    for path in protected:
        if output == path or output in path.parents or path in output.parents:
            raise ValueError(f'Output overlaps a protected project path: {path}')
    targets = [(output / '.runner.lock', False), (output / 'run_summary.json', False)]
    for aid in papers:
        targets.extend([(output / aid, True), (output / aid / 'reviews.json', False)])
    for path, directory in targets:
        if path.is_symlink():
            raise ValueError(f'Output target must not be a symlink: {path}')
        if path.exists() and not (path.is_dir() if directory else path.is_file()):
            raise ValueError(f'Output target must be a {"directory" if directory else "file"}: {path}')
    return output
