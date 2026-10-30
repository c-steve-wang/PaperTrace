"""Frozen venue profiles and bounded, evidence-based source resolution."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re


def _rating(value, kind):
    if kind == 'integer':
        return type(value) is int
    if kind == 'number':
        return type(value) is int or (type(value) is float and math.isfinite(value))
    return kind == 'string' and type(value) is str and bool(value.strip())


def _scale(kind, values):
    if (kind not in ('integer', 'number', 'string') or
            not isinstance(values, (list, tuple)) or not values or
            any(not _rating(v, kind) for v in values) or
            len(set(values)) != len(values)):
        raise ValueError('Invalid OA type or enum values')


def _schema(kind, values):
    _scale(kind, values)
    return {'type': 'object', 'properties': {
        'oa': {'type': kind, 'enum': list(values)},
        'strengths': {'type': 'array', 'items': {'type': 'string'}},
        'weaknesses': {'type': 'array', 'items': {'type': 'string'}},
    }, 'required': ['oa', 'strengths', 'weaknesses'], 'additionalProperties': False}


@dataclass(frozen=True)
class SystemProfile:
    source_id: str
    system_file: str
    text: str
    sha256: str
    oa_type: str
    oa_values: tuple
    scale_kind: str = 'native'
    oa_labels: tuple = ()

    def __post_init__(self):
        _scale(self.oa_type, self.oa_values)
        object.__setattr__(self, 'oa_values', tuple(self.oa_values))
        if self.scale_kind not in ('native', 'experimental_encoding'):
            raise ValueError('Invalid scale_kind')
        if (not isinstance(self.oa_labels, (tuple, list)) or
                (self.oa_labels and len(self.oa_labels) != len(self.oa_values)) or
                any(not isinstance(label, str) or not label.strip() for label in self.oa_labels)):
            raise ValueError('OA labels must be nonempty strings parallel to OA values')
        object.__setattr__(self, 'oa_labels', tuple(self.oa_labels))

    @property
    def schema(self):
        return _schema(self.oa_type, self.oa_values)

    def metadata(self):
        return {'source_id': self.source_id, 'system_file': self.system_file,
                'sha256': self.sha256, 'oa_type': self.oa_type,
                'oa_values': list(self.oa_values), 'scale_kind': self.scale_kind,
                'oa_labels': list(self.oa_labels)}


def _safe_path(root, name):
    if (not isinstance(name, str) or not name or '\\' in name or
            PurePosixPath(name).is_absolute() or
            any(part in ('', '.', '..') for part in name.split('/'))):
        raise ValueError(f'Invalid relative path: {name!r}')
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Missing file or path escape: {name}')
    return path


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _json(data, path):
    try:
        value = json.loads(data.decode('utf-8'))
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f'Invalid UTF-8 JSON: {path}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


def _checked_bytes(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise ValueError(f'Invalid SHA256: {path}')
    data = path.read_bytes()
    if _digest(data) != expected:
        raise ValueError(f'Checksum mismatch: {path}')
    return data


def _source(value, known=None):
    if not isinstance(value, str) or not re.fullmatch(r'[a-z0-9][a-z0-9_]*', value):
        raise ValueError(f'Invalid source ID: {value!r}')
    if known is not None and value not in known:
        raise ValueError(f'Unknown source: {value}')
    return value


def load_prompts(folder: Path, experiments=None, conditions=None, *, with_manifest=False):
    folder = Path(folder)
    manifest_path = _safe_path(folder, 'manifest.json')
    manifest = _json(manifest_path.read_bytes(), manifest_path)
    files, systems, available = (manifest.get(k) for k in ('files', 'systems', 'conditions'))
    if not isinstance(files, dict) or not files or not isinstance(systems, dict) or not systems:
        raise ValueError('Manifest requires nonempty files and systems maps')
    texts = {}
    for name, expected in files.items():
        data = _checked_bytes(_safe_path(folder, name), expected)
        try:
            texts[name] = data.decode('utf-8')
        except UnicodeError as exc:
            raise ValueError(f'Invalid UTF-8 prompt: {name}') from exc
    profiles = {}
    for source_id, config in systems.items():
        _source(source_id)
        if not isinstance(config, dict):
            raise ValueError(f'Invalid system profile: {source_id}')
        name = config.get('file')
        if not isinstance(name, str) or name not in texts or not texts[name].strip():
            raise ValueError(f'Missing or empty system prompt: {source_id}')
        _scale(config.get('oa_type'), config.get('oa_values'))
        if 'oa_labels' in config and (not isinstance(config['oa_labels'], list) or
                                      len(config['oa_labels']) != len(config['oa_values'])):
            raise ValueError(f'OA label count mismatch: {source_id}')
        profiles[source_id] = SystemProfile(source_id, name, texts[name], files[name],
                                            config['oa_type'], tuple(config['oa_values']),
                                            config.get('scale_kind', 'native'),
                                            config.get('oa_labels', ()))
    if not isinstance(available, list) or not available:
        raise ValueError('Manifest requires nonempty conditions')
    ids = set()
    for item in available:
        if not isinstance(item, dict):
            raise ValueError('Invalid condition')
        cid, experiment, name = (item.get(k) for k in ('id', 'experiment', 'user_file'))
        if (not isinstance(cid, str) or not cid.strip() or cid in ids or
                not isinstance(experiment, str) or not experiment.strip() or
                not isinstance(name, str) or name not in texts or not texts[name].strip()):
            raise ValueError('Invalid, duplicate, or missing condition')
        shared = item.get('shared_baseline_for', [])
        if not isinstance(shared, list) or any(
                not isinstance(value, str) or not value.strip() for value in shared):
            raise ValueError('shared_baseline_for must be a list of nonempty experiment strings')
        ids.add(cid)
    if conditions and set(conditions) - ids:
        raise ValueError(f'Unknown condition IDs: {sorted(set(conditions) - ids)}')
    selected = [(c['id'], texts[c['user_file']]) for c in available
                if (not conditions or c['id'] in conditions)
                and (not experiments or c['experiment'] in experiments or
                     any(value in experiments for value in c.get('shared_baseline_for', [])))]
    if not selected:
        raise ValueError('No conditions selected')
    return (profiles, selected, manifest) if with_manifest else (profiles, selected)


def validate_result(result, schema):
    try:
        oa = schema['properties']['oa']
        expected = _schema(oa['type'], oa['enum'])
        if schema != expected:
            raise ValueError('Unexpected review schema')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('Invalid review schema') from exc
    if (not isinstance(result, dict) or set(result) != {'oa', 'strengths', 'weaknesses'} or
            not _rating(result['oa'], oa['type']) or result['oa'] not in oa['enum']):
        raise ValueError('Expected exactly oa, strengths, weaknesses with a valid OA')
    for key in ('strengths', 'weaknesses'):
        if not isinstance(result[key], list) or any(
                not isinstance(x, str) or not x.strip() for x in result[key]):
            raise ValueError(f'{key} must be a list of nonempty strings')
    return result
