#!/usr/bin/env python3
"""Download only the frozen AI-review sample's exact arXiv versions."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = 'PaperTraceSampleDownloader/1.0 (https://huggingface.co/datasets/MisakiWang/PaperTrace)'


def load_manifest(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    seen = set()
    for row in rows:
        key = (row['arxiv_id'], row['version'])
        if not re.fullmatch(r'\d{4}\.\d{4,5}', key[0]) or type(key[1]) is not int or key[1] < 1:
            raise ValueError('Invalid arXiv version in manifest')
        if not re.fullmatch(r'[a-z]+\d{4}', row['conference']) or key in seen:
            raise ValueError('Invalid conference or duplicate manuscript version')
        if not re.fullmatch(r'[0-9a-f]{64}', row.get('original_sha256') or ''):
            raise ValueError('Every frozen version must have an original SHA-256')
        seen.add(key)
    return rows


def validate_file(path, expected_hash):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        prefix = stream.read(1024)
        if not prefix.startswith(b'%PDF-'):
            raise ValueError('The downloaded object is not a PDF')
        digest.update(prefix)
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_hash:
        raise ValueError(f'Frozen-source SHA-256 mismatch: expected {expected_hash}, received {actual}')
    return actual


def download_one(row, destination, opener=urllib.request.urlopen):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return {'status':'cached', 'sha256':validate_file(destination, row['original_sha256'])}
    url = f"https://arxiv.org/pdf/{row['arxiv_id']}v{row['version']}"
    request = urllib.request.Request(url, headers={'User-Agent':USER_AGENT})
    temporary = None
    try:
        with opener(request, timeout=90) as response:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.download-', suffix='.part', delete=False) as target:
                temporary = Path(target.name)
                while True:
                    chunk = response.read(1024*1024)
                    if not chunk:
                        break
                    target.write(chunk)
        actual = validate_file(temporary, row['original_sha256'])
        temporary.replace(destination)
        return {'status':'downloaded', 'sha256':actual}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'data/versions.jsonl.gz')
    parser.add_argument('--output', type=Path, default=ROOT/'downloads_papers')
    parser.add_argument('--paper', action='append', default=[], help='Repeatable arXiv ID filter')
    parser.add_argument('--conference', action='append', default=[], help='Repeatable conference filter')
    parser.add_argument('--limit', type=int, help='Maximum number of version files')
    parser.add_argument('--interval', type=float, default=15.1, help='Request spacing in seconds; minimum 15')
    parser.add_argument('--dry-run', action='store_true', help='List scope; no network requests or file writes')
    args = parser.parse_args(argv)
    if args.interval < 15 or not args.interval < float('inf'):
        parser.error('--interval must be finite and at least 15 seconds')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    rows = load_manifest(args.manifest)
    unknown = set(args.paper) - {r['arxiv_id'] for r in rows}
    unknown_confs = set(args.conference) - {r['conference'] for r in rows}
    if unknown or unknown_confs:
        parser.error('Requested paper or conference is outside the frozen review sample')
    rows = [r for r in rows if (not args.paper or r['arxiv_id'] in args.paper)
            and (not args.conference or r['conference'] in args.conference)]
    rows = sorted(rows, key=lambda r:(r['conference'],r['arxiv_id'],r['version']))
    if args.limit is not None:
        rows = rows[:args.limit]
    output = args.output.resolve()
    if args.dry_run:
        print(json.dumps({'papers':len({r['arxiv_id'] for r in rows}), 'versions':len(rows),
                          'output':str(output), 'network_requests':0,
                          'version_ids':[f"{r['arxiv_id']}v{r['version']}" for r in rows]}, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    report = {'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'started_at_utc':datetime.now(timezone.utc).isoformat(), 'requested_versions':len(rows), 'files':[]}
    last_request = None
    for row in rows:
        destination = output/row['conference']/row['arxiv_id']/f"v{row['version']}.pdf"
        if not destination.resolve().is_relative_to(output):
            raise ValueError('Download destination escapes the requested output directory')
        entry = {'arxiv_id':row['arxiv_id'], 'version':row['version'], 'path':str(destination.relative_to(output))}
        try:
            if not destination.exists():
                if last_request is not None:
                    time.sleep(max(0,args.interval-(time.monotonic()-last_request)))
                last_request = time.monotonic()
            entry.update(download_one(row, destination))
        except (OSError, ValueError) as exc:
            entry.update(status='failed', error=str(exc), http_status=getattr(exc,'code',None))
        report['files'].append(entry)
        report['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        temporary = output/'download_manifest.json.tmp'
        temporary.write_text(json.dumps(report,indent=2)+'\n')
        temporary.replace(output/'download_manifest.json')
        print(json.dumps(entry), flush=True)
        if entry.get('http_status') in (403,429):
            break
    return 0 if len(report['files'])==len(rows) and all(r['status']!='failed' for r in report['files']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
