#!/usr/bin/env python3
"""Independent single-PDF reviews through OpenRouter; no Codex/model conversation reuse."""
import argparse
import base64
import copy
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

if __package__:
    from .source_profiles import load_prompts, validate_result
    from .io_utils import Version, read_document, save_json, validate_output
else:
    from source_profiles import load_prompts, validate_result
    from io_utils import Version, read_document, save_json, validate_output

ROOT = Path(__file__).resolve().parents[1]
API = 'https://openrouter.ai/api/v1'
DEFAULT_INPUT = ROOT / 'clean_data'
ADAPTER_VERSION = 1


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def json_hash(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def redact(value, secret):
    """Do not persist credentials even when an upstream error echoes a request."""
    if isinstance(value, str):
        value = value.replace(secret, '[REDACTED]') if secret else value
        return re.sub(r'sk-or-v1-[A-Za-z0-9_-]+', '[REDACTED]', value)
    if isinstance(value, list):
        return [redact(v, secret) for v in value]
    if isinstance(value, dict):
        return {k: redact(v, secret) for k, v in value.items()}
    return value


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class APIError(RuntimeError):
    def __init__(self, status, body, retry_after=None):
        super().__init__(f'OpenRouter HTTP {status}: {str(body)[:2000]}')
        self.status, self.body, self.retry_after = status, body, retry_after


def api_json(path, key=None, payload=None, timeout=180):
    headers = {'Content-Type': 'application/json', 'X-OpenRouter-Title': 'PaperTrace'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    request = Request(API + path, data=None if payload is None else json.dumps(payload).encode(), headers=headers)
    try:
        with build_opener(NoRedirect()).open(request, timeout=timeout) as response:
            return json.loads(response.read())
    except HTTPError as exc:
        body = redact(exc.read().decode('utf-8', errors='replace'), key)
        try:
            retry_after = float(exc.headers.get('Retry-After', ''))
        except ValueError:
            retry_after = None
        raise APIError(exc.code, body, retry_after) from None


def check_model(model, timeout):
    models = api_json('/models', timeout=timeout)['data']
    found = next((m for m in models if m['id'] == model), None)
    if found is None:
        raise ValueError(f'Model ID is not in the OpenRouter catalog: {model}')
    if 'file' not in found.get('architecture', {}).get('input_modalities', []):
        raise ValueError(f'{model} does not advertise native file input; no automatic OCR fallback')
    if 'structured_outputs' not in found.get('supported_parameters', []):
        raise ValueError(f'{model} does not advertise structured outputs')
    return found


def discover(args, profiles):
    root = args.input.resolve()
    if not root.exists():
        raise ValueError(f'Input does not exist: {root}')
    paths = [root] if root.is_file() else sorted(
        p for p in root.rglob('*.pdf') if p.is_file() and not p.name.startswith('._'))
    grouped, sources = {}, {}
    for path in paths:
        match = re.fullmatch(r'(?:(\d{4}\.\d{4,5}))?v([1-9]\d*)\.pdf', path.name)
        if not match:
            raise ValueError(f'Unrecognized PDF filename: {path}')
        aid = match[1] or next((p.name for p in path.parents if re.fullmatch(r'\d{4}\.\d{4,5}', p.name)), None)
        if not aid:
            raise ValueError(f'Cannot infer paper ID: {path}')
        if args.paper and aid not in args.paper:
            continue
        parent_source = next((p.name for p in path.parents if p.name in profiles), None)
        candidates = {s for s in (args.source, parent_source) if s}
        if len(candidates) != 1:
            raise ValueError(f'Missing/ambiguous conference for {path}; use --source')
        source = candidates.pop()
        if source not in profiles:
            raise ValueError(f'No prompt profile for {source}')
        if args.conf and source not in args.conf:
            continue
        if aid in sources and sources[aid] != source:
            raise ValueError(f'{aid} belongs to multiple conferences; select one using --conf')
        sources[aid] = source
        grouped.setdefault(aid, {}).setdefault(int(match[2]), []).append(path)
    if args.conf and set(args.conf) - set(profiles):
        raise ValueError(f'Unknown conferences: {set(args.conf) - set(profiles)}')
    if args.paper and set(args.paper) - set(grouped):
        raise ValueError(f'Paper not found in selected scope: {set(args.paper) - set(grouped)}')
    selected = {}
    for aid in sorted(grouped)[:args.limit_papers]:
        versions = grouped[aid]
        if args.version and set(args.version) - set(versions):
            raise ValueError(f'{aid}: requested versions missing')
        numbers = [max(versions)] if args.latest_only else sorted(versions)
        selected[aid] = [Version(aid, n, tuple(versions[n])) for n in numbers
                         if not args.version or n in args.version]
    if not selected:
        raise ValueError('No PDFs selected')
    return selected, {aid: sources[aid] for aid in selected}


def payload_for(args, profile, template, data):
    # Keep prompt bytes except the existing manuscript placeholder. Never send local paths/IDs.
    payload = {'model': args.model, 'stream': False,
        'messages': [{'role': 'system', 'content': profile.text},
                     {'role': 'user', 'content': [
                         {'type': 'text', 'text': template.replace('{PAPER_CONTENT}', 'manuscript.pdf')},
                         {'type': 'file', 'file': {'filename': 'manuscript.pdf',
                            'file_data': 'data:application/pdf;base64,' + base64.b64encode(data).decode()}}]}],
        'plugins': [{'id': 'file-parser', 'pdf': {'engine': 'native'}}],
        'provider': {'require_parameters': True}, 'max_tokens': args.max_tokens,
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'paper_review', 'strict': True, 'schema': profile.schema}}}
    if args.response_format == 'json_object':
        payload['response_format'] = {'type': 'json_object'}
    if args.effort:
        payload['reasoning'] = {'effort': args.effort}
    if args.temperature is not None:
        payload['temperature'] = args.temperature
    return payload


def safe_request(payload, checksum):
    result = copy.deepcopy(payload)
    result['messages'][1]['content'][1]['file']['file_data'] = f'[PDF base64 omitted; sha256={checksum}]'
    return result


def decode_response(response, profile):
    if response.get('error'):
        raise ValueError(f'OpenRouter error: {response["error"]}')
    choice = response['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError(f'Incomplete/nonstandard finish_reason: {choice.get("finish_reason")}')
    message = choice['message']
    if message.get('refusal'):
        raise ValueError(f'Provider refusal: {message["refusal"]}')
    text = message.get('content')
    if not isinstance(text, str):
        raise ValueError('Expected JSON text in assistant message')
    return validate_result(json.loads(text), profile.schema)


def usage_summary(attempts):
    result = {'attempts': len(attempts), 'attempts_with_usage': 0, 'attempts_without_usage': 0}
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cost'):
        values = [a['usage'][key] for a in attempts if isinstance(a.get('usage'), dict)
                  and isinstance(a['usage'].get(key), (int, float))]
        result[key] = sum(values) if values else None
        result[key + '_missing_attempts'] = len(attempts) - len(values)
    result['attempts_with_usage'] = sum(isinstance(a.get('usage'), dict) for a in attempts)
    result['attempts_without_usage'] = len(attempts) - result['attempts_with_usage']
    return result


def review_cell(args, aid, version, profile, condition, template, data, checksum, document, key, config, transport):
    path = args.output / aid / 'reviews.json'
    record = document['versions'][f'v{version.number}']
    cell_id = f'{condition}/r1'
    payload = payload_for(args, profile, template, data)
    request_record = safe_request(payload, checksum)
    mark = json_hash({'request': request_record, 'config': config, 'adapter_version': ADAPTER_VERSION})
    previous = record['reviews'].get(cell_id)
    if previous and previous.get('fingerprint') != mark:
        raise ValueError(f'Changed request for {aid}/v{version.number}/{cell_id}; use a fresh output directory')
    if previous and previous.get('status') == 'completed':
        validate_result(previous['result'], profile.schema)
        return 'skipped'
    review = previous or {'condition': condition, 'replicate': 1, 'fingerprint': mark,
        'source_id': profile.source_id, 'system_prompt': profile.metadata(), 'model': args.model,
        'effort': args.effort, 'transport': 'openrouter', 'status': 'failed', 'attempts': [],
        'paper_id': aid, 'version_ids': [f'v{version.number}'], 'version_pair_id': None,
        'content_hashes': {f'v{version.number}': checksum},
        'condition_metadata': next(c for c in config['prompt_manifest']['conditions'] if c['id'] == condition),
        'prompt_version': config['prompt_manifest'].get('design_version')}
    record['reviews'][cell_id] = review
    fatal = False
    for retry in range(args.retries + 1):
        number = len(review['attempts']) + 1
        relative = Path('attempts') / aid / f'v{version.number}' / condition / f'attempt-{number:03d}'
        folder = args.output / relative
        attempt = {'started_at': now(), 'status': 'in_progress', 'artifacts': str(relative), 'usage': None}
        review['attempts'].append(attempt)
        save_json(folder / 'request.json', request_record)
        save_json(path, document)
        print(f'START {aid}/v{version.number} {condition} attempt={number}', flush=True)
        started, retryable, delay = time.monotonic(), False, 2 ** retry
        try:
            response = redact(transport('/chat/completions', key=key, payload=payload, timeout=args.timeout), key)
            save_json(folder / 'response.json', response)
            attempt['usage'] = response.get('usage')
            attempt['response_id'] = response.get('id')
            attempt['actual_model'] = response.get('model')
            attempt['provider'] = response.get('provider')
            review['result'] = decode_response(response, profile)
            review['status'] = attempt['status'] = 'completed'
            print(f'OK {aid}/v{version.number} {condition} OA={review["result"]["oa"]}', flush=True)
        except (APIError, URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError) as exc:
            attempt['status'] = review['status'] = 'failed'
            attempt['error'] = redact(f'{type(exc).__name__}: {exc}', key)
            if isinstance(exc, APIError):
                attempt['http_status'] = exc.status
                save_json(folder / 'error.json', redact({'status': exc.status, 'body': exc.body}, key))
                retryable = exc.status in (408, 429, 500, 502, 503, 504)
                delay = max(delay, exc.retry_after or 0)
                fatal = exc.status in (401, 402, 403)
            else:
                retryable = isinstance(exc, (URLError, TimeoutError))
            print(f'FAILED {aid}/v{version.number} {condition}: {attempt["error"][:300]}', flush=True)
        finally:
            attempt.update(finished_at=now(), elapsed_seconds=round(time.monotonic() - started, 3))
            review['token_usage'] = usage_summary(review['attempts'])
            document['updated_at'] = now()
            save_json(folder / 'receipt.json', attempt)
            save_json(path, document)
        if review['status'] == 'completed' or not retryable or retry == args.retries or delay > 60:
            break
        time.sleep(delay)
    return 'fatal' if fatal else review['status']


def execute(args, profiles, conditions, manifest, papers, sources, key, transport=api_json):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another runner owns this output directory') from None
        config = {'adapter': 'openrouter', 'adapter_version': ADAPTER_VERSION, 'model': args.model,
            'endpoint': API, 'effort': args.effort, 'temperature': args.temperature,
            'max_tokens': args.max_tokens, 'pdf_engine': 'native', 'order_seed': args.order_seed,
            'response_format': args.response_format,
            'prompt_manifest': manifest, 'script_sha256': digest(Path(__file__).read_bytes())}
        config_path = args.output / 'run_config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError('Run configuration changed; use a fresh output directory')
        save_json(config_path, config)
        counts = {'completed': 0, 'skipped': 0, 'failed': 0}
        summary = {'transport': 'openrouter', 'model': args.model, 'started_at': now(),
            'input_root': str(args.input.resolve()), 'prompt_root': str(args.prompts.resolve()),
            'papers': len(papers), 'unique_versions': sum(map(len, papers.values())),
            'conditions': [cid for cid, _ in conditions], 'counts': counts, 'status': 'in_progress'}
        selected_attempts = []
        save_json(args.output / 'run_summary.json', summary)
        fatal = False
        try:
            for aid, versions in papers.items():
                path = args.output / aid / 'reviews.json'
                document = read_document(path, aid, sources[aid])
                for version in versions:
                    data, checksum = version.read()
                    record = document['versions'].setdefault(f'v{version.number}', {'reviews': {}})
                    if record.get('sha256') not in (None, checksum):
                        raise ValueError('PDF changed; use a fresh output directory')
                    record.update(sha256=checksum, source_paths=[str(p) for p in version.sources])
                    ordered = list(conditions)
                    random.Random(f'{args.order_seed}/{aid}/{version.number}').shuffle(ordered)
                    record['condition_orders'] = {'r1': {'conditions': [c for c,_ in ordered], 'seed': args.order_seed}}
                    for condition, template in ordered:
                        status = review_cell(args, aid, version, profiles[sources[aid]], condition,
                                             template, data, checksum, document, key, config, transport)
                        counts['failed' if status == 'fatal' else status] += 1
                        selected_attempts.extend(record['reviews'][f'{condition}/r1']['attempts'])
                        summary['token_usage'] = usage_summary(selected_attempts)
                        save_json(args.output / 'run_summary.json', summary)
                        if status == 'fatal':
                            fatal = True
                            break
                    if fatal:
                        break
                selected_cells = [document['versions'].get(f'v{v.number}', {}).get('reviews', {}).get(f'{c}/r1', {})
                                  for v in versions for c, _ in conditions]
                document['latest_selection_status'] = ('completed' if all(
                    r.get('status') == 'completed' for r in selected_cells) else 'failed')
                save_json(path, document)
                if fatal:
                    break
            summary['status'] = 'failed' if counts['failed'] else 'completed'
        except BaseException:
            summary['status'] = 'interrupted'
            raise
        finally:
            summary.update(finished_at=now(), token_usage=usage_summary(selected_attempts))
            save_json(args.output / 'run_summary.json', summary)
        return 1 if counts['failed'] else 0


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', nargs='?', type=Path, default=DEFAULT_INPUT)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--prompts', type=Path, default=Path(__file__).resolve().parent / 'prompts')
    ap.add_argument('--model', required=True, help='Explicit OpenRouter API model ID for this new run')
    ap.add_argument('--source', help='Conference for a flat directory/single PDF')
    ap.add_argument('--conf', action='append')
    ap.add_argument('--paper', action='append')
    ap.add_argument('--limit-papers', type=int)
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument('--version', type=int, action='append')
    scope.add_argument('--latest-only', action='store_true')
    ap.add_argument('--condition', action='append')
    ap.add_argument('--experiment', choices=['1', '3'], action='append')
    ap.add_argument('--effort', choices=['none', 'minimal', 'low', 'medium', 'high'], default='medium')
    ap.add_argument('--max-tokens', type=int, default=8192)
    ap.add_argument('--response-format', choices=['json_schema', 'json_object'], default='json_schema',
                    help='API output format; both modes retain strict local review validation')
    ap.add_argument('--temperature', type=float)
    ap.add_argument('--order-seed', type=int, default=0)
    ap.add_argument('--timeout', type=float, default=180)
    ap.add_argument('--retries', type=int, default=1, help='Retries for transient transport/HTTP failures only')
    ap.add_argument('--ask-api-key', action='store_true', help='Read key with hidden terminal input instead of environment')
    ap.add_argument('--dry-run', action='store_true', help='Inventory only: no network, key, PDF upload, or output writes')
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if (args.limit_papers is not None and args.limit_papers < 1 or args.retries < 0
                or args.max_tokens < 1 or not math.isfinite(args.timeout) or args.timeout <= 0
                or args.version and min(args.version) < 1
                or args.temperature is not None and (not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2)):
            raise ValueError('Invalid limit, retries, timeout, version, temperature, or output budget')
        profiles, conditions, manifest = load_prompts(args.prompts, args.experiment, args.condition, with_manifest=True)
        papers, sources = discover(args, profiles)
        args.output = validate_output(args.input, args.output, papers)
        if args.output.exists() and any(p.is_symlink() for p in args.output.rglob('*')):
            raise ValueError('Output must not contain symlinks')
        if args.dry_run:
            print(json.dumps({'model': args.model, 'papers': len(papers),
                'unique_versions': sum(map(len, papers.values())), 'source_by_paper': sources,
                'versions_by_paper': {aid: [v.number for v in vs] for aid, vs in papers.items()},
                'conditions': [c for c,_ in conditions], 'planned_reviews': sum(map(len,papers.values())) * len(conditions),
                'input_root': str(args.input.resolve()), 'output_root': str(args.output),
                'prompts': str(args.prompts.resolve()), 'model_calls_performed': False}, indent=2))
            return 0
        key = getpass.getpass('OpenRouter API key (hidden): ') if args.ask_api_key else os.environ.get('OPENROUTER_API_KEY', '')
        if not key.strip():
            raise ValueError('Set OPENROUTER_API_KEY or use --ask-api-key')
        model = check_model(args.model, args.timeout)
        if args.effort and 'reasoning' not in model.get('supported_parameters', []):
            raise ValueError('Model does not advertise reasoning controls')
        return execute(args, profiles, conditions, manifest, papers, sources, key.strip())
    except (ValueError, OSError, APIError) as exc:
        print(redact(str(exc), locals().get('key')), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
