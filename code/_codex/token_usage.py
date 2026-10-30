"""Account only for token counts reported by completed Codex CLI turns."""
import json


FIELDS = ('input_tokens', 'cached_input_tokens', 'cache_write_input_tokens',
          'output_tokens', 'reasoning_output_tokens')


def read_events(stream):
    records, errors = [], []
    stream.seek(0)
    for line_number, line in enumerate(stream, 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            errors.append(f'Invalid event JSON at line {line_number}')
            continue
        if not isinstance(event, dict) or event.get('type') != 'turn.completed':
            continue
        usage = event.get('usage')
        if not isinstance(usage, dict):
            errors.append(f'Missing turn usage at line {line_number}')
            continue
        invalid = [key for key in FIELDS if key in usage and
                   (type(usage[key]) is not int or usage[key] < 0)]
        if invalid:
            errors.append(f'Invalid token counters at line {line_number}: {invalid}')
            continue
        if (usage.get('cached_input_tokens', 0) > usage.get('input_tokens', float('inf')) or
                usage.get('reasoning_output_tokens', 0) > usage.get('output_tokens', float('inf'))):
            errors.append(f'Token subset exceeds its total at line {line_number}')
            continue
        records.append({'event_line': line_number, 'usage': usage})
    return {'source': 'codex_exec_turn_completed', 'records': records, 'errors': errors}


def summarize(attempts):
    """Sum observed usage; unavailable attempts remain explicit rather than zero."""
    reported = [record['usage'] for attempt in attempts
                for record in attempt.get('token_usage', {}).get('records', [])]
    totals = {key: sum(row[key] for row in reported if key in row)
              if any(key in row for row in reported) else None for key in FIELDS}
    field_coverage = {key: sum(key in row for row in reported) for key in FIELDS}
    total_fields_complete = all(field_coverage[key] == len(reported) and reported
                                for key in ('input_tokens', 'output_tokens'))
    totals['total_tokens'] = (totals['input_tokens'] + totals['output_tokens']
                              if total_fields_complete else None)
    totals['non_cached_input_tokens'] = (
        totals['input_tokens'] - totals['cached_input_tokens']
        if reported and field_coverage['input_tokens'] == field_coverage['cached_input_tokens'] == len(reported)
        else None)
    if not attempts:
        totals = {key: 0 for key in totals}
    unknown = sum(not attempt.get('token_usage', {}).get('records') for attempt in attempts)
    errors = [error for attempt in attempts for error in attempt.get('token_usage', {}).get('errors', [])]
    complete = bool(attempts) and not unknown and not errors and total_fields_complete
    return {'source': 'codex_exec_turn_completed',
            'status': 'not_run' if not attempts else 'complete' if complete else 'partial' if reported else 'unavailable',
            'reported_totals': totals, 'attempt_count': len(attempts),
            'attempts_without_reported_usage': unknown, 'reported_turn_count': len(reported),
            'field_report_counts': field_coverage, 'errors': errors,
            'accounting_note': 'Input totals include cached input; output totals include reasoning output. '
                               'These subsets are not added again. Totals cover reported usage only; '
                               'attempts without usage may still have consumed tokens.'}
