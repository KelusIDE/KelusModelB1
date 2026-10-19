"""Requested languages and checkpoint-bound verification; no capability claims."""
import hashlib
from functools import lru_cache
import json
from pathlib import Path

SCOPE_PATH = Path(__file__).resolve().parent.parent / 'config/languages.json'


def scope():
    return json.loads(SCOPE_PATH.read_text(encoding='utf-8'))


def scope_sha256():
    return hashlib.sha256(SCOPE_PATH.read_bytes()).hexdigest()


def canonical(value, category):
    if not isinstance(value, str) or category not in ('programming', 'chat'):
        raise ValueError('Expected a language name and programming/chat category')
    return _canonical(value.strip().casefold(), category)


@lru_cache(maxsize=256)
def _canonical(value, category):
    for key, entry in scope()[category].items():
        if value in {x.casefold() for x in [key, entry['name'], *entry['aliases']]}:
            return key
    raise ValueError(f'Unknown or ambiguous {category} language: {value!r}')


def verification_problem(report, checkpoint_sha256):
    """Review record must cover the current scope and this exact checkpoint.

    This validates evidence completeness, not its truth. Only populate a record
    after checking real generated outputs. Syntax checks alone do not qualify.
    """
    targets = scope()
    if report.get('checkpoint_sha256') != checkpoint_sha256 or report.get('scope_sha256') != scope_sha256():
        return 'Language verification belongs to another checkpoint or target list.'
    for category, checks_key in [('programming', 'code_checks'), ('chat', 'chat_checks')]:
        entries = report.get(category)
        if not isinstance(entries, dict):
            return f'Missing {category} language verification.'
        for language in targets[category]:
            evidence = entries.get(language, {})
            if not isinstance(evidence, dict):
                return f'Invalid language verification for {language}.'
            if not evidence.get('reviewer') or not evidence.get('evidence'):
                return f'Missing review evidence for {language}.'
            checks = evidence.get('checks', {})
            if not isinstance(checks, dict) or any(checks.get(name) is not True for name in targets[checks_key]):
                return f'Language checks have not passed for {language}.'
    return None


def verification_template(checkpoint_sha256):
    targets = scope()
    result = {'checkpoint_sha256': checkpoint_sha256, 'scope_sha256': scope_sha256(),
              'note': 'Unverified template. Fill only after real output and behavior checks. Evidence should identify prompts, outputs, toolchain/dialect, results and reviewer.'}
    for category, checks_key in [('programming', 'code_checks'), ('chat', 'chat_checks')]:
        result[category] = {language: {'reviewer': '', 'evidence': '',
                                      'checks': {check: False for check in targets[checks_key]}}
                            for language in targets[category]}
    return result
