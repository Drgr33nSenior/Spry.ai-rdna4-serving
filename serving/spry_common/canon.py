"""Explicit JSON encodings for source and configuration pins."""
import hashlib
import json


def need(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_bare(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def canonical_line(value):
    return canonical_bare(value) + b'\n'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def parse_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('nonfinite JSON value')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
