"""Stable fingerprints used to prevent stale results from being resumed."""

import hashlib
import json


def fingerprint(value):
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
