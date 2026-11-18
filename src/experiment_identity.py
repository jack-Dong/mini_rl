"""Stable identity for completed experiment results reused by --resume."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def experiment_fingerprint(config: dict[str, Any], source_files: list[Path]) -> str:
    """Hash experiment configuration and the actual source bytes, not file locations."""
    digest = hashlib.sha256()
    digest.update(json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode())
    for path in source_files:
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()
