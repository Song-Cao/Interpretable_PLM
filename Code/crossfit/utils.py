"""Shared helpers: config loading, path resolution, hashing, determinism."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import yaml

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2]  # Code/crossfit/utils.py -> repo root
CONFIG_PATH = REPO_ROOT / "Code" / "config" / "step1.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    with open(path or CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"]["root"] = str(REPO_ROOT)
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    """Resolve a key under cfg['paths'] to an absolute Path."""
    return REPO_ROOT / cfg["paths"][key]


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def stable_hash(*parts: object) -> int:
    """Deterministic 63-bit integer from arbitrary parts.

    Used to derive per-(assay, scheme, seed) RNG seeds so that splits are a pure
    function of their identifiers and never depend on iteration order.
    """
    s = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16) >> 1
