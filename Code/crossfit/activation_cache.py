"""Sharded, fixed-scalar activation caches for crosscoder training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np


def normalization_scalar(sum_norms: float, count: int, dimension: int) -> float:
    """Return the fixed scalar making the mean row norm equal sqrt(d)."""
    if count <= 0 or dimension <= 0 or not np.isfinite(sum_norms) or sum_norms <= 0:
        raise ValueError("normalization inputs must be finite and positive")
    return float(np.sqrt(dimension) / (sum_norms / count))


def normalize_shards_in_place(paths: list[Path], scalar: float) -> float:
    """Scale fp16 NPY shards and return their realized mean row norm."""
    total = 0.0
    count = 0
    for path in paths:
        values = np.load(path, mmap_mode="r+")
        if values.dtype != np.float16 or values.ndim != 3:
            raise ValueError(f"unexpected activation shard: {path} {values.shape} {values.dtype}")
        for start in range(0, len(values), 8):
            stop = min(start + 8, len(values))
            scaled = values[start:stop].astype(np.float32) * float(scalar)
            values[start:stop] = scaled.astype(np.float16)
            total += float(np.linalg.norm(values[start:stop].astype(np.float32), axis=-1).sum())
            count += int(np.prod(values[start:stop].shape[:-1]))
        values.flush()
    return total / count


def shard_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash one cache shard without reading it into memory at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def load_cache_manifest(cache_dir: str | Path) -> dict:
    """Load and minimally validate a completed activation-cache manifest."""
    path = Path(cache_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"completed cache manifest is absent: {path}")
    value = json.loads(path.read_text())
    if value.get("status") != "complete":
        raise ValueError("activation cache is not marked complete")
    return value


def iter_cached_pairs(cache_dir: str | Path) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield aligned normalized evo/fit activation shards."""
    root = Path(cache_dir)
    manifest = load_cache_manifest(root)
    evo = manifest["models"]["evo"]["shards"]
    fit = manifest["models"]["fit"]["shards"]
    if [row["name"] for row in evo] != [row["name"] for row in fit]:
        raise AssertionError("evo and fit cache shard names are not aligned")
    for evo_row, fit_row in zip(evo, fit):
        left = np.load(root / "evo" / evo_row["name"], mmap_mode="r")
        right = np.load(root / "fit" / fit_row["name"], mmap_mode="r")
        if left.shape != right.shape:
            raise AssertionError("evo and fit cache shard shapes differ")
        yield left, right


def load_cached_pair(cache_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Materialize the completed cache as two position-by-channel arrays."""
    pairs = list(iter_cached_pairs(cache_dir))
    if not pairs:
        raise ValueError("activation cache contains no shards")
    evo = np.concatenate([left.reshape(-1, left.shape[-1]) for left, _ in pairs])
    fit = np.concatenate([right.reshape(-1, right.shape[-1]) for _, right in pairs])
    return evo, fit
