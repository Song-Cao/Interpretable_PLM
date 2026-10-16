"""Readers for ProteinGym 2.0 ``.pgdata`` bundles.

A ``.pgdata`` file is a zip containing:
    dataset.pgdata   -- itself a zip: manifest.lock (TOML), assays/*.csv,
                        sequences/*.fasta, structures/*.pdb, msas/*.a3m
    slices.json      -- published split definitions as per-record boolean masks

Nothing in this module interprets or filters the data; it only decodes it.
All QC decisions live in ``crossfit.qc``.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:  # py311+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


@dataclass
class Bundle:
    """Decoded contents of one ``.pgdata`` file."""

    assay_id: str
    path: Path
    table: pd.DataFrame
    wt_sequence: str
    manifest: dict[str, Any]
    slices: dict[str, Any]
    pdb_bytes: bytes | None
    member_names: list[str]

    @property
    def assay_meta(self) -> dict[str, Any]:
        return self.manifest["assays"][0]

    @property
    def publication(self) -> dict[str, Any]:
        return self.manifest.get("publication", {})


def _one(names: list[str], prefix: str = "", suffix: str = "") -> str | None:
    hits = [n for n in names if n.startswith(prefix) and n.endswith(suffix)]
    return hits[0] if hits else None


def read_bundle(path: str | Path, want_structure: bool = False) -> Bundle:
    path = Path(path)
    outer = zipfile.ZipFile(path)
    inner = zipfile.ZipFile(io.BytesIO(outer.read("dataset.pgdata")))
    names = inner.namelist()

    csv_name = _one(names, prefix="assays/")
    if csv_name is None:
        raise ValueError(f"{path.name}: no assays/*.csv member")
    table = pd.read_csv(io.BytesIO(inner.read(csv_name)))

    fasta_name = _one(names, suffix=".fasta")
    if fasta_name is None:
        raise ValueError(f"{path.name}: no sequences/*.fasta member")
    fasta = inner.read(fasta_name).decode()
    # Multi-line FASTA: join every non-header line. Reading only line 1 silently
    # truncates the reference and makes every position check spuriously fail.
    wt = "".join(l.strip() for l in fasta.splitlines() if not l.startswith(">"))

    manifest = tomllib.loads(inner.read("manifest.lock").decode())
    slices = json.loads(outer.read("slices.json"))

    pdb = None
    if want_structure:
        pdb_name = _one(names, prefix="structures/")
        if pdb_name is not None:
            pdb = inner.read(pdb_name)

    return Bundle(
        assay_id=path.stem,
        path=path,
        table=table,
        wt_sequence=wt,
        manifest=manifest,
        slices=slices,
        pdb_bytes=pdb,
        member_names=names,
    )


def published_folds(bundle: Bundle) -> dict[str, list[list[bool]]]:
    """Extract published split masks, keyed by scheme name.

    Each mask is a list of per-record booleans. NOTE: the mask length is the
    number of rows in the *source* assay table, which for bundles containing
    duplicate sequences is longer than the deduplicated table. Callers must
    align masks to the pre-dedup row order.
    """
    out: dict[str, list[list[bool]]] = {}
    for scheme, folds in bundle.slices.items():
        masks = []
        for f in folds:
            masks.append(list(f["assays"][0]["records"]))
        out[scheme] = masks
    return out
