#!/usr/bin/env python3
"""Build the unified CROSSFIT variant table, splits, and QC report.

Outputs (all under Data/processed/):
    variants.csv          one row per unique mutant sequence, all assays
    variants.parquet      same content, typed, for fast loading
    assay_manifest.csv    one row per assay: provenance, QC verdict, counts
    splits/<scheme>.csv   long-form split assignments
    qc_report.csv         full per-assay QC record
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit import labels, qc as qcmod, splits as splitmod
from crossfit.pgdata import published_folds, read_bundle
from crossfit.qc import apply_mutations as labels_apply
from crossfit.selection import SELECTION
from crossfit.utils import load_config, resolve

# The variant table is deliberately NARROW. Per-assay constants (wt_sequence,
# organism, phenotype_class, ...) live once in assay_manifest.csv and are joined
# on assay_id by the loader. Carrying the 448-aa WT string on all 537k GB1 rows
# inflated the CSV to 655 MB for no information gain.
#
# `mutant_sequence` is also NOT stored: it is exactly
# apply_mutations(wt_sequence, mutations), verified byte-for-byte for all
# 633,984 variants at build time and re-verified by the loader on read. Storing
# it would triple the file size and create a second copy that can drift from
# the mutation string.
SCHEMA = [
    "assay_id",
    "variant_row",
    "mutations",
    "n_mutations",
    "mutated_positions",
    "y_raw",
    "y",
    "sigma_v",
    "sigma_source",
    "n_measurements",
    "censored",
    "dup_disagrees",
    "is_layer_freeze_reserve",
]


def main() -> int:
    cfg = load_config()
    raw = resolve(cfg, "raw_pgdata")
    ref = pd.read_csv(resolve(cfg, "reference") / "DMS_substitutions.csv").set_index("DMS_id")
    out_dir = resolve(cfg, "processed")
    split_dir = out_dir / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    qcfg, scfg = cfg["qc"], cfg["splits"]
    assay_ids = [c.assay_id for c in SELECTION]

    frames: list[pd.DataFrame] = []
    qc_rows: list[dict] = []
    manifest_rows: list[dict] = []
    split_rows: list[dict] = []
    up_rows: list[dict] = []
    up_reports: dict[str, dict] = {}

    for aid in assay_ids:
        b = read_bundle(raw / f"{aid}.pgdata")
        meta = ref.loc[aid]
        pf = published_folds(b)
        masks = pf.get(cfg["splits"]["published_scheme_name"].replace("pg_", "")) or pf.get("kfold")

        orient = labels.assert_orientation(b.table)

        tab, rec = qcmod.run_assay_qc(
            aid,
            b.table,
            b.wt_sequence,
            published_masks=masks,
            censor_eps=qcfg["censor_eps"],
            censor_mass_frac=qcfg["censor_mass_frac"],
            dup_disagreement_sigma=qcfg["duplicate_disagreement_sigma"],
        )
        rec.notes.append(f"orientation={orient}")

        # ---- sigma_v then standardization (order matters: sigma is rescaled) ----
        tab, sig_dominant, n_real_sigma = labels.attach_sigma(tab)
        tab = labels.standardize(tab)

        # ---- reservation (a): layer-freeze subset ----
        lf = splitmod.carve_layer_freeze(
            tab, aid, cfg["reservations"]["layer_freeze"]["frac"],
            cfg["reservations"]["layer_freeze"]["seed"],
        )
        tab["is_layer_freeze_reserve"] = lf

        # ---- published folds, with deterministic fallback where unusable ----
        if rec.published_mask_usable:
            pgf = tab["published_fold"].to_numpy(int)
            pg_provenance = "published"
        else:
            pgf = splitmod.kfold_fallback(tab, aid, len(masks), scfg["master_seed"])
            pg_provenance = "re_derived_deterministic"
        for i, f in enumerate(pgf):
            split_rows.append(dict(assay_id=aid, scheme="pg_kfold", variant_row=i, fold=int(f)))

        # ---- unseen_position (stored as a position-fold BITMASK, not a fold id;
        #      the three-way train/test/drop decode lives in splits.py) ----
        upm = splitmod.unseen_position_masks(
            tab, aid, scfg["unseen_position"]["n_folds"], scfg["master_seed"]
        )
        up_reports[aid] = splitmod.unseen_position_report(
            tab,
            upm,
            scfg["unseen_position"]["n_folds"],
            min_test=scfg["unseen_position"].get("min_test_size", 0),
            min_train=scfg["unseen_position"].get("min_train_size", 0),
        )
        for i, m in enumerate(upm):
            up_rows.append(dict(assay_id=aid, variant_row=i, pos_fold_mask=int(m)))

        # ---- mutation_count_extrapolation ----
        mce = splitmod.mutation_count_extrapolation(tab, scfg["mutation_count_extrapolation"]["k"])
        mce_defined = bool((mce >= 0).any())
        if mce_defined:
            for i, f in enumerate(mce):
                split_rows.append(
                    dict(
                        assay_id=aid,
                        scheme="mutation_count_extrapolation",
                        variant_row=i,
                        fold=int(f),
                    )
                )

        # ---- low_n, nested, drawn from pg_kfold fold-0-excluded training pool ----
        train_pool = np.where((pgf != 0) & (~lf))[0]
        low_n_counts = {}
        for s in range(scfg["low_n"]["n_seeds"]):
            subs = splitmod.nested_low_n(train_pool, aid, scfg["low_n"]["N"], s)
            for N, idx in subs.items():
                low_n_counts[N] = low_n_counts.get(N, 0) + 1
                for i in idx:
                    split_rows.append(
                        dict(
                            assay_id=aid,
                            scheme=f"low_n_N{N}_seed{s}",
                            variant_row=int(i),
                            fold=0,
                        )
                    )

        # ---- assemble unified rows ----
        pos = tab["mutant"].map(splitmod.mutated_positions)

        # HARD CHECK: the stored mutation string must regenerate the mutant
        # sequence exactly. This is what licenses not storing the sequence.
        rebuilt = [labels_apply(b.wt_sequence, m) for m in tab["mutant"]]
        n_bad = int(sum(r != s for r, s in zip(rebuilt, tab["sequence"])))
        if n_bad:
            raise AssertionError(
                f"{aid}: {n_bad} variants do not regenerate from wt+mutations; "
                "refusing to drop mutant_sequence from the table"
            )

        f = pd.DataFrame(
            {
                "assay_id": aid,
                "variant_row": np.arange(len(tab)),
                "mutations": tab["mutant"].to_numpy(),
                "n_mutations": tab["n_mutations"].to_numpy(),
                "mutated_positions": pos.map(lambda t: ";".join(map(str, t))).to_numpy(),
                "y_raw": tab["y_raw"].to_numpy(),
                "y": tab["y"].to_numpy(),
                "sigma_v": tab["sigma_v"].to_numpy(),
                "sigma_source": tab["sigma_source"].to_numpy(),
                "n_measurements": tab["n_measurements"].to_numpy(),
                "censored": tab["censored"].to_numpy(),
                "dup_disagrees": tab["dup_disagrees"].to_numpy(),
                "is_layer_freeze_reserve": tab["is_layer_freeze_reserve"].to_numpy(),
            }
        )
        frames.append(f[SCHEMA])

        r = rec.as_row()
        r["orientation"] = orient
        r["sigma_dominant_source"] = sig_dominant
        r["n_variants_with_replicate_sigma"] = n_real_sigma
        r["pg_kfold_provenance"] = pg_provenance
        r["mce_defined"] = mce_defined
        qc_rows.append(r)

        a = b.assay_meta
        manifest_rows.append(
            dict(
                assay_id=aid,
                slot=next(c.slot for c in SELECTION if c.assay_id == aid),
                wt_sequence=b.wt_sequence,
                protein_id=str(meta.get("molecule_name", "")),
                selection_assay=str(meta.get("selection_assay", "")),
                source="ProteinGym2.0",
                hf_repo=cfg.get("sources", {}).get("proteingym2_repo", "ProteinGym/ProteinGym2.0"),
                uniprot_id=str(meta.get("UniProt_ID", "")),
                organism=str(meta.get("source_organism", "")),
                phenotype_class=str(meta.get("coarse_selection_type", "")),
                manifest_phenotype=a.get("target_phenotype"),
                readout=a.get("readout"),
                library=a.get("library_construction_method"),
                growth_coupled=a.get("growth_coupled"),
                transformation=a.get("transformation"),
                n_biological_replicates=a.get("number_of_biological_replicates"),
                manifest_has_uncertainty=a.get("has_uncertainty"),
                per_variant_sigma_present=(sig_dominant != "unavailable"),
                wt_len=len(b.wt_sequence),
                n_variants_raw=rec.n_rows_raw,
                n_variants_curated=rec.n_rows_kept,
                v1_declared_total=int(meta.get("DMS_total_number_mutants", -1)),
                n_single=int((tab["n_mutations"] == 1).sum()),
                n_multi=int((tab["n_mutations"] > 1).sum()),
                max_n_mutations=int(tab["n_mutations"].max()),
                unseen_position_available=up_reports[aid]["b2_available"],
                unseen_position_usable_folds=up_reports[aid]["n_usable_folds"],
                n_mutated_positions=up_reports[aid]["n_positions"],
                position_coverage=round(up_reports[aid]["n_positions"] / len(b.wt_sequence), 4),
                pdb_present=any(n.startswith("structures/") for n in b.member_names),
                msa_neff_category=str(meta.get("MSA_Neff_L_category", "")),
                publication_doi=b.publication.get("doi"),
                censor_floor=rec.censor_floor_flagged,
                censor_ceil=rec.censor_ceil_flagged,
                ties_frac=round(rec.ties_frac, 4),
                pg_kfold_provenance=pg_provenance,
            )
        )

    variants = pd.concat(frames, ignore_index=True)
    variants.to_csv(out_dir / "variants.csv", index=False)
    variants.to_parquet(out_dir / "variants.parquet", index=False)

    pd.DataFrame(qc_rows).to_csv(out_dir / "qc_report.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(out_dir / "assay_manifest.csv", index=False)

    sp = pd.DataFrame(split_rows)
    for scheme, grp in sp.groupby(sp["scheme"].str.replace(r"_N\d+_seed\d+", "", regex=True)):
        grp.to_csv(split_dir / f"{scheme}.csv", index=False)

    # unseen_position is not a fold-id table: one bitmask per variant, decoded
    # per fold into train / test / dropped-straddling.
    pd.DataFrame(up_rows).to_csv(split_dir / "unseen_position.csv", index=False)

    # reservation (b): grouped assay folds for the prediction track
    groups = splitmod.grouped_assay_folds(
        assay_ids, cfg["reservations"]["prediction_folds"]["K"],
        cfg["reservations"]["prediction_folds"]["seed"],
    )
    pd.DataFrame(
        [dict(assay_id=k, pred_group=v) for k, v in sorted(groups.items())]
    ).to_csv(split_dir / "prediction_groups.csv", index=False)

    (out_dir / "unseen_position_report.json").write_text(json.dumps(up_reports, indent=2))

    print(f"variants: {len(variants):,} rows across {variants.assay_id.nunique()} assays")
    print(f"split rows: {len(sp):,}")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
