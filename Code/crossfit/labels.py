"""Label normalization and uncertainty accounting.

ORIENTATION — verified empirically, not assumed
------------------------------------------------
ProteinGym's ``DMS Score`` is ALREADY oriented so that higher = fitter; the
``raw_DMS_directionality`` column in ``DMS_substitutions.csv`` records the sign
that was applied to the *raw* phenotype when ProteinGym built ``DMS Score``.
Applying it a second time is a silent sign flip. Verified against the
``DMS Score Bin`` column: for every assay carrying it, mean ``DMS Score`` is
higher in bin 1 (functional) than bin 0, including ``CCDB_ECOLI_Adkar_2012``
whose directionality is -1. ``assert_orientation`` re-checks this at build time.

UNCERTAINTY — sigma_v
---------------------
ProteinGym 2.0 declares ``has_uncertainty = true`` in every bundle manifest, but
no bundle ships a per-variant uncertainty column, and ``DMS_substitutions.csv``
has no uncertainty field either. The flag describes the underlying experiment,
not the repackaged table. Consequently sigma_v is genuinely unavailable for the
supervised ProteinGym benchmark, which is also why no ProteinGym-based
supervised baseline (ProteinNPT, Kermut, the ProteinGym supervised leaderboard)
consumes a per-variant sigma.

We therefore report sigma_v honestly rather than inventing one:
  ``replicate_sd``  - SD across duplicate measurements of the SAME mutant
                      sequence within the assay. Real, but only available where
                      the source table repeats a sequence.
  ``mv_trend``      - imputed from a within-assay mean-variance trend fitted to
                      the replicate_sd observations. Only fitted when there are
                      at least ``min_groups_for_trend`` such observations.
  ``unavailable``   - no replicate information exists in the distributed data.
                      sigma_v is NaN. Downstream code MUST treat NaN as "no
                      uncertainty information", never as zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_GROUPS_FOR_TREND = 30


def assert_orientation(table: pd.DataFrame, score_col: str = "DMS Score") -> str:
    """Confirm higher score == fitter using the bundle's own binarized column.

    Returns a short verdict string. Raises if the bundle contradicts itself.
    """
    if "DMS Score Bin" not in table.columns:
        return "no_bin_column_orientation_unverified"
    g = table.groupby("DMS Score Bin")[score_col].mean()
    if not {0, 1}.issubset(set(g.index)):
        return "bin_column_not_binary_orientation_unverified"
    if not g.loc[1] > g.loc[0]:
        raise AssertionError(
            "orientation contradiction: mean score in functional bin 1 "
            f"({g.loc[1]:.4g}) is not above bin 0 ({g.loc[0]:.4g})"
        )
    return "verified_higher_is_fitter"


def fit_mv_trend(y: np.ndarray, sd: np.ndarray) -> np.poly1d | None:
    """Fit log(sd) as a linear function of y over observed replicate SDs."""
    ok = np.isfinite(y) & np.isfinite(sd) & (sd > 0)
    if ok.sum() < MIN_GROUPS_FOR_TREND:
        return None
    return np.poly1d(np.polyfit(y[ok], np.log(sd[ok]), 1))


def attach_sigma(df: pd.DataFrame) -> tuple[pd.DataFrame, str, int]:
    """Populate ``sigma_v`` / ``sigma_source``.

    Expects ``y_raw``, ``y_replicate_sd``, ``n_measurements`` from qc.run_assay_qc.
    Returns (df, dominant_source, n_with_real_sigma).
    """
    out = df.copy()
    y = out["y_raw"].to_numpy(float)
    sd = out["y_replicate_sd"].to_numpy(float)

    sigma = np.full(len(out), np.nan)
    source = np.full(len(out), "unavailable", dtype=object)

    have = np.isfinite(sd) & (out["n_measurements"].to_numpy() > 1)
    # standard error of the mean of n replicates
    sigma[have] = sd[have] / np.sqrt(out["n_measurements"].to_numpy(float)[have])
    source[have] = "replicate_sd"
    n_real = int(have.sum())

    trend = fit_mv_trend(y[have], sd[have])
    if trend is not None:
        fill = ~have
        sigma[fill] = np.exp(trend(y[fill]))
        source[fill] = "mv_trend"

    out["sigma_v"] = sigma
    out["sigma_source"] = source
    dominant = pd.Series(source).value_counts().idxmax()
    return out, str(dominant), n_real


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Within-assay z-score. sigma_v is scaled by the same factor so it stays
    on the same scale as y."""
    out = df.copy()
    mu = float(out["y_raw"].mean())
    sd = float(out["y_raw"].std(ddof=0))
    if sd == 0:
        raise ValueError("degenerate label: zero variance")
    out["y"] = (out["y_raw"] - mu) / sd
    out["sigma_v"] = out["sigma_v"] / sd
    out.attrs["y_center"] = mu
    out.attrs["y_scale"] = sd
    return out
