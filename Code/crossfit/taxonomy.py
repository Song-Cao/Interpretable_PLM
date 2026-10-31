"""Null-calibrated crosscoder statistics and provisional feature taxonomy."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import beta as beta_distribution

from .crosscoder import CrossCoder, decoder_delta


@dataclass(frozen=True)
class FeatureStatistics:
    """Per-feature quantities needed for artifact filtering and taxonomy."""

    delta: np.ndarray
    evo_decoder_norm: np.ndarray
    fit_decoder_norm: np.ndarray
    decoder_cosine: np.ndarray
    activation_count: np.ndarray
    activation_frequency: np.ndarray
    self_pair_activation_frequency: np.ndarray
    activation_frequency_change: np.ndarray
    error_ratio: np.ndarray
    reconstruction_ratio: np.ndarray


def benjamini_hochberg(pvalues: np.ndarray, q: float) -> np.ndarray:
    """Return the Benjamini-Hochberg rejection mask for finite p-values."""
    values = np.asarray(pvalues, dtype=float)
    rejected = np.zeros(values.shape, dtype=bool)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return rejected
    order = finite[np.argsort(values[finite], kind="stable")]
    ranked = values[order]
    thresholds = q * np.arange(1, len(order) + 1) / len(order)
    passing = np.flatnonzero(ranked <= thresholds)
    if len(passing):
        rejected[order[: passing[-1] + 1]] = True
    return rejected


def fit_beta_null(values: np.ndarray) -> tuple[float, float]:
    """Fit a Beta distribution on (0,1) with fixed support."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 3:
        raise ValueError("Beta null calibration requires at least three finite features")
    clipped = np.clip(finite, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    shape_a, shape_b, _, _ = beta_distribution.fit(clipped, floc=0.0, fscale=1.0)
    return float(shape_a), float(shape_b)


def beta_pvalues(values: np.ndarray, parameters: tuple[float, float], tail: str) -> np.ndarray:
    """Convert bounded statistics to one-sided p-values under a fitted Beta null."""
    clipped = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    cdf = beta_distribution.cdf(clipped, parameters[0], parameters[1])
    if tail == "lower":
        return cdf
    if tail == "upper":
        return 1.0 - cdf
    raise ValueError(f"unknown tail: {tail}")


def empirical_pvalues(values: np.ndarray, null: np.ndarray, tail: str) -> np.ndarray:
    """Return finite-sample one-sided p-values against one null run."""
    observed = np.asarray(values, dtype=float)
    reference = np.asarray(null, dtype=float)
    reference = reference[np.isfinite(reference)]
    if not len(reference):
        raise ValueError("empirical calibration requires finite null values")
    if tail == "lower":
        counts = np.asarray([(reference <= value).sum() for value in observed])
    elif tail == "upper":
        counts = np.asarray([(reference >= value).sum() for value in observed])
    else:
        raise ValueError(f"unknown tail: {tail}")
    return (counts + 1.0) / (len(reference) + 1.0)


@torch.inference_mode()
def feature_statistics(
    model: CrossCoder,
    evo: np.ndarray | torch.Tensor,
    fit: np.ndarray | torch.Tensor,
    *,
    batch_size: int,
    device: str | torch.device,
) -> FeatureStatistics:
    """Stream exact feature statistics over an aligned activation pool."""
    if evo.shape != fit.shape or evo.ndim != 2:
        raise ValueError("feature statistics require aligned matrices")
    model = model.to(device).eval()
    features = model.features
    counts = torch.zeros(features, dtype=torch.float64, device=device)
    self_counts = torch.zeros(features, dtype=torch.float64, device=device)
    code_squared = torch.zeros(features, dtype=torch.float64, device=device)
    rec_evo_num = torch.zeros(features, dtype=torch.float64, device=device)
    rec_fit_num = torch.zeros(features, dtype=torch.float64, device=device)
    err_evo_num = torch.zeros(features, dtype=torch.float64, device=device)
    err_fit_num = torch.zeros(features, dtype=torch.float64, device=device)
    evo_decoder = model.evo_decoder.weight.detach().float()
    fit_decoder = model.fit_decoder.weight.detach().float()
    decoder_cross = torch.sum(evo_decoder.double() * fit_decoder.double(), dim=0)
    fit_decoder_squared = torch.sum(fit_decoder.double() ** 2, dim=0)
    n = len(evo)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        left = torch.as_tensor(evo[start:stop], dtype=torch.float32, device=device)
        right = torch.as_tensor(fit[start:stop], dtype=torch.float32, device=device)
        code, left_hat, right_hat = model(left, right)
        self_code = model.encode(left, left)
        counts += (code > 0).sum(dim=0).double()
        self_counts += (self_code > 0).sum(dim=0).double()
        code_double = code.double()
        squared = torch.sum(code_double**2, dim=0)
        code_squared += squared
        left_projection = left @ fit_decoder
        right_projection = right @ fit_decoder
        rec_evo_num += torch.sum(code_double * left_projection.double(), dim=0)
        rec_fit_num += torch.sum(code_double * right_projection.double(), dim=0)
        left_residual_projection = (left - left_hat) @ fit_decoder
        right_residual_projection = (right - right_hat) @ fit_decoder
        err_evo_num += torch.sum(code_double * left_residual_projection.double(), dim=0)
        err_evo_num += squared * decoder_cross
        err_fit_num += torch.sum(code_double * right_residual_projection.double(), dim=0)
        err_fit_num += squared * fit_decoder_squared
    denominator = code_squared * fit_decoder_squared
    rec_evo_beta = torch.where(denominator > 0, rec_evo_num / denominator, torch.nan)
    rec_fit_beta = torch.where(denominator > 0, rec_fit_num / denominator, torch.nan)
    err_evo_beta = torch.where(denominator > 0, err_evo_num / denominator, torch.nan)
    err_fit_beta = torch.where(denominator > 0, err_fit_num / denominator, torch.nan)
    reconstruction_ratio = rec_evo_beta / rec_fit_beta
    error_ratio = err_evo_beta / err_fit_beta
    evo_norm = torch.linalg.vector_norm(evo_decoder.double(), dim=0)
    fit_norm = torch.linalg.vector_norm(fit_decoder.double(), dim=0)
    cosine = decoder_cross / (evo_norm * fit_norm)
    frequency = counts / n
    self_frequency = self_counts / n
    return FeatureStatistics(
        delta=decoder_delta(model).detach().cpu().numpy(),
        evo_decoder_norm=evo_norm.cpu().numpy(), fit_decoder_norm=fit_norm.cpu().numpy(),
        decoder_cosine=cosine.cpu().numpy(), activation_count=counts.cpu().numpy(),
        activation_frequency=frequency.cpu().numpy(),
        self_pair_activation_frequency=self_frequency.cpu().numpy(),
        activation_frequency_change=torch.abs(frequency - self_frequency).cpu().numpy(),
        error_ratio=error_ratio.cpu().numpy(),
        reconstruction_ratio=reconstruction_ratio.cpu().numpy(),
    )


def null_eligible(statistics: FeatureStatistics, floor_fraction: float) -> np.ndarray:
    """Select live null features clearing each decoder's own norm floor."""
    evo_floor = floor_fraction * np.median(statistics.evo_decoder_norm)
    fit_floor = floor_fraction * np.median(statistics.fit_decoder_norm)
    return (
        (statistics.activation_count > 0)
        & (statistics.evo_decoder_norm >= evo_floor)
        & (statistics.fit_decoder_norm >= fit_floor)
    )


def calibrate_beta_tail(
    real: np.ndarray,
    null_runs: list[np.ndarray],
    *,
    tail: str,
    q: float,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    """Calibrate a bounded tail per null run and take its conservative cutoff."""
    cutoffs = []
    reports = []
    for index, null in enumerate(null_runs):
        parameters = fit_beta_null(null)
        pvalues = beta_pvalues(real, parameters, tail)
        rejected = benjamini_hochberg(pvalues, q)
        if rejected.any():
            cutoff = float(np.max(real[rejected]) if tail == "lower" else np.min(real[rejected]))
        else:
            cutoff = -np.inf if tail == "lower" else np.inf
        cutoffs.append(cutoff)
        reports.append({
            "null_run": float(index), "beta_a": parameters[0], "beta_b": parameters[1],
            "cutoff": cutoff, "rejections": float(rejected.sum()),
        })
    cutoff = float(np.min(cutoffs) if tail == "lower" else np.max(cutoffs))
    passed = real <= cutoff if tail == "lower" else real >= cutoff
    return passed, cutoff, reports


def calibrate_empirical_tail(
    real: np.ndarray,
    null_runs: list[np.ndarray],
    *,
    tail: str,
    q: float,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    """Calibrate an arbitrary statistic per null run and take a conservative cutoff."""
    cutoffs = []
    reports = []
    for index, null in enumerate(null_runs):
        pvalues = empirical_pvalues(real, null, tail)
        rejected = benjamini_hochberg(pvalues, q)
        if rejected.any():
            cutoff = float(np.max(real[rejected]) if tail == "lower" else np.min(real[rejected]))
        else:
            cutoff = -np.inf if tail == "lower" else np.inf
        cutoffs.append(cutoff)
        reports.append({
            "null_run": float(index), "cutoff": cutoff,
            "rejections": float(rejected.sum()), "null_features": float(np.isfinite(null).sum()),
        })
    cutoff = float(np.min(cutoffs) if tail == "lower" else np.max(cutoffs))
    passed = real <= cutoff if tail == "lower" else real >= cutoff
    return passed, cutoff, reports


def provisional_classes(
    statistics: FeatureStatistics,
    *,
    lower_delta: np.ndarray,
    upper_delta: np.ndarray,
    latent_error_pass: np.ndarray,
    latent_reconstruction_pass: np.ndarray,
    rotated: np.ndarray,
    frequency_shifted: np.ndarray,
    floor_fraction: float,
    activation_floor_count: int,
) -> np.ndarray:
    """Apply the registered precedence order to provisional Part B classes."""
    size = len(statistics.delta)
    labels = np.full(size, "unclassified", dtype=object)
    evo_floor = floor_fraction * np.median(statistics.evo_decoder_norm)
    fit_floor = floor_fraction * np.median(statistics.fit_decoder_norm)
    live = statistics.activation_count >= activation_floor_count
    evo_clear = statistics.evo_decoder_norm >= evo_floor
    fit_clear = statistics.fit_decoder_norm >= fit_floor
    labels[~live | (~evo_clear & ~fit_clear)] = "dead"
    available = labels == "unclassified"
    emergent = (
        available & fit_clear & upper_delta & latent_error_pass & latent_reconstruction_pass
    )
    labels[emergent] = "emergent_candidate"
    available = labels == "unclassified"
    labels[available & evo_clear & lower_delta] = "evolution_only"
    available = labels == "unclassified"
    labels[available & ~(evo_clear & fit_clear)] = "dead"
    available = labels == "unclassified"
    labels[available & rotated] = "rotated"
    available = labels == "unclassified"
    labels[available & frequency_shifted] = "re_weighted"
    labels[labels == "unclassified"] = "conserved"
    return labels.astype(str)
