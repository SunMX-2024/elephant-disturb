"""Historical block bootstrap for frozen African elephant OOF predictions.

The original implementation is retained, including repeated source block IDs
being grouped together after resampling and the unweighted DR trim in both
outcomes. The wrapper returns only numeric and fixed-width Unicode arrays.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .covariate_adjustment import (
    METHOD_LABELS,
    OUTCOME_COLUMNS,
    OUTCOME_LABELS,
    dr_ate,
    gcomp_ate,
    ipw_ate_blocked,
    matching_att_within_blocks_adaptive,
)


_BOOT_METHOD_KEYS = ("match", "ipw", "g", "dr")
_DEFAULT_CONFIG: dict[str, Any] = {
    "caliper_mult": 0.1,
    "ratio_max": 5,
    "pair_weight": "design",
    "winsorize_diff": None,
    "clip": [0.1, 0.9],
    "trim": [0.1, 0.9],
    "trim_on": "aug",
    "weighted_trim": False,
    "use_design_weight_ipw": True,
    "use_design_weight_g": True,
    "use_design_weight_dr": True,
    "min_per_group_ipw": 5,
    "block_weight_ipw": "design",
    "max_tries_factor": 5,
}


def block_bootstrap_causal(
    sampled_df: pd.DataFrame,
    e_hat: np.ndarray,
    m1: np.ndarray,
    m0: np.ndarray,
    Y_col: str,
    B: int = 500,
    seed: int = 2025,
    caliper_mult: float = 0.2,
    ratio_max: int = 5,
    pair_weight: str = "design",
    winsorize_diff: tuple[float, float] | None = None,
    clip: tuple[float, float] = (0.02, 0.98),
    trim: tuple[float, float] | None = None,
    trim_on: str = "aug",
    weighted_trim: bool = False,
    use_design_weight_ipw: bool = True,
    use_design_weight_g: bool = True,
    use_design_weight_dr: bool = True,
    min_per_group_ipw: int = 5,
    block_weight_ipw: str = "design",
    max_tries_factor: int = 5,
) -> dict[str, Any]:
    """Reproduce the recovered notebook's block bootstrap algorithm.

    Repeated sampled blocks deliberately keep their original ``block_id``.
    Consequently, downstream grouping combines repeated copies with the same
    ID, matching the original code's behavior.
    """
    rng = np.random.default_rng(seed)
    blocks = sampled_df["block_id"].unique()
    idx_by_block = {b: np.where(sampled_df["block_id"].values == b)[0] for b in blocks}
    boots_match, boots_ipw, boots_g, boots_dr = [], [], [], []
    tries = 0
    max_tries = int(max_tries_factor * B)
    e_hat = np.asarray(e_hat)
    m1 = np.asarray(m1)
    m0 = np.asarray(m0)
    while (
        len(boots_match) < B
        or len(boots_ipw) < B
        or len(boots_g) < B
        or len(boots_dr) < B
    ) and (tries < max_tries):
        tries += 1
        rs = int(seed + tries)
        sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        take_idx = np.concatenate([idx_by_block[bb] for bb in sampled_blocks])
        df_b = sampled_df.iloc[take_idx].copy().reset_index(drop=True)
        e_b = e_hat[take_idx]
        m1_b = m1[take_idx]
        m0_b = m0[take_idx]

        if len(boots_match) < B:
            try:
                att_b, pairs_b = matching_att_within_blocks_adaptive(
                    df_b,
                    e_b,
                    Y_col,
                    A_col="A",
                    block_col="block_id",
                    min_per_group=5,
                    caliper_mult=caliper_mult,
                    ratio_max=ratio_max,
                    random_state=rs,
                    pair_weight=pair_weight,
                    with_replacement=True,
                    winsorize_diff=winsorize_diff,
                )
                if np.isfinite(att_b):
                    boots_match.append(att_b)
            except Exception:
                pass
        if len(boots_ipw) < B:
            try:
                ipw_b = ipw_ate_blocked(
                    df_b,
                    e_b,
                    Y_col,
                    A_col="A",
                    block_col="block_id",
                    use_design_weight=use_design_weight_ipw,
                    clip=clip,
                    trim=trim,
                    min_per_group=min_per_group_ipw,
                    block_weight=block_weight_ipw,
                )
                if np.isfinite(ipw_b):
                    boots_ipw.append(ipw_b)
            except Exception:
                pass
        if len(boots_g) < B:
            try:
                g_b = gcomp_ate(df_b, m1_b, m0_b, use_design_weight=use_design_weight_g)
                if np.isfinite(g_b):
                    boots_g.append(g_b)
            except Exception:
                pass
        if len(boots_dr) < B:
            try:
                dr_b, _ = dr_ate(
                    df_b,
                    e_b,
                    m1_b,
                    m0_b,
                    Y_col,
                    A_col="A",
                    use_design_weight=use_design_weight_dr,
                    block_col="block_id",
                    clip=clip,
                    trim=trim,
                    trim_on=trim_on,
                    weighted_trim=weighted_trim,
                    cluster_se=False,
                )
                if np.isfinite(dr_b):
                    boots_dr.append(dr_b)
            except Exception:
                pass

    return {
        "match": np.array(boots_match, dtype=float),
        "ipw": np.array(boots_ipw, dtype=float),
        "g": np.array(boots_g, dtype=float),
        "dr": np.array(boots_dr, dtype=float),
        "tries": tries,
    }


def ci_from_boot(
    arr: np.ndarray, alpha: float = 0.05, method: str = "percentile"
) -> tuple[float, float]:
    """Return the historical finite-value percentile interval."""
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    if method == "percentile":
        lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(lo), float(hi)
    raise ValueError(f"unknown CI method: {method}")


def _merge_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(_DEFAULT_CONFIG)
    if config is None:
        return merged
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping of historical bootstrap options")
    unknown = set(config) - set(merged)
    if unknown:
        raise ValueError(f"unknown bootstrap options: {sorted(unknown)}")
    merged.update(config)
    return merged


def _vector(inputs: Mapping[str, Any], key: str, n: int) -> np.ndarray:
    if key not in inputs:
        raise KeyError(f"missing required input array: {key}")
    value = np.asarray(inputs[key])
    if value.dtype.kind not in "biuf":
        raise TypeError(f"{key} must be a numeric array")
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if value.shape != (n,):
        raise ValueError(f"{key} must have shape ({n},), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{key} contains non-finite values")
    return value


def bootstrap_adjusted_contrasts(
    inputs: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
    n_boot: int = 100,
    seed: int = 2025,
) -> dict[str, np.ndarray]:
    """Bootstrap four historical contrasts from frozen, aligned predictions.

    This wrapper does not refit propensity or outcome models. It returns
    ``draws`` with shape ``(2, 4, n_boot)``; unavailable draws are
    padded with NaN and their counts are reported separately. The TH DR
    bootstrap deliberately keeps the notebook's default ``weighted_trim=False``
    although its point-estimate call used weighted trimming.
    """
    if not isinstance(inputs, Mapping):
        raise TypeError("inputs must be a mapping of arrays")
    if isinstance(n_boot, (bool, np.bool_)) or not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if "sample_values" not in inputs or "sample_columns" not in inputs:
        raise KeyError("inputs require sample_values and sample_columns")
    sample_values = np.asarray(inputs["sample_values"])
    if sample_values.dtype.kind not in "biuf":
        raise TypeError("sample_values must be a numeric array")
    sample_values = np.asarray(sample_values, dtype=np.float64)
    columns = np.asarray(inputs["sample_columns"]).astype(str).tolist()
    if sample_values.ndim != 2 or sample_values.shape[1] != len(columns):
        raise ValueError("sample_values must be a 2D matrix aligned to sample_columns")
    if len(set(columns)) != len(columns):
        raise ValueError("sample_columns must be unique")
    required = {"block_id", "A", "w_design", *OUTCOME_COLUMNS.values()}
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"sample is missing required columns: {missing}")
    if sample_values.shape[0] == 0 or not np.isfinite(sample_values).all():
        raise ValueError("sample_values must be nonempty and finite")
    n = sample_values.shape[0]
    data = pd.DataFrame(sample_values, columns=columns)
    if not np.isin(data["A"].to_numpy(), (0.0, 1.0)).all():
        raise ValueError("A must contain only 0 and 1")
    if (data["w_design"].to_numpy() <= 0).any():
        raise ValueError("w_design must be positive")

    predictions = {}
    for outcome in OUTCOME_LABELS:
        for prediction in ("e_hat", "m1", "m0"):
            key = f"{prediction}_{outcome}"
            predictions[key] = _vector(inputs, key, n)

    options = _merge_config(config)
    if options["max_tries_factor"] < 1:
        raise ValueError("max_tries_factor must be at least 1")
    clip = tuple(options["clip"])
    trim = None if options["trim"] is None else tuple(options["trim"])
    draws = np.full((len(OUTCOME_LABELS), len(METHOD_LABELS), int(n_boot)), np.nan)
    counts = np.zeros((len(OUTCOME_LABELS), len(METHOD_LABELS)), dtype=np.int64)
    tries = np.zeros(len(OUTCOME_LABELS), dtype=np.int64)

    for outcome_index, outcome in enumerate(OUTCOME_LABELS):
        bootstrap_values = block_bootstrap_causal(
            data,
            predictions[f"e_hat_{outcome}"],
            predictions[f"m1_{outcome}"],
            predictions[f"m0_{outcome}"],
            OUTCOME_COLUMNS[outcome],
            B=int(n_boot),
            seed=int(seed),
            caliper_mult=options["caliper_mult"],
            ratio_max=options["ratio_max"],
            pair_weight=options["pair_weight"],
            winsorize_diff=options["winsorize_diff"],
            clip=clip,
            trim=trim,
            trim_on=options["trim_on"],
            weighted_trim=options["weighted_trim"],
            use_design_weight_ipw=options["use_design_weight_ipw"],
            use_design_weight_g=options["use_design_weight_g"],
            use_design_weight_dr=options["use_design_weight_dr"],
            min_per_group_ipw=options["min_per_group_ipw"],
            block_weight_ipw=options["block_weight_ipw"],
            max_tries_factor=options["max_tries_factor"],
        )
        tries[outcome_index] = bootstrap_values["tries"]
        for method_index, key in enumerate(_BOOT_METHOD_KEYS):
            values = np.asarray(bootstrap_values[key], dtype=np.float64)
            count = min(values.size, int(n_boot))
            draws[outcome_index, method_index, :count] = values[:count]
            counts[outcome_index, method_index] = count

    ci_lower = np.full((len(OUTCOME_LABELS), len(METHOD_LABELS)), np.nan)
    ci_upper = np.full_like(ci_lower, np.nan)
    for i in range(len(OUTCOME_LABELS)):
        for j in range(len(METHOD_LABELS)):
            lo, hi = ci_from_boot(draws[i, j, : counts[i, j]])
            ci_lower[i, j] = lo
            ci_upper[i, j] = hi

    ci_status = np.full(
        ci_lower.shape,
        "legacy_algorithm_interval_not_independently_validated",
        dtype="<U64",
    )
    ci_note = np.full(
        ci_lower.shape,
        "Legacy block bootstrap retains repeated source block IDs; duplicate draws are grouped together.",
        dtype="<U180",
    )
    ci_note[0, 3] = (
        "Legacy block bootstrap retains repeated source block IDs; TH DR uses weighted_trim=False, "
        "unlike its weighted_trim=True point estimate."
    )

    return {
        "outcome_labels": np.asarray([label.upper() for label in OUTCOME_LABELS]),
        "method_labels": np.asarray(METHOD_LABELS),
        "draws": draws,
        "draw_count": counts,
        "bootstrap_tries": tries,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_status": ci_status,
        "ci_note": ci_note,
        "bootstrap_recomputed": np.asarray(True),
        "algorithm_status": np.asarray("legacy_block_bootstrap_preserved_not_independently_validated"),
        "n_boot_requested": np.asarray(n_boot, dtype=np.int64),
        "seed": np.asarray(seed, dtype=np.int64),
    }
