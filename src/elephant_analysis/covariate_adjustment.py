"""Historical covariate-adjusted contrasts using frozen out-of-fold inputs.

This module contains no file access and performs no model fitting or bootstrap
resampling. Its four estimators preserve the inspected notebook algorithms.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import pandas as pd


SOURCE_METHOD_LABELS = ("ATT-Match", "ATE-IPW", "ATE-Gcomp", "ATE-DR")
METHOD_LABELS = ("Matching", "IPW", "Standardization", "DR")
OUTCOME_LABELS = ("th", "tc")
OUTCOME_COLUMNS = {"tc": "tc_africa", "th": "th_africa"}
SAMPLE_KEY_COLUMNS = ("block_id", "row", "col")
PREDICTION_KEYS = tuple(
    f"{name}_{outcome}"
    for outcome in OUTCOME_LABELS
    for name in ("e_hat", "m1", "m0", "yhat_oof")
)

DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    "matching": {
        "min_per_group": 5,
        "caliper_mult": 0.1,
        "ratio_max": 5,
        "random_state": 42,
        "pair_weight": "design",
        "with_replacement": True,
        "winsorize_diff": None,
    },
    "ipw": {
        "use_design_weight": True,
        "clip": [0.1, 0.9],
        "trim": [0.1, 0.9],
        "min_per_group": 5,
        "block_weight": "design",
    },
    "gcomp": {"use_design_weight": True},
    "dr": {
        "use_design_weight": True,
        "clip": [0.1, 0.9],
        "trim": [0.1, 0.9],
        "trim_on": "aug",
        "weighted_trim_by_outcome": {"tc": False, "th": True},
        "cluster_se": False,
    },
}


def matching_att_within_blocks_adaptive(
    sampled_df: pd.DataFrame,
    e_hat: np.ndarray,
    Y_col: str,
    A_col: str = "A",
    block_col: str = "block_id",
    min_per_group: int = 5,
    caliper_mult: float = 0.2,
    ratio_max: int = 5,
    random_state: int = 42,
    pair_weight: str = "design",
    with_replacement: bool = False,
    winsorize_diff: tuple[float, float] | None = None,
) -> tuple[float, int]:
    """Run the historical adaptive within-block logit-propensity match.

    The notebook labels this contrast ATT. Control reuse is allowed when
    requested, and each matched event uses the treated design weight divided
    by the running reuse count for its selected control.
    """
    rng = np.random.default_rng(random_state)
    df = sampled_df[[block_col, A_col, Y_col]].copy().reset_index(drop=True)
    e_hat = np.asarray(e_hat).reshape(-1)
    if len(e_hat) != len(df):
        raise ValueError("e_hat length must match sampled_df rows")
    df["e_hat"] = np.clip(e_hat, 0.1, 0.9)
    df["z"] = np.log(df["e_hat"] / (1 - df["e_hat"]))
    has_w = "w_design" in sampled_df.columns
    if has_w:
        df["w_design"] = sampled_df["w_design"].values.astype(float)

    atts, W_blocks = [], []
    total_pairs = 0

    for _, sub in df.groupby(block_col, sort=False):
        sub = sub.reset_index()
        A = sub[A_col].values.astype(int)
        Y = sub[Y_col].values.astype(float)
        z = sub["z"].values
        t_idx = np.where(A == 1)[0]
        c_idx = np.where(A == 0)[0]
        n_t, n_c = len(t_idx), len(c_idx)
        if n_t == 0 or n_c == 0:
            continue

        sd_z = float(np.std(z))
        if (not np.isfinite(sd_z)) or sd_z == 0.0 or min(n_t, n_c) < 3:
            continue

        caliper = caliper_mult * sd_z
        z_t = z[t_idx][:, None]
        z_c = z[c_idx][None, :]
        D = np.abs(z_t - z_c) + rng.normal(0.0, 1e-12, size=(n_t, n_c))
        within = (D <= caliper).sum(axis=1)
        need = np.minimum(ratio_max, np.maximum(1, within)).astype(int)
        order_t = t_idx.copy()
        rng.shuffle(order_t)

        match_counts = np.zeros(n_c, dtype=float) if with_replacement else None
        used_c = np.zeros(n_c, dtype=bool) if not with_replacement else None
        have = np.zeros(n_t, dtype=int)
        diffs, w_pairs = [], []

        for _ in range(int(np.max(need))):
            if not with_replacement and used_c.all():
                break
            for t in order_t:
                it = int(np.where(t_idx == t)[0][0])
                if have[it] >= need[it]:
                    continue
                drow = D[it, :].copy()
                if not with_replacement:
                    drow[used_c] = np.inf
                j = int(np.argmin(drow))
                dmin = float(drow[j])
                if not np.isfinite(dmin) or dmin > caliper:
                    continue
                diffs.append(float(Y[t] - Y[c_idx[j]]))
                if with_replacement:
                    match_counts[j] += 1
                    w_control = 1.0 / max(1.0, match_counts[j])
                    if pair_weight == "design" and has_w:
                        w_pairs.append(float(sub.loc[t, "w_design"]) * w_control)
                    else:
                        w_pairs.append(w_control)
                else:
                    if pair_weight == "design" and has_w:
                        w_pairs.append(float(sub.loc[t, "w_design"]))
                    else:
                        w_pairs.append(1.0)
                if not with_replacement:
                    used_c[j] = True
                have[it] += 1

        if len(diffs) == 0 or (np.sum(np.isfinite(diffs)) < min_per_group):
            continue

        diffs = np.asarray(diffs, dtype=float)
        w_pairs = np.asarray(w_pairs, dtype=float)
        if winsorize_diff is not None and len(diffs) > 3:
            ql, qh = winsorize_diff
            if 0 <= ql < qh <= 1:
                lo = np.quantile(diffs, ql)
                hi = np.quantile(diffs, qh)
                diffs = np.clip(diffs, lo, hi)

        att_b = float(np.average(diffs, weights=w_pairs))
        atts.append(att_b)
        W_blocks.append(float(len(diffs)) if pair_weight == "count" else float(np.sum(w_pairs)))
        total_pairs += int(len(diffs))

    if len(atts) == 0:
        return np.nan, 0

    att = float(np.average(atts, weights=W_blocks))
    return att, total_pairs


def ipw_ate_blocked(
    sampled_df: pd.DataFrame,
    e_hat: np.ndarray,
    Y_col: str,
    A_col: str = "A",
    block_col: str = "block_id",
    use_design_weight: bool = True,
    clip: tuple[float, float] = (0.02, 0.98),
    trim: tuple[float, float] | None = None,
    min_per_group: int = 5,
    block_weight: str = "design",
) -> float:
    """Run the historical block-stabilized and winsorized IPW contrast."""
    df = sampled_df.copy().reset_index(drop=True)
    n = len(df)
    e = np.asarray(e_hat, float).reshape(-1)
    if e.shape[0] != n:
        raise ValueError("e_hat length must match sampled_df rows")
    e = np.clip(e, clip[0], clip[1])
    A = df[A_col].values.astype(int)
    Y = df[Y_col].values.astype(float)
    w_design = (
        df["w_design"].values.astype(float)
        if (use_design_weight and "w_design" in df.columns)
        else np.ones(n, float)
    )
    ate_blocks, W_blocks = [], []
    for _, sub in df.assign(e_hat=e, w_design=w_design).groupby(block_col, sort=False):
        A_b = sub[A_col].values.astype(int)
        Y_b = sub[Y_col].values.astype(float)
        e_b = sub["e_hat"].values.astype(float)
        wS_b = sub["w_design"].values.astype(float)

        if (np.sum(A_b == 1) < min_per_group) or (np.sum(A_b == 0) < min_per_group):
            continue
        sw = np.sum(wS_b)
        if sw <= 0:
            continue
        p_b = float(np.sum(wS_b * A_b) / sw)
        w_causal = np.where(A_b == 1, p_b / e_b, (1.0 - p_b) / (1.0 - e_b))
        if trim is not None:
            ql, qh = np.quantile(w_causal, [trim[0], trim[1]])
            w_causal = np.clip(w_causal, ql, qh)
        w_final = wS_b * w_causal if use_design_weight else w_causal
        wf1 = np.sum(w_final[A_b == 1])
        wf0 = np.sum(w_final[A_b == 0])
        if wf1 <= 0.0 or wf0 <= 0.0:
            continue
        mu1 = float(np.sum(w_final[A_b == 1] * Y_b[A_b == 1]) / wf1)
        mu0 = float(np.sum(w_final[A_b == 0] * Y_b[A_b == 0]) / wf0)
        ate_blocks.append(mu1 - mu0)
        W_blocks.append(float(np.sum(wS_b)) if block_weight == "design" else 1.0)
    if len(ate_blocks) == 0:
        return np.nan
    return float(np.average(ate_blocks, weights=np.asarray(W_blocks, float)))


def gcomp_ate(
    sampled_df: pd.DataFrame,
    m1: np.ndarray,
    m0: np.ndarray,
    use_design_weight: bool = True,
) -> float:
    """Average the frozen potential-outcome prediction difference."""
    n = len(sampled_df)
    mu1 = np.asarray(m1, float).reshape(-1)
    mu0 = np.asarray(m0, float).reshape(-1)
    if (mu1.shape[0] != n) or (mu0.shape[0] != n):
        raise ValueError("m1/m0 lengths must match sampled_df rows")

    if use_design_weight and ("w_design" in sampled_df.columns):
        wS = sampled_df["w_design"].values.astype(float)
    else:
        wS = np.ones(n, float)

    diff = mu1 - mu0
    den = float(np.sum(wS))
    if den <= 0.0:
        return np.nan
    return float(np.sum(wS * diff) / den)


def dr_ate(
    sampled_df: pd.DataFrame,
    e_hat: np.ndarray,
    m1: np.ndarray,
    m0: np.ndarray,
    Y_col: str,
    A_col: str = "A",
    use_design_weight: bool = True,
    block_col: str = "block_id",
    clip: tuple[float, float] = (0.02, 0.98),
    trim: tuple[float, float] | None = None,
    trim_on: str = "aug",
    weighted_trim: bool = False,
    cluster_se: bool = True,
) -> tuple[float, float] | float:
    """Run the historical winsorized augmented inverse-probability estimate."""
    df = sampled_df.reset_index(drop=True)
    n = len(df)

    A = df[A_col].values.astype(int)
    Y = df[Y_col].values.astype(float)
    e = np.asarray(e_hat, float).reshape(-1)
    mu1 = np.asarray(m1, float).reshape(-1)
    mu0 = np.asarray(m0, float).reshape(-1)
    if (e.shape[0] != n) or (mu1.shape[0] != n) or (mu0.shape[0] != n):
        raise ValueError("e_hat/m1/m0 lengths must match sampled_df rows")

    e = np.clip(e, clip[0], clip[1])
    if use_design_weight and ("w_design" in df.columns):
        wS = df["w_design"].values.astype(float)
    else:
        wS = np.ones(n, float)

    def winsorize(x: np.ndarray, q=(0.01, 0.99), w=None) -> np.ndarray:
        if q is None:
            return x
        ql, qh = float(q[0]), float(q[1])
        x = np.asarray(x, float)
        if w is None:
            lo, hi = np.quantile(x, [ql, qh])
        else:
            order = np.argsort(x)
            xs = x[order]
            ws = np.asarray(w, float)[order]
            cs = np.cumsum(ws) / np.sum(ws)
            lo = np.interp(ql, cs, xs)
            hi = np.interp(qh, cs, xs)
        return np.clip(x, lo, hi)

    aug = A * (Y - mu1) / e - (1 - A) * (Y - mu0) / (1 - e)
    if trim is not None and trim_on.lower() == "aug":
        aug = winsorize(aug, q=trim, w=(wS if weighted_trim else None))
        psi = (mu1 - mu0) + aug
    else:
        psi = (mu1 - mu0) + aug
        if trim is not None and trim_on.lower() == "psi":
            psi = winsorize(psi, q=trim, w=(wS if weighted_trim else None))

    num = np.sum(wS * psi)
    den = np.sum(wS)
    if den <= 0.0:
        return (np.nan, np.nan) if cluster_se else np.nan
    ate = float(num / den)

    if not cluster_se:
        var = np.sum((wS**2) * (psi - ate) ** 2) / (den**2)
        se = float(np.sqrt(max(var, 0.0)))
        return ate, se

    blocks = df[block_col].values
    score = wS * (psi - ate)
    uniq = np.unique(blocks)
    sgsq = 0.0
    for b in uniq:
        sgsq += np.sum(score[blocks == b]) ** 2
    G = len(uniq)
    factor = G / (G - 1) if G > 1 else 1.0
    var_cluster = factor * sgsq / (den**2)
    se = float(np.sqrt(max(var_cluster, 0.0)))
    return ate, se


def _merged_config(config: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Merge documented method-level overrides and reject unknown options."""
    merged = deepcopy(DEFAULT_CONFIG)
    if config is None:
        return merged
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping of estimator options")
    for method, overrides in config.items():
        if method not in merged:
            raise ValueError(f"unknown estimator configuration: {method}")
        if not isinstance(overrides, Mapping):
            raise TypeError(f"config[{method!r}] must be a mapping")
        unknown = set(overrides) - set(merged[method])
        if unknown:
            raise ValueError(f"unknown {method} options: {sorted(unknown)}")
        merged[method].update(overrides)
    if not isinstance(merged["dr"]["weighted_trim_by_outcome"], Mapping):
        raise TypeError("dr.weighted_trim_by_outcome must map tc/th to booleans")
    for outcome in OUTCOME_LABELS:
        if outcome not in merged["dr"]["weighted_trim_by_outcome"]:
            raise ValueError(f"missing DR weighted-trim setting for {outcome}")
    return merged


def _as_finite_vector(inputs: Mapping[str, Any], key: str, n: int) -> np.ndarray:
    if key not in inputs:
        raise KeyError(f"missing required input array: {key}")
    array = np.asarray(inputs[key])
    if array.dtype.kind not in "biuf":
        raise TypeError(f"{key} must be a numeric array")
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    if array.shape != (n,):
        raise ValueError(f"{key} must have shape ({n},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{key} contains non-finite values")
    return array


def _reference_arrays(inputs: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    keys = (
        "reference_method_labels",
        "reference_outcome_labels",
        "reference_point_estimates",
        "reference_ci_lower",
        "reference_ci_upper",
    )
    missing = [key for key in keys if key not in inputs]
    if missing:
        raise KeyError(f"missing archived-reference arrays: {missing}")
    method_labels = np.asarray(inputs["reference_method_labels"]).astype(str)
    outcome_labels = np.asarray(inputs["reference_outcome_labels"]).astype(str)
    expected_methods = np.asarray(SOURCE_METHOD_LABELS)
    expected_outcomes = np.asarray(OUTCOME_LABELS)
    if not np.array_equal(method_labels, expected_methods):
        raise ValueError(f"reference methods must be ordered as {SOURCE_METHOD_LABELS}")
    if not np.array_equal(outcome_labels, expected_outcomes):
        raise ValueError(f"reference outcomes must be ordered as {OUTCOME_LABELS}")
    shape = (len(OUTCOME_LABELS), len(METHOD_LABELS))
    arrays = []
    for key in ("reference_point_estimates", "reference_ci_lower", "reference_ci_upper"):
        array = np.asarray(inputs[key])
        if array.dtype.kind not in "biuf":
            raise TypeError(f"{key} must be numeric")
        array = np.asarray(array, dtype=np.float64)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"{key} must be finite with shape {shape}")
        arrays.append(array)
    if np.any(arrays[1] > arrays[2]):
        raise ValueError("archived confidence interval lower bound exceeds upper bound")
    return method_labels, outcome_labels, *arrays


def compute_adjusted_contrasts(
    inputs: Mapping[str, Any], config: Mapping[str, Any] | None = None
) -> dict[str, np.ndarray]:
    """Compute four historical contrasts from aligned frozen OOF arrays.

    Required input keys are ``sample_values``, ``sample_columns``, aligned
    ``e_hat/m1/m0/yhat_oof`` arrays for TC and TH, and the five
    ``reference_*`` arrays returned by the archived reference NPZ. The
    archived confidence intervals are copied into the result and tagged;
    this function never recomputes them.
    """
    if not isinstance(inputs, Mapping):
        raise TypeError("inputs must be a mapping of arrays")
    if "sample_values" not in inputs or "sample_columns" not in inputs:
        raise KeyError("inputs require sample_values and sample_columns")
    values = np.asarray(inputs["sample_values"])
    if values.dtype.kind not in "biuf":
        raise TypeError("sample_values must be a numeric array")
    values = np.asarray(values, dtype=np.float64)
    columns = np.asarray(inputs["sample_columns"]).astype(str).tolist()
    if values.ndim != 2 or values.shape[1] != len(columns):
        raise ValueError("sample_values must be a 2D matrix aligned to sample_columns")
    if len(set(columns)) != len(columns):
        raise ValueError("sample_columns must be unique")
    required_columns = set(SAMPLE_KEY_COLUMNS) | {"A", "w_design", *OUTCOME_COLUMNS.values()}
    missing_columns = sorted(required_columns - set(columns))
    if missing_columns:
        raise ValueError(f"sample is missing required columns: {missing_columns}")
    if not np.isfinite(values).all():
        raise ValueError("sample_values contains non-finite values")
    n = values.shape[0]
    if n == 0:
        raise ValueError("sample_values must not be empty")

    data = pd.DataFrame(values, columns=columns)
    if not np.isin(data["A"].to_numpy(), (0.0, 1.0)).all():
        raise ValueError("A must contain only 0 and 1")
    if (data["w_design"].to_numpy() <= 0).any():
        raise ValueError("w_design must be positive")
    predictions: dict[str, np.ndarray] = {}
    for key in PREDICTION_KEYS:
        predictions[key] = _as_finite_vector(inputs, key, n)

    _, source_outcome_labels, reference_points, ci_lower, ci_upper = _reference_arrays(inputs)
    method_labels = np.asarray(METHOD_LABELS)
    outcome_labels = np.asarray([label.upper() for label in source_outcome_labels])
    options = _merged_config(config)

    point_estimates = np.empty((len(source_outcome_labels), len(method_labels)), dtype=np.float64)
    standard_errors = np.full_like(point_estimates, np.nan)
    matched_pair_events = np.zeros(len(source_outcome_labels), dtype=np.int64)

    for outcome_index, outcome in enumerate(source_outcome_labels.tolist()):
        y_col = OUTCOME_COLUMNS[outcome]
        e = predictions[f"e_hat_{outcome}"]
        m1 = predictions[f"m1_{outcome}"]
        m0 = predictions[f"m0_{outcome}"]

        match_config = options["matching"]
        estimate_match, pair_count = matching_att_within_blocks_adaptive(
            data,
            e,
            y_col,
            A_col="A",
            block_col="block_id",
            min_per_group=match_config["min_per_group"],
            caliper_mult=match_config["caliper_mult"],
            ratio_max=match_config["ratio_max"],
            random_state=match_config["random_state"],
            pair_weight=match_config["pair_weight"],
            with_replacement=match_config["with_replacement"],
            winsorize_diff=match_config["winsorize_diff"],
        )
        ipw_config = options["ipw"]
        estimate_ipw = ipw_ate_blocked(
            data,
            e,
            y_col,
            A_col="A",
            block_col="block_id",
            use_design_weight=ipw_config["use_design_weight"],
            clip=tuple(ipw_config["clip"]),
            trim=None if ipw_config["trim"] is None else tuple(ipw_config["trim"]),
            min_per_group=ipw_config["min_per_group"],
            block_weight=ipw_config["block_weight"],
        )
        gcomp_config = options["gcomp"]
        estimate_gcomp = gcomp_ate(data, m1, m0, use_design_weight=gcomp_config["use_design_weight"])
        dr_config = options["dr"]
        estimate_dr, se_dr = dr_ate(
            data,
            e,
            m1,
            m0,
            y_col,
            A_col="A",
            block_col="block_id",
            use_design_weight=dr_config["use_design_weight"],
            clip=tuple(dr_config["clip"]),
            trim=None if dr_config["trim"] is None else tuple(dr_config["trim"]),
            trim_on=dr_config["trim_on"],
            weighted_trim=dr_config["weighted_trim_by_outcome"][outcome],
            cluster_se=dr_config["cluster_se"],
        )
        if isinstance(estimate_dr, tuple):
            estimate_dr, se_dr = estimate_dr

        point_estimates[outcome_index] = (estimate_match, estimate_ipw, estimate_gcomp, estimate_dr)
        standard_errors[outcome_index, 3] = se_dr
        matched_pair_events[outcome_index] = pair_count

    return {
        "outcome_labels": outcome_labels,
        "method_labels": method_labels,
        "estimand_labels": np.asarray(
            [
                "ATT label; custom matched contrast with replacement-use weights",
                "ATE; block-stabilized, winsorized IPW with design weights",
                "ATE; design-weighted mean of frozen m1-m0 predictions",
                "ATE; design-weighted augmented inverse-probability contrast",
            ]
        ),
        "point_estimates": point_estimates,
        "reference_point_estimates": reference_points,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_status": np.full(
            ci_lower.shape, "archived_reference_not_recomputed", dtype="<U40"
        ),
        "ci_source": np.full(
            ci_lower.shape,
            "elephant_effect_results.npz; percentile bootstrap outputs, B=100",
            dtype="<U80",
        ),
        "ci_note": np.full(
            ci_lower.shape,
            "Bootstrap not rerun; repeated block IDs were not relabeled; TH DR weighted_trim differed between point and bootstrap calls.",
            dtype="<U140",
        ),
        "bootstrap_recomputed": np.asarray(False),
        "standard_errors": standard_errors,
        "matched_pair_events": matched_pair_events,
    }
