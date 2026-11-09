"""Original Pingouin partial-Spearman calculations and Fig. 5a-d vectors."""

from __future__ import annotations

from typing import Any

import numpy as np


def residualize_without_intercept(y: Any, covariate: Any) -> np.ndarray:
    """Match the plotted residual vectors in the legacy Fig. 5a-d code.

    The displayed partial correlations are calculated separately by Pingouin
    on ranked values. The legacy conditional trend lines used OLS without an
    intercept; this function preserves that plotting behavior explicitly.
    """
    y = np.asarray(y, dtype=np.float64)
    covariate = np.asarray(covariate, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(covariate)
    if not np.all(valid):
        raise ValueError("Residual vectors require the shared complete-case rows")
    coefficient = np.linalg.lstsq(covariate[:, None], y, rcond=None)[0][0]
    return y - coefficient * covariate


def figure5_plot_vectors(ndvi, sif) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    """Return legacy vectors, grouped by outcome then TH,TH|LST,TC,TC|LST."""
    required = {
        "ndvi": {"TH", "TC", "LST", "NDVI_res"},
        "sif": {"TH", "TC", "LST", "SIF_res"},
    }
    for label, frame in (("ndvi", ndvi), ("sif", sif)):
        missing = required[label] - set(frame.columns)
        if missing:
            raise ValueError(f"{label} table missing columns: {sorted(missing)}")
    x_by_outcome = []
    y_by_outcome = []
    for frame, response in ((ndvi, "NDVI_res"), (sif, "SIF_res")):
        x_by_outcome.append(
            [
                frame["TH"].to_numpy(dtype=np.float64),
                residualize_without_intercept(frame["TH"], frame["LST"]),
                frame["TC"].to_numpy(dtype=np.float64),
                residualize_without_intercept(frame["TC"], frame["LST"]),
            ]
        )
        y_by_outcome.append(
            [
                frame[response].to_numpy(dtype=np.float64),
                residualize_without_intercept(frame[response], frame["LST"]),
                frame[response].to_numpy(dtype=np.float64),
                residualize_without_intercept(frame[response], frame["LST"]),
            ]
        )
    return x_by_outcome, y_by_outcome


def compute_figure5_correlations(ndvi, sif):
    """Use the installed Pingouin implementation, with no shim/fallback.

    Returns rows ordered as Fig. 5a-d panels, NDVI resistance then SIF
    resistance. The raw coefficients are preserved; notebook labels round to
    three decimals to match the source figure code.
    """
    import pandas as pd
    import pingouin as pg

    specifications = (
        ("TH", "", ""),
        ("TH", "LST", "|LST"),
        ("TC", "", ""),
        ("TC", "LST", "|LST"),
    )
    rows = []
    for panel, (predictor, covariate, condition_label) in enumerate(specifications, 1):
        for outcome, frame in (("NDVI_res", ndvi), ("SIF_res", sif)):
            covars = [] if not covariate else covariate
            result = pg.partial_corr(
                data=frame,
                x=predictor,
                y=outcome,
                covar=covars,
                method="spearman",
            ).iloc[0]
            rows.append(
                {
                    "panel": panel,
                    "predictor": predictor,
                    "condition": condition_label,
                    "outcome": outcome,
                    "n": int(result["n"]),
                    "r": float(result["r"]),
                    "p_value": float(result["p-val"]),
                }
            )
    return pd.DataFrame(rows)


def correlation_results_to_arrays(results) -> dict[str, np.ndarray]:
    return {
        "correlation_panel": results["panel"].to_numpy(dtype=np.int64),
        "correlation_predictor": results["predictor"].to_numpy(dtype="U8"),
        "correlation_condition": results["condition"].to_numpy(dtype="U16"),
        "correlation_outcome": results["outcome"].to_numpy(dtype="U16"),
        "correlation_n": results["n"].to_numpy(dtype=np.int64),
        "correlation_r": results["r"].to_numpy(dtype=np.float64),
        "correlation_p": results["p_value"].to_numpy(dtype=np.float64),
    }
