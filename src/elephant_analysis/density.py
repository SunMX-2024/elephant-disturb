"""Density-panel calculations for Fig. 2a-d.

Country and patch observations are supplied in the density input bundle.
This module calculates the figure statistics from those non-raster inputs;
the upstream observation compilation is outside the raster workflow.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


COUNTRY_DENSITY_KEY = "ele_veg_dens_250613"
COUNTRY_VALUE_KEY = "ele_veg_val_250613"
COUNTRY_SE_KEY = "ele_veg_std_250613"
COUNTRY_TABLE_KEY = "country_density_rows"
COUNTRY_COLUMNS_KEY = "country_density_columns"
PATCH_TABLE_KEY = "patch_density_rows"
PATCH_COLUMNS_KEY = "patch_density_columns"


def _table(bundle: dict[str, Any], rows_key: str, columns_key: str):
    import pandas as pd

    rows = np.asarray(bundle[rows_key], dtype=np.float64)
    columns = [str(x) for x in np.asarray(bundle[columns_key]).tolist()]
    if rows.ndim != 2 or rows.shape[1] != len(columns):
        raise ValueError(f"{rows_key} must be a 2D matrix matching {columns_key}")
    return pd.DataFrame(rows, columns=columns)


def _as_two_series(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] != 2:
        raise ValueError(f"{name} must contain two rows: tree height and tree cover")
    return array


def _safe_nanstd(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    mean = np.nanmean(values)
    return float(np.sqrt(np.nanmean((values - mean) ** 2)))


def adaptive_bin_stats(
    x: Any,
    y: Any,
    min_pts: int = 20,
    min_bins: int = 5,
    max_bins: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match the adaptive quantile bins used in the current Fig. 2c-d cell."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    n = x.size
    if n < max(3, min_pts):
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, empty, empty

    n_bins = int(np.clip(n // max(1, min_pts), min_bins, max_bins))
    unique_x = np.unique(x)
    n_bins = int(np.clip(n_bins, 1, max(1, unique_x.size - 1)))
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)

    indices = np.digitize(x, edges) - 1
    means_x: list[float] = []
    means_y: list[float] = []
    errors: list[float] = []
    counts: list[int] = []
    for i in range(n_bins):
        selected = indices == i
        count = int(selected.sum())
        if count == 0:
            continue
        means_x.append(float(np.nanmean(x[selected])))
        means_y.append(float(np.nanmean(y[selected])))
        std = _safe_nanstd(y[selected])
        errors.append(float(std / np.sqrt(count)) if count > 1 else float("nan"))
        counts.append(count)
    return (
        np.asarray(means_x),
        np.asarray(means_y),
        np.asarray(errors),
        np.asarray(counts, dtype=np.float64),
        edges,
    )


def permutation_slope_p(x: Any, y: Any, n_perm: int = 5000, seed: int = 42) -> float:
    """Two-sided permutation p-value for the OLS slope, with add-one correction."""
    from scipy.stats import linregress

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3 or np.unique(x).size < 2:
        return float("nan")
    slope0 = float(linregress(x, y).slope)
    rng = np.random.RandomState(seed)
    exceedances = 0
    for _ in range(n_perm):
        permuted_slope = float(linregress(x, rng.permutation(y)).slope)
        exceedances += abs(permuted_slope) >= abs(slope0)
    return float((exceedances + 1) / (n_perm + 1))


def wilcoxon_mean_p(values: Any) -> float:
    """Two-sided Wilcoxon signed-rank test against zero, matching the notebook."""
    from scipy.stats import wilcoxon

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    nonzero = values[values != 0]
    if nonzero.size < 5:
        return float("nan")
    try:
        result = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox", mode="auto")
    except TypeError:
        result = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox", method="auto")
    return float(result.pvalue)


def _ols_fit(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    import statsmodels.api as sm
    from sklearn.metrics import r2_score

    design = sm.add_constant(x)
    model = sm.OLS(y, design).fit()
    prediction = model.get_prediction(sm.add_constant(x_grid)).summary_frame(alpha=0.05)
    r2 = float(r2_score(y, model.predict(design)))
    return (
        prediction["mean"].to_numpy(),
        prediction["mean_ci_lower"].to_numpy(),
        prediction["mean_ci_upper"].to_numpy(),
        float(model.params[1]),
        r2,
    )


def _fit_arrays(
    prefix: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    draw_if_nonsignificant: bool,
    grid_max: Optional[float] = None,
    n_perm: int = 5000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[valid], np.asarray(y)[valid]
    p_fit = permutation_slope_p(x, y, n_perm=n_perm, seed=seed)
    p_mean = wilcoxon_mean_p(y)
    if x.size < 3 or np.unique(x).size < 2:
        grid = np.array([], dtype=np.float64)
        mean = low = high = grid
        slope = r2 = float("nan")
    else:
        xmax = float(np.max(x)) if grid_max is None else grid_max
        grid = np.linspace(float(np.min(x)), xmax, 400)
        mean, low, high, slope, r2 = _ols_fit(x, y, grid)
        if not draw_if_nonsignificant and p_fit >= 0.05:
            grid = np.array([], dtype=np.float64)
            mean = low = high = grid
    return {
        f"{prefix}_fit_x": grid,
        f"{prefix}_fit_y": mean,
        f"{prefix}_fit_ci_low": low,
        f"{prefix}_fit_ci_high": high,
        f"{prefix}_stats": np.asarray([slope, r2, p_fit, p_mean], dtype=np.float64),
    }


def compute_density_results(
    bundle: dict[str, Any], n_perm: int = 5000, seed: int = 42
) -> dict[str, np.ndarray]:
    """Compute plot-ready arrays for country and patch density panels.

    The result contains only numeric and Unicode arrays and can be stored with
    ``elephant_analysis.artifacts.save_arrays`` without pickle/object arrays.
    ``*_stats`` order is slope, R-squared, permutation p, and mean signed-rank p.
    """
    import pandas as pd

    density = _as_two_series(bundle[COUNTRY_DENSITY_KEY], COUNTRY_DENSITY_KEY)
    means = _as_two_series(bundle[COUNTRY_VALUE_KEY], COUNTRY_VALUE_KEY)
    errors = _as_two_series(bundle[COUNTRY_SE_KEY], COUNTRY_SE_KEY)
    if density.shape != means.shape or means.shape != errors.shape:
        raise ValueError("Country summary arrays must have identical shapes")
    country = _table(bundle, COUNTRY_TABLE_KEY, COUNTRY_COLUMNS_KEY)
    patches = _table(bundle, PATCH_TABLE_KEY, PATCH_COLUMNS_KEY)
    for frame, label in ((country, "country"), (patches, "patch")):
        missing = {"TH", "TC", "Ele_Dens"} - set(frame.columns)
        if missing:
            raise ValueError(f"{label} table missing columns: {sorted(missing)}")

    results: dict[str, np.ndarray] = {
        "country_density": density,
        "country_mean": means,
        "country_se": errors,
        "country_th": country["TH"].to_numpy(dtype=np.float64),
        "country_tc": country["TC"].to_numpy(dtype=np.float64),
        "patch_rows": patches.to_numpy(dtype=np.float64),
        "patch_columns": np.asarray(patches.columns, dtype="U32"),
    }
    for i, label in enumerate(("th", "tc")):
        results.update(
            _fit_arrays(
                f"country_{label}",
                density[i],
                means[i],
                draw_if_nonsignificant=False,
                n_perm=n_perm,
                seed=seed,
            )
        )
        results.update(
            _fit_arrays(
                f"patch_{label}",
                patches["Ele_Dens"].to_numpy(dtype=np.float64),
                patches["TH" if label == "th" else "TC"].to_numpy(dtype=np.float64),
                draw_if_nonsignificant=True,
                grid_max=3.3,
                n_perm=n_perm,
                seed=seed,
            )
        )
        xm, ym, se, count, edges = adaptive_bin_stats(
            patches["Ele_Dens"].to_numpy(dtype=np.float64),
            patches["TH" if label == "th" else "TC"].to_numpy(dtype=np.float64),
            min_pts=15,
            min_bins=5,
            max_bins=50,
        )
        results.update(
            {
                f"patch_{label}_bin_x": xm,
                f"patch_{label}_bin_y": ym,
                f"patch_{label}_bin_se": se,
                f"patch_{label}_bin_count": count,
                f"patch_{label}_bin_edges": edges,
            }
        )
    return results
