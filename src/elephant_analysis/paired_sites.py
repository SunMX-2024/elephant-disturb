"""Source-faithful gridding and paired-site calculations.

The input aliases and transformations follow the archived Revise_R2 notebooks.
The emitted array keys are the original NPY filename stems so they can be
compared directly with the archived numeric results.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import stats


REQUIRED_RASTERS: dict[str, dict[str, str]] = {
    "deaa": {"path": "SPEI_deaa001_240504.tif"},
    "daa": {"path": "SPEI_daa001_240504.tif"},
    "deaa_1214": {"path": "drought-elephant_area_12-14_241126.tif"},
    "daa_1214": {"path": "drought-nonelephant_area_12-14_241126.tif"},
    "protected_area": {"path": "WDPA_WDOECM_poly_Nov2023_AF_001_afr.tif"},
    "dem": {"path": "gtopoDEM_afr_001.tif"},
    "slope": {"path": "gtopoSlope_afr_001.tif"},
    "human_footprint": {"path": "hfp2018_Afirca_001_230525.tif"},
    "tree_height": {"path": "African_th2019_001_230525.tif"},
    "tree_cover": {"path": "Africa_tc2019_mosaic_001_fillNaN_231215.tif"},
    "lst_1516": {"path": "lst_1516_afr_001.tif"},
    "lst_1214": {"path": "MOD11_mean_2012-14_001.tif"},
    "spei": {"path": "SPEI_yrMean_afr_1516_001_sub.tif"},
    "ndvi_resistance": {"path": "ndvi_resistance_230321.tif"},
    "sif_resistance": {"path": "csif_resistance_230321.tif"},
    "evapotranspiration": {"path": "evaporation_mean_2015-16_001_afr.tif"},
    "albedo_2015": {"path": "MCD_albedo_WSA_2015_001.tif"},
    "albedo_2016": {"path": "MCD_albedo_WSA_2016_001.tif"},
    "elephant_density": {"path": "ele-density_001_231123.tif"},
}

VARIABLES = ("treeH", "tc", "lst", "spei", "ndvi_res", "csif_res", "et", "albedo", "dens", "dem", "slope")
MAIN_VARIABLES = ("treeH", "tc", "lst", "spei", "et", "albedo")
RESISTANCE_VARIABLES = ("ndvi_res", "csif_res")
STATES = ("protected", "non-protected")
DISTURBANCE_STATES = (("DEAA", "deaa"), ("DAA", "daa"))

_RAW_VARIABLES = {
    "treeH": "tree_height",
    "tc": "tree_cover",
    "lst": "lst_1516",
    "spei": "spei",
    "ndvi_res": "ndvi_resistance",
    "csif_res": "sif_resistance",
    "et": "evapotranspiration",
    "dens": "elephant_density",
}

_DEFAULT_CONFIG = {
    "block_size": 25,
    "minimum_valid_fraction": 0.05,
    "dem_difference_limit": 200.0,
    "slope_difference_limit": 10.0,
}


def slide_window_mean(
    disturbance: np.ndarray,
    variable: np.ndarray,
    block_size: int = 25,
    minimum_valid_fraction: float = 0.05,
) -> np.ndarray:
    """Calculate cell means, falling back to the best covered adjacent cell.

    Cells are formed by non-overlapping ``block_size`` windows. If the current
    window has no more than the configured valid fraction, the source's eight
    neighboring windows are checked in their original tie-breaking order. The
    final incomplete row or column is ignored, as in the source notebook.
    """
    disturbance = np.asarray(disturbance)
    variable = np.asarray(variable)
    if disturbance.ndim != 2 or variable.ndim != 2:
        raise ValueError("disturbance and variable inputs must be two-dimensional")
    if disturbance.shape != variable.shape:
        raise ValueError("disturbance and variable inputs must have identical shapes")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if not 0 <= minimum_valid_fraction <= 1:
        raise ValueError("minimum_valid_fraction must be between 0 and 1")

    nrows = disturbance.shape[0] // block_size
    ncols = disturbance.shape[1] // block_size
    if nrows == 0 or ncols == 0:
        raise ValueError("input is smaller than one complete block")

    result = np.full((nrows, ncols), np.nan, dtype=float)
    block_pixels = block_size * block_size
    # Keep the exact source order; np.argmax selects the first maximum.
    offsets = ((-1, 0), (0, -1), (-1, -1), (1, 0), (0, 1), (1, 1), (-1, 1), (1, -1))

    def block_mask(row: int, col: int) -> np.ndarray:
        row_start, row_end = row * block_size, (row + 1) * block_size
        col_start, col_end = col * block_size, (col + 1) * block_size
        return np.multiply(
            disturbance[row_start:row_end, col_start:col_end],
            variable[row_start:row_end, col_start:col_end],
        )

    for row in range(nrows):
        for col in range(ncols):
            mask = block_mask(row, col)
            count = np.count_nonzero(~np.isnan(mask))
            if count / block_pixels > minimum_valid_fraction:
                result[row, col] = np.nanmean(mask)
                continue

            neighbor_masks = [block_mask(row + dr, col + dc) for dr, dc in offsets]
            neighbor_counts = [np.count_nonzero(~np.isnan(item)) / block_pixels for item in neighbor_masks]
            best = int(np.argmax(neighbor_counts))
            if neighbor_counts[best] > minimum_valid_fraction:
                result[row, col] = np.nanmean(neighbor_masks[best])

    return result


def geo_mask(
    dem_deaa: np.ndarray,
    dem_daa: np.ndarray,
    slope_deaa: np.ndarray,
    slope_daa: np.ndarray,
    dem_difference_limit: float = 200.0,
    slope_difference_limit: float = 10.0,
) -> np.ndarray:
    """Keep paired cells within the source DEM and slope difference limits."""
    arrays = [np.asarray(value, dtype=float) for value in (dem_deaa, dem_daa, slope_deaa, slope_daa)]
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("all DEM and slope arrays must have identical shapes")
    dem_difference = arrays[0] - arrays[1]
    slope_difference = arrays[2] - arrays[3]
    dem_difference[(dem_difference < -dem_difference_limit) | (dem_difference > dem_difference_limit)] = np.nan
    slope_difference[(slope_difference < -slope_difference_limit) | (slope_difference > slope_difference_limit)] = np.nan
    dem_difference[~np.isnan(dem_difference)] = 1
    slope_difference[~np.isnan(slope_difference)] = 1
    return np.ones(arrays[0].shape, dtype=float) * dem_difference * slope_difference


def _copy_float(raw: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    if name not in raw:
        raise KeyError(f"missing paired raster alias: {name}")
    value = np.asarray(raw[name])
    if not np.issubdtype(value.dtype, np.floating):
        value = value.astype(float)
    else:
        value = value.copy()
    if value.ndim != 2:
        raise ValueError(f"raw[{name!r}] must be two-dimensional")
    return value


def _wilcoxon_pvalue(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first).ravel()
    second = np.asarray(second).ravel()
    if first.shape != second.shape:
        raise ValueError("paired arrays must have identical shapes")
    keep = ~(np.isnan(first) | np.isnan(second))
    if not np.any(keep):
        return float("nan")
    return float(stats.wilcoxon(first[keep], second[keep]).pvalue)


def _output_stem(area: str, disturbance: str, variable: str) -> str:
    return f"Vegetation_structure_{area}_{disturbance}_{variable}_0.25deg_25613"


def _add_state_array(
    output: dict[str, np.ndarray], area: str, disturbance: str, variable: str, values: np.ndarray
) -> None:
    output[_output_stem(area, disturbance, variable)] = values


def _preprocess(raw: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Apply the source notebook's raster masks and scale transformations."""
    deaa = _copy_float(raw, "deaa")
    daa = _copy_float(raw, "daa")
    deaa[deaa == 0] = np.nan
    daa[daa == 0] = np.nan

    protected = _copy_float(raw, "protected_area")
    protected[protected > 0] = 1
    nonprotected = np.ones(protected.shape, dtype=np.float32) - protected
    protected = protected.astype(np.float32, copy=False)
    protected[protected == 0] = np.nan
    nonprotected[nonprotected == 0] = np.nan

    dem = _copy_float(raw, "dem").astype(np.float32)
    dem[dem == -9999] = np.nan
    slope = _copy_float(raw, "slope")
    slope[slope == 0] = np.nan

    human_footprint = _copy_float(raw, "human_footprint")
    human_footprint[human_footprint <= 15] = 1
    human_footprint[human_footprint > 1] = np.nan

    variables: dict[str, np.ndarray] = {}
    for output_name, raw_name in _RAW_VARIABLES.items():
        variable = _copy_float(raw, raw_name)
        if output_name == "treeH":
            variable[variable == 0] = np.nan
        elif output_name == "tc":
            variable = variable.astype(np.float32, copy=False)
            variable[variable == 0] = np.nan
            variable[variable > 100] = np.nan
        elif output_name == "lst":
            variable[variable <= 0] = np.nan
            variable = variable * 0.02 - 273.15
        elif output_name in {"spei", "ndvi_res", "csif_res", "et"}:
            variable[variable == 0] = np.nan
        variables[output_name] = variable

    albedo_2015 = _copy_float(raw, "albedo_2015") / 1000
    albedo_2016 = _copy_float(raw, "albedo_2016") / 1000
    albedo = (albedo_2015 + albedo_2016) / 2
    albedo[albedo == 0] = np.nan
    variables["albedo"] = albedo
    variables["dem"] = dem
    variables["slope"] = slope

    disturbance_masks = {}
    for area, area_mask in (("protected", protected), ("non-protected", nonprotected)):
        area_mask = area_mask * human_footprint
        for disturbance_name, disturbance in (("DEAA", deaa), ("DAA", daa)):
            disturbance_masks[f"{area}:{disturbance_name}"] = disturbance * area_mask
    return variables, disturbance_masks


def _slide_state_arrays(
    output: dict[str, np.ndarray],
    variables: Mapping[str, np.ndarray],
    disturbance_masks: Mapping[str, np.ndarray],
    variable_names: tuple[str, ...],
    disturbance_suffix: str = "",
    block_size: int = 25,
    minimum_valid_fraction: float = 0.05,
) -> None:
    for area in STATES:
        for disturbance_name, _ in DISTURBANCE_STATES:
            mask = disturbance_masks[f"{area}:{disturbance_name}{disturbance_suffix}"]
            for variable_name in variable_names:
                output_name = f"{disturbance_name}{disturbance_suffix}"
                output[_output_stem(area, output_name, variable_name)] = slide_window_mean(
                    mask,
                    variables[variable_name],
                    block_size=block_size,
                    minimum_valid_fraction=minimum_valid_fraction,
                )


def _paired_feature_arrays(
    output: dict[str, np.ndarray],
    area: str,
    state_key: str,
    disturbance_prefix: str,
    geo: np.ndarray,
    reduction_masks: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], list[float]]:
    values: list[np.ndarray] = []
    pvalues: list[float] = []
    for variable in MAIN_VARIABLES:
        deaa = output[_output_stem(area, f"DEAA{disturbance_prefix}", variable)] * geo
        daa = output[_output_stem(area, f"DAA{disturbance_prefix}", variable)] * geo
        # The source deliberately stores the mean SPEI level here, not DEAA - DAA.
        delta = (deaa + daa) / 2 if variable == "spei" else deaa - daa
        values.append(delta)
        pvalues.append(_wilcoxon_pvalue(deaa, daa))
        if reduction_masks is not None:
            for mask_index, mask in enumerate(reduction_masks):
                resistance = RESISTANCE_VARIABLES[mask_index]
                output[f"delta_{state_key}_{resistance}_reduction_{variable}"] = delta * mask
    return values, pvalues


def _paired_resistance_arrays(
    output: dict[str, np.ndarray],
    area: str,
    state_key: str,
    geo: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], list[float]]:
    deltas: list[np.ndarray] = []
    reductions: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    pvalues: list[float] = []
    reduction_pvalues: list[float] = []
    for variable in RESISTANCE_VARIABLES:
        deaa = output[_output_stem(area, f"DEAA", variable)] * geo
        daa = output[_output_stem(area, f"DAA", variable)] * geo
        delta = deaa - daa
        reduction_mask = daa.copy()
        reduction_mask[reduction_mask > 1] = np.nan
        reduction_mask[reduction_mask <= 1] = 1
        reduced_deaa = deaa * reduction_mask
        reduced_daa = daa * reduction_mask
        reduced_delta = delta * reduction_mask
        deltas.append(delta)
        reductions.append(reduced_delta)
        masks.append(reduction_mask)
        pvalues.append(_wilcoxon_pvalue(deaa, daa))
        reduction_pvalues.append(_wilcoxon_pvalue(reduced_deaa, reduced_daa))
    output[f"delta_res_{state_key}_250613"] = np.stack(deltas)
    output[f"delta_res_reduc_{state_key}_250613"] = np.stack(reductions)
    output[f"mask_reduc_{state_key}_250613"] = np.stack(masks)
    return deltas, reductions, pvalues, reduction_pvalues


def compute_paired(raw: dict[str, np.ndarray], config: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    """Compute the paired-site arrays from the original raster inputs.

    ``raw`` keys are the aliases in :data:`REQUIRED_RASTERS`. No data are read
    at import time. The returned mapping includes state-level grids, paired
    deltas, p-values, reduction masks, and the 2012--14 LST branch.
    """
    options = {**_DEFAULT_CONFIG, **(config or {})}
    block_size = int(options["block_size"])
    minimum_valid_fraction = float(options["minimum_valid_fraction"])
    dem_limit = float(options["dem_difference_limit"])
    slope_limit = float(options["slope_difference_limit"])
    variables, disturbance_masks = _preprocess(raw)
    output: dict[str, np.ndarray] = {}

    _slide_state_arrays(
        output,
        variables,
        disturbance_masks,
        VARIABLES,
        block_size=block_size,
        minimum_valid_fraction=minimum_valid_fraction,
    )

    geos: dict[str, np.ndarray] = {}
    for area in STATES:
        geos[area] = geo_mask(
            output[_output_stem(area, "DEAA", "dem")],
            output[_output_stem(area, "DAA", "dem")],
            output[_output_stem(area, "DEAA", "slope")],
            output[_output_stem(area, "DAA", "slope")],
            dem_difference_limit=dem_limit,
            slope_difference_limit=slope_limit,
        )
        output[f"geo_mask_{area.replace('-', '_')}"] = geos[area]

    for area, state_key in (("protected", "protect"), ("non-protected", "unprotect")):
        reduction_masks: list[np.ndarray] = []
        for variable in RESISTANCE_VARIABLES:
            daa = output[_output_stem(area, "DAA", variable)] * geos[area]
            mask = daa.copy()
            mask[mask > 1] = np.nan
            mask[mask <= 1] = 1
            reduction_masks.append(mask)
        values, pvalues = _paired_feature_arrays(
            output,
            area,
            state_key,
            "",
            geos[area],
            reduction_masks=reduction_masks,
        )
        output[f"delta_{state_key}_250613"] = np.stack(values)
        output[f"delta_{state_key}_pvalue_250613"] = np.asarray(pvalues, dtype=float)
        resistance = _paired_resistance_arrays(output, area, state_key, geos[area])
        output[f"delta_res_{state_key}_pvalue_250613"] = np.asarray(resistance[2], dtype=float)
        output[f"delta_res_reduc_{state_key}_pvalue_250613"] = np.asarray(resistance[3], dtype=float)

    # The separate 2012--14 notebook reused the same masks, gridding and filters.
    deaa_1214 = _copy_float(raw, "deaa_1214")
    daa_1214 = _copy_float(raw, "daa_1214")
    deaa_1214[deaa_1214 == 0] = np.nan
    daa_1214[daa_1214 == 0] = np.nan
    lst_1214 = _copy_float(raw, "lst_1214")
    lst_1214[lst_1214 <= 0] = np.nan
    lst_1214 = lst_1214 * 0.02 - 273.15
    variables_1214 = {"lst1214": lst_1214, "dem": variables["dem"], "slope": variables["slope"]}
    masks_1214 = {}
    for area in STATES:
        area_mask = _copy_float(raw, "protected_area")
        area_mask[area_mask > 0] = 1
        if area == "protected":
            area_mask[area_mask == 0] = np.nan
        else:
            area_mask = np.ones(area_mask.shape, dtype=np.float32) - area_mask
            area_mask[area_mask == 0] = np.nan
        hfp = _copy_float(raw, "human_footprint")
        hfp[hfp <= 15] = 1
        hfp[hfp > 1] = np.nan
        area_mask *= hfp
        masks_1214[f"{area}:DEAA1214"] = deaa_1214 * area_mask
        masks_1214[f"{area}:DAA1214"] = daa_1214 * area_mask
    _slide_state_arrays(
        output,
        variables_1214,
        masks_1214,
        ("lst1214", "dem", "slope"),
        disturbance_suffix="1214",
        block_size=block_size,
        minimum_valid_fraction=minimum_valid_fraction,
    )
    lst_deltas: dict[str, np.ndarray] = {}
    lst_pvalues = []
    for area, state_key in (("protected", "protect"), ("non-protected", "unprotect")):
        geo = geo_mask(
            output[_output_stem(area, "DEAA1214", "dem")],
            output[_output_stem(area, "DAA1214", "dem")],
            output[_output_stem(area, "DEAA1214", "slope")],
            output[_output_stem(area, "DAA1214", "slope")],
            dem_difference_limit=dem_limit,
            slope_difference_limit=slope_limit,
        )
        output[f"geo_mask_1214_{area.replace('-', '_')}"] = geo
        deaa = output[_output_stem(area, "DEAA1214", "lst1214")] * geo
        daa = output[_output_stem(area, "DAA1214", "lst1214")] * geo
        lst_deltas[state_key] = deaa - daa
        lst_pvalues.append(_wilcoxon_pvalue(deaa, daa))
    output["delta_lst1214_protect_250613"] = lst_deltas["protect"]
    output["delta_lst1214_unprotect_250613"] = lst_deltas["unprotect"]
    output["delta_lst1214_protect_pvalue_250613"] = np.asarray(lst_pvalues, dtype=float)

    return output
