"""SEM sample construction, reference fit loading, and the Python-R bridge."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import numpy as np


SEM_COLUMNS = ["NDVI_res", "SIF_res", "TH", "TC", "LST", "ET", "Albedo", "SPEI_m", "Ele_Dens"]
SEM_BOOTSTRAP = 5000
FRESH_RUN_SEED = 20261107


def verify_sem_reference_samples(data, reference) -> None:
    """Reject use of archived coefficients with different prepared observations.

    Compare sample values and column ordering with the saved input matrix.
    This check does not refit the model or repeat bootstrap draws.
    """
    columns = reference["sem_input_columns"].astype(str).tolist()
    values = reference["sem_input_values"]
    current = data.to_numpy(dtype=np.float64)
    if (list(data.columns) != columns or current.shape != values.shape or
            not np.array_equal(current, values, equal_nan=True)):
        raise ValueError("Current SEM samples differ from the archived input matrix; "
                         "set COMPUTE_SEM=True to fit these samples rather than reuse archived coefficients")


def _stem(group: str, state: str, variable: str) -> str:
    return f"Vegetation_structure_{group}_{state}_{variable}_0.25deg_25613"


def _paired_array(paired: dict[str, Any], group: str, state: str, variable: str) -> np.ndarray:
    key = _stem(group, state, variable)
    if key not in paired:
        raise KeyError(f"paired input missing original array stem {key}")
    return np.asarray(paired[key], dtype=np.float64)


def _array_pair(paired, group, variable):
    affected = _paired_array(paired, group, "DEAA", variable)
    comparison = _paired_array(paired, group, "DAA", variable)
    if affected.shape != comparison.shape:
        raise ValueError(f"shape mismatch for {group}/{variable}: {affected.shape} vs {comparison.shape}")
    return affected, comparison


def _geographic_mask(paired, group) -> np.ndarray:
    dem_a, dem_b = _array_pair(paired, group, "dem")
    slope_a, slope_b = _array_pair(paired, group, "slope")
    dem_delta = dem_a - dem_b
    slope_delta = slope_a - slope_b
    mask = np.ones(dem_delta.shape, dtype=np.float64)
    mask[(dem_delta < -200) | (dem_delta > 200)] = np.nan
    mask[(slope_delta < -10) | (slope_delta > 10)] = np.nan
    mask[np.isnan(dem_delta) | np.isnan(slope_delta)] = np.nan
    return mask


def _reduction_mask(paired, group, variable, geographic_mask) -> np.ndarray:
    _, reference = _array_pair(paired, group, variable)
    reference = reference * geographic_mask
    mask = np.ones(reference.shape, dtype=np.float64)
    mask[reference > 1] = np.nan
    mask[reference <= 1] = 1
    return mask


def _paired_fields(paired, group, geographic_mask, reduction_mask) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    for variable in ("ndvi_res", "csif_res", "treeH", "tc", "lst", "spei", "et", "albedo"):
        affected, comparison = _array_pair(paired, group, variable)
        affected = affected * geographic_mask
        comparison = comparison * geographic_mask
        if variable == "spei":
            value = (affected + comparison) / 2
        else:
            value = affected - comparison
        fields[variable] = value * reduction_mask

    # Historical SEM density is the elephant-side density itself, not a paired
    # density difference; preserve that source definition.
    density = _paired_array(paired, group, "DEAA", "dens")
    fields["Ele_Dens"] = density * geographic_mask * reduction_mask
    return fields


def _complete_case(values: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    ordered = list(values)
    shapes = {values[key].shape for key in ordered}
    if len(shapes) != 1:
        raise ValueError("all paired pathway variables must share the same grid shape")
    valid = np.ones(values[ordered[0]].shape, dtype=bool)
    for value in values.values():
        valid &= ~np.isnan(value)
    return {key: values[key][valid].ravel() for key in ordered}, valid


def _combine_strata(protected, unprotected, columns):
    import pandas as pd

    joined = {
        column: np.concatenate((protected[column], unprotected[column]))
        for column in columns
    }
    lengths = {len(value) for value in joined.values()}
    if len(lengths) != 1:
        raise ValueError("complete-case row counts differ among pathway variables")
    return pd.DataFrame(joined, columns=columns)


def build_pathway_tables(paired: dict[str, Any]) -> dict[str, Any]:
    """Build Fig. 5a-d and SEM tables from the original paired array stems.

    The masks and ordering reproduce Main_figure cells 35-42 and the SEM
    producer in Revise_R2 cells 50-55. `ndvi` and `sif` use their own
    reduction masks. The SEM uses the common NDVI-and-SIF reduction mask,
    then complete cases across all nine SEM variables. Protected rows precede
    unprotected rows; cells are flattened in NumPy C order.
    """
    strata = {}
    ndvi_fields = {}
    sif_fields = {}
    sem_fields = {}

    for group in ("protected", "non-protected"):
        geo = _geographic_mask(paired, group)
        ndvi_mask = _reduction_mask(paired, group, "ndvi_res", geo)
        sif_mask = _reduction_mask(paired, group, "csif_res", geo)
        ndvi_fields[group] = _paired_fields(paired, group, geo, ndvi_mask)
        sif_fields[group] = _paired_fields(paired, group, geo, sif_mask)
        both_mask = ndvi_mask * sif_mask
        sem_fields[group] = _paired_fields(paired, group, geo, both_mask)
        strata[group] = {"geographic": geo, "ndvi": ndvi_mask, "sif": sif_mask, "both": both_mask}

    figure_columns = ("NDVI_res", "TH", "TC", "LST", "SPEI_m")
    ndvi_by_group = {}
    sif_by_group = {}
    sem_by_group = {}
    for group in ("protected", "non-protected"):
        ndvi_by_group[group], _ = _complete_case(
            {name: ndvi_fields[group][key] for name, key in zip(figure_columns, ("ndvi_res", "treeH", "tc", "lst", "spei"))}
        )
        sif_by_group[group], _ = _complete_case(
            {name: sif_fields[group][key] for name, key in zip(("SIF_res", "TH", "TC", "LST", "SPEI_m"), ("csif_res", "treeH", "tc", "lst", "spei"))}
        )
        sem_by_group[group], _ = _complete_case(
            {name: sem_fields[group][key] for name, key in zip(SEM_COLUMNS, ("ndvi_res", "csif_res", "treeH", "tc", "lst", "et", "albedo", "spei", "Ele_Dens"))}
        )

    return {
        "ndvi": _combine_strata(ndvi_by_group["protected"], ndvi_by_group["non-protected"], figure_columns),
        "sif": _combine_strata(sif_by_group["protected"], sif_by_group["non-protected"], ("SIF_res", "TH", "TC", "LST", "SPEI_m")),
        "sem": _combine_strata(sem_by_group["protected"], sem_by_group["non-protected"], SEM_COLUMNS),
        "masks": strata,
    }


def pathway_tables_to_arrays(tables: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name in ("ndvi", "sif", "sem"):
        frame = tables[name]
        arrays[f"{name}_values"] = frame.to_numpy(dtype="<f8")
        arrays[f"{name}_columns"] = np.asarray(frame.columns, dtype="U24")
    for group in ("protected", "non-protected"):
        for mask_name in ("geographic", "ndvi", "sif", "both"):
            arrays[f"mask_{group}_{mask_name}"] = np.asarray(tables["masks"][group][mask_name], dtype="<f8")
    return arrays


def pathway_tables_from_arrays(arrays: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    result = {}
    for name in ("ndvi", "sif", "sem"):
        columns = [str(value) for value in np.asarray(arrays[f"{name}_columns"]).tolist()]
        result[name] = pd.DataFrame(np.asarray(arrays[f"{name}_values"], dtype=np.float64), columns=columns)
    return result


def load_sem_reference(bundle: dict[str, Any]) -> dict[str, Any]:
    """Return the saved lavaan parameter rows from sem_reference.npz."""
    names = ("lhs", "op", "rhs")
    paths = {name: np.asarray(bundle[f"reference_{name}"]).astype(str) for name in names}
    for name in ("estimate", "se", "z", "pvalue", "std_lv", "std_all", "std_nox", "std_all_se"):
        paths[name] = np.asarray(bundle[f"reference_{name}"], dtype=np.float64)
    if len({len(value) for value in paths.values()}) != 1:
        raise ValueError("SEM reference arrays must have matching path counts")
    return {
        "coefficients": paths,
        "fit_measures": {},
        "software": {"lavaan": "0.6-15", "R": "not recorded in source fit"},
        "source_status": "frozen local SEM fit export; exact fit indices were not preserved in the source bundle",
    }


def sem_result_to_arrays(result: dict[str, Any]) -> dict[str, np.ndarray]:
    coefficients = result["coefficients"]
    fields = ("lhs", "op", "rhs", "estimate", "se", "z", "pvalue", "std_all", "std_all_se")
    if isinstance(coefficients, list):
        values = {name: [row.get(name) for row in coefficients] for name in fields}
    else:
        values = coefficients
    arrays = {f"sem_{name}": np.asarray(values[name]) for name in ("lhs", "op", "rhs")}
    arrays.update(
        {
            f"sem_{name}": np.asarray(values[name], dtype=np.float64)
            for name in fields[3:]
        }
    )
    fit_measures = result.get("fit_measures", {})
    arrays["fit_measure_names"] = np.asarray(list(fit_measures), dtype="U24")
    arrays["fit_measure_values"] = np.asarray(list(fit_measures.values()), dtype=np.float64)
    arrays["sem_software"] = np.asarray(
        [result.get("software", {}).get("R", ""), result.get("software", {}).get("lavaan", "")],
        dtype="U64",
    )
    arrays["sem_run_metadata"] = np.asarray(
        [result.get("seed", -1), result.get("bootstrap", 0)], dtype=np.int64
    )
    return arrays


def sem_path_records(arrays: dict[str, Any]) -> list[dict[str, Any]]:
    """Map saved lavaan rows to directed paths by variable names, not row order.

    For ``lhs ~ rhs`` the arrow is rhs -> lhs; for ``lhs =~ rhs`` it is
    latent lhs -> indicator rhs. Coefficients are std.all and p-values are
    those of the exported parameter estimates. Covariances are not included
    in the saved directed-path export.
    """
    expected = (
        ("tree", "=~", "TH"), ("tree", "=~", "TC"),
        ("tree", "~", "Ele_Dens"), ("ET", "~", "tree"),
        ("Albedo", "~", "tree"), ("LST", "~", "tree"),
        ("LST", "~", "ET"), ("LST", "~", "Albedo"),
        ("SPEI_m", "~", "tree"),
        ("NDVI_res", "~", "LST"), ("NDVI_res", "~", "SPEI_m"),
        ("NDVI_res", "~", "tree"), ("SIF_res", "~", "LST"),
        ("SIF_res", "~", "SPEI_m"), ("SIF_res", "~", "tree"),
    )
    fields = ("lhs", "op", "rhs", "estimate", "se", "std_all", "std_all_se", "pvalue")
    columns = {field: np.asarray(arrays["sem_" + field]) for field in fields}
    if any(values.ndim != 1 for values in columns.values()):
        raise ValueError("SEM path fields must be one-dimensional")
    if len({len(values) for values in columns.values()}) != 1:
        raise ValueError("SEM path fields have different lengths")
    records = {}
    for i in range(len(columns["lhs"])):
        lhs, op, rhs = (str(columns[field][i]) for field in fields[:3])
        key = (lhs, op, rhs)
        if key in records:
            raise ValueError(f"Duplicate SEM path: {key}")
        if key not in expected:
            raise ValueError(f"Unexpected SEM path for the supplied model: {key}")
        estimate, se, beta, beta_se, pvalue = (float(columns[field][i]) for field in fields[3:])
        if not np.isfinite([estimate, se, beta, beta_se]).all() or se < 0 or beta_se < 0:
            raise ValueError(f"Invalid coefficient or standard error for {key}")
        fixed = bool(op == "=~" and se == 0 and np.isnan(pvalue))
        if not fixed and (not np.isfinite(pvalue) or not 0 <= pvalue <= 1):
            raise ValueError(f"Invalid p-value for {key}")
        source, target = (lhs, rhs) if op == "=~" else (rhs, lhs)
        records[key] = {
            "lhs": lhs, "op": op, "rhs": rhs, "source": source, "target": target,
            "std_all": beta, "std_all_se": beta_se, "estimate": estimate, "se": se,
            "pvalue": None if fixed else pvalue, "fixed_unstandardized": fixed,
        }
    missing = set(expected) - set(records)
    if missing:
        raise ValueError(f"Missing SEM paths: {sorted(missing)}")
    return [records[key] for key in expected]


def write_sem_input_binary(data, binary_path: Path, metadata_path: Path, seed: int, n_boot: int) -> None:
    """Write C-order little-endian float64 rows and exact shape/column metadata."""
    import pandas as pd

    if not isinstance(data, pd.DataFrame):
        raise TypeError("SEM input must be a pandas DataFrame")
    if list(data.columns) != SEM_COLUMNS:
        raise ValueError(f"SEM input columns must be exactly {SEM_COLUMNS}")
    matrix = data.to_numpy(dtype="<f8", copy=True)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("SEM input must be a finite numeric matrix")
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.tofile(binary_path)
    metadata = {
        "format": "little-endian-float64",
        "order": "C-row-major",
        "nrow": int(matrix.shape[0]),
        "ncol": int(matrix.shape[1]),
        "columns": SEM_COLUMNS,
        "seed": int(seed),
        "bootstrap": int(n_boot),
        "seed_status": "new reproducibility-run choice; not the unknown historical seed",
    }
    hasher = hashlib.sha256()
    with binary_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    metadata["input_sha256"] = hasher.hexdigest()
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_sem(
    data,
    output_dir: Path,
    *,
    script_path: Path,
    rscript: Optional[str] = None,
    n_boot: int = SEM_BOOTSTRAP,
    seed: int = FRESH_RUN_SEED,
) -> dict[str, Any]:
    """Fit the historical lavaan model through the binary/JSON R bridge.

    Raises when Rscript is unavailable; it never silently substitutes the
    frozen reference coefficients for a requested fit.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    executable = rscript or os.environ.get("RSCRIPT") or shutil.which("Rscript") or shutil.which("Rscript.exe")
    if not executable:
        raise FileNotFoundError("Rscript is unavailable; set COMPUTE_SEM=False to use frozen reference results")

    input_binary = output_dir / "sem_input.bin"
    input_metadata = output_dir / "sem_input.json"
    fit_rds = output_dir / "sem_fit.rds"
    result_json = output_dir / "sem_result.json"
    write_sem_input_binary(data, input_binary, input_metadata, seed=seed, n_boot=n_boot)
    subprocess.run(
        [
            executable,
            str(Path(script_path).resolve()),
            str(input_binary.resolve()),
            str(input_metadata.resolve()),
            str(fit_rds.resolve()),
            str(result_json.resolve()),
        ],
        check=True,
        cwd=str(Path(script_path).resolve().parents[1]),
    )
    return json.loads(result_json.read_text(encoding="utf-8"))
