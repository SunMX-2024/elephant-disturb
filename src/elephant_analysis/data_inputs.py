"""One data entry point for every notebook; no institutional paths or CSV inputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .artifacts import load_arrays

REFERENCE_FILES = {
    "paired": "reference/paired_reference.npz",
    "density": "reference/density_inputs.npz",
    "sem": "reference/sem_reference.npz",
    "covariate": "covariate_adjustment/covariate_inputs.npz",
    "covariate_reference": "reference/covariate_reference.npz",
    "protected_area_display": "reference/protected_area_display.npz",
}


def read_tiff(path: str | Path, *, return_metadata: bool = False):
    """Read stored TIFF values, preserving NoData until the original analysis masks it.

    No resampling, automatic unit conversion or NoData replacement is performed here.
    Those operations belong to the documented analysis-specific preparation code.
    """
    try:
        from osgeo import gdal
    except ImportError as exc:
        raise ImportError("TIFF input requires GDAL; follow the conda-forge installation instructions in README.md") from exc
    dataset = gdal.Open(str(Path(path)), gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Cannot open raster: {path}")
    values = dataset.ReadAsArray()
    metadata = {"shape": list(values.shape), "transform": list(dataset.GetGeoTransform()),
                "crs": dataset.GetProjection(), "bands": dataset.RasterCount,
                "nodata": [dataset.GetRasterBand(i + 1).GetNoDataValue() for i in range(dataset.RasterCount)]}
    dataset = None
    return (values, metadata) if return_metadata else values


def _resolve_raster(data_root: Path, specification) -> Path:
    relative = specification if isinstance(specification, str) else specification["path"]
    candidates = [data_root / "raw" / relative, data_root / relative]
    if not isinstance(specification, str):
        candidates.extend(data_root / "raw" / alias for alias in specification.get("aliases", []))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing raster {relative}. Put it in {data_root / 'raw'} or update the input manifest.")


def _validate_raster_grid(key: str, meta: dict, expected_grid: dict | None, reference=None):
    """Validate the same grid contract for setup inspection and pixel loading."""
    transform = meta["transform"]
    if meta["bands"] != 1 or len(meta["shape"]) != 2:
        raise ValueError(f"{key}: expected a single-band analysis raster")
    if expected_grid is not None:
        if (list(meta["shape"]) != expected_grid["shape"] or not np.allclose(
                transform, expected_grid["transform"], rtol=0, atol=1e-10)):
            raise ValueError(f"{key}: raster does not match the configured study grid")
    if not np.allclose([transform[1], transform[2], transform[4], transform[5]],
                       [0.01, 0, 0, -0.01], rtol=0, atol=1e-10):
        raise ValueError(f"{key}: expected an aligned, north-up 0.01-degree grid")
    from osgeo import osr
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromWkt(meta["crs"])
    if not spatial_reference.IsGeographic():
        raise ValueError(f"{key}: expected geographic coordinates in degrees")
    signature = (tuple(meta["shape"]), tuple(transform), meta["crs"])
    if reference is not None and signature != reference:
        raise ValueError(f"{key}: raster grid differs; this loader does not silently resample")
    return signature


def describe_raw_inputs(data_root: str | Path, requirements: dict, *, check_grid: bool = False) -> list[dict]:
    """Report required main-figure files; optional checks read headers only.

    Existence and grid checks do not validate variable units, scale factors,
    temporal aggregation or study-specific mask construction.
    """
    from .paired_sites import REQUIRED_RASTERS
    specifications = requirements["required_rasters"]
    if set(specifications) != set(REQUIRED_RASTERS):
        raise ValueError("Raw requirements must contain all and only the main-analysis raster keys")
    root = Path(data_root)
    rows, grid = [], None
    for key, specification in specifications.items():
        row = {"variable": specification["description"], "file": specification["path"],
               "main_figures": "; ".join(specification["main_figures"]),
               "purpose": specification["purpose"], "input_scale": specification["input_scale"],
               "available": False, "grid_check": "not checked"}
        try:
            path = _resolve_raster(root, specification)
        except FileNotFoundError:
            row["grid_check"] = "missing"
        else:
            row["available"] = True
            if check_grid:
                try:
                    from osgeo import gdal
                    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
                    if dataset is None:
                        raise ValueError(f"Cannot open raster: {path.name}")
                    meta = {"shape": [dataset.RasterYSize, dataset.RasterXSize],
                            "bands": dataset.RasterCount, "transform": list(dataset.GetGeoTransform()),
                            "crs": dataset.GetProjection()}
                    dataset = None
                    grid = _validate_raster_grid(key, meta, requirements["grid"], grid)
                    row["grid_check"] = "ok"
                except (ImportError, ValueError, RuntimeError) as exc:
                    row["grid_check"] = str(exc)
        rows.append(row)
    return rows


def load_rasters(data_root: str | Path, specifications: dict, keys=None, *, expected_grid=None) -> dict[str, np.ndarray]:
    """Load the requested aligned rasters and reject inconsistent grids."""
    root = Path(data_root)
    selected = list(keys if keys is not None else specifications)
    # Fail before allocating continental rasters if even one required input is absent.
    paths = {key: _resolve_raster(root, specifications[key]) for key in selected}
    arrays, grid = {}, None
    for key in selected:
        path = paths[key]
        array, meta = read_tiff(path, return_metadata=True)
        grid = _validate_raster_grid(key, meta, expected_grid, grid)
        arrays[key] = array
    return arrays


def load_inputs(name: str, data_root: str | Path, mode: str = "reference", *,
                raster_manifest: str | Path | None = None) -> dict[str, np.ndarray]:
    """Load explicit reference, frozen-prediction, or raw-raster inputs.

    Reference mode never silently substitutes for raw mode. Fig. 2 also needs
    non-raster density observations; these are documented separately.
    """
    root = Path(data_root)
    if mode in ("reference", "precomputed"):
        if name not in REFERENCE_FILES:
            raise ValueError(f"Unknown analysis {name!r}")
        path = root / REFERENCE_FILES[name]
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. See README Data setup and notebook 00.")
        inputs = load_arrays(path)
        if name == "covariate":
            # Reference intervals are explicitly tagged by the estimator module.
            # Loading them is not a claim to have recomputed their bootstrap.
            reference_path = root / REFERENCE_FILES["covariate_reference"]
            inputs.update(load_arrays(reference_path))
        return inputs
    if mode == "raw" and name in ("paired", "sem"):
        from .paired_sites import REQUIRED_RASTERS
        manifest_path = (Path(raster_manifest) if raster_manifest is not None else
                         Path(__file__).resolve().parents[2] / "config" / "raw_data_require.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        specifications = manifest["required_rasters"]
        if set(specifications) != set(REQUIRED_RASTERS):
            raise ValueError("Raster manifest must define all and only the required analysis keys")
        return load_rasters(root, specifications, expected_grid=manifest["grid"])
    if mode == "raw" and name == "density":
        raise ValueError("Density observations cannot be reconstructed from vegetation TIFFs alone; use the documented density source bundle")
    raise ValueError(f"Unsupported input mode {mode!r} for {name!r}")


def describe_inputs(data_root: str | Path) -> list[dict]:
    root = Path(data_root)
    return [{"analysis": name, "file": relative, "available": (root / relative).is_file()}
            for name, relative in REFERENCE_FILES.items()]
