"""Small platform setup for notebook kernels; no analysis or external writes."""
import os
import sys
from pathlib import Path

_dll_handles = []


def configure_runtime():
    """Keep Windows Conda DLL directories visible for GDAL/numerical libraries."""
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for folder in (Path(sys.prefix) / "Library" / "bin", Path(sys.prefix) / "DLLs"):
            if folder.is_dir():
                _dll_handles.append(os.add_dll_directory(str(folder)))


def dependency_report():
    import importlib.metadata
    import shutil
    report = {}
    for name in ("numpy", "pandas", "scipy", "matplotlib", "statsmodels", "scikit-learn",
                 "seaborn", "GDAL", "basemap", "pingouin", "nbformat", "nbclient"):
        try:
            report[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report[name] = "missing"
    report["Rscript"] = shutil.which("Rscript") or "not on PATH; configure RSCRIPT explicitly for refitting"
    return report
