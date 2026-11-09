"""Portable, pickle-free NumPy artifacts and their provenance records."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def save_arrays(path: str | Path, **arrays) -> Path:
    """Atomically save named arrays. Object arrays/pickle are deliberately excluded."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Array bundles must have an .npz extension")
    converted = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"{name}: use numerical/string arrays, not Python objects")
        converted[name] = array
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.stem + "_", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **converted)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Read an NPZ without permitting serialized executable Python objects."""
    with np.load(Path(path), allow_pickle=False) as bundle:
        return {key: bundle[key] for key in bundle.files}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: str | Path, value: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return path


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def record_run(path: str | Path, *, analysis: str, inputs: list[Path], parameters: dict,
               mode: str, outputs: list[Path], notes: list[str] | None = None) -> Path:
    """Record file fingerprints and settings without embedding private absolute paths."""
    import importlib.metadata
    import platform
    versions = {}
    for name in ("numpy", "pandas", "scipy", "matplotlib", "statsmodels", "scikit-learn", "pingouin"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return save_json(path, {"analysis": analysis, "mode": mode, "parameters": parameters,
        "python": platform.python_version(), "packages": versions,
        "inputs": [{"name": p.name, "sha256": sha256_file(p)} for p in inputs],
        "outputs": [{"name": p.name, "sha256": sha256_file(p)} for p in outputs],
        "notes": notes or []})
