"""Verify packages, runtime, CUDA and optional model/dataset connectivity."""
from __future__ import annotations

import importlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "environment_check.json"


def _version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return str(getattr(module, "__version__", "installed"))


def main() -> None:
    package_names = ["torch", "transformers", "datasets", "sentence_transformers",
                     "pandas", "numpy", "matplotlib", "yaml", "accelerate"]
    versions = {}
    for package in package_names:
        try:
            versions[package] = _version(package)
        except Exception as exc:
            versions[package] = f"IMPORT_ERROR: {type(exc).__name__}: {exc}"

    import torch
    cuda_available = bool(torch.cuda.is_available())
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_memory_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            if cuda_available else None
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if any(str(v).startswith("IMPORT_ERROR") for v in versions.values()):
        raise SystemExit("One or more required package imports failed; inspect environment_check.json")
    print(f"Environment report saved to {OUT}")


if __name__ == "__main__":
    main()
