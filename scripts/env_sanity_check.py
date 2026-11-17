"""Environment and hardware sanity check for the Colab T4 workflow."""
from __future__ import annotations
import json
import platform
import sys
from pathlib import Path

import torch
import transformers
import datasets
import sentence_transformers

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "environment_check.json"


def main() -> None:
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            if torch.cuda.is_available() else None
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
