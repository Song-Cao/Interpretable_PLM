#!/usr/bin/env python3
"""Retrain the frozen generation-model choices on all curated IRED variants."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from applications.ired_design.training import train_generation_models
from crossfit.utils import REPO_ROOT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = train_generation_models(REPO_ROOT / "Code/config/ired_design.yaml", device=args.device)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
