#!/usr/bin/env python3
"""Retrain the frozen generation-model choices on all curated IRED variants."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from applications.ired_design.training import train_generation_models
from applications.ired_design.training import CLASSICAL_ARMS, TORCH_ARMS
from crossfit.utils import REPO_ROOT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--arm", choices=(*CLASSICAL_ARMS, "esm_mean_random_forest", *TORCH_ARMS),
        action="append", default=None,
    )
    args = parser.parse_args()
    result = train_generation_models(
        REPO_ROOT / "Code/config/ired_design.yaml", device=args.device,
        only_arms=None if args.arm is None else set(args.arm),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
