#!/usr/bin/env python3
"""Stage-1 benchmark: persistence vs climatology vs ensemble on archive holdout."""

from __future__ import annotations

import json
from pathlib import Path

from quantum_earth.orchestration.loop import OperatingLoop
from quantum_earth.storage import StorageLayout


def main() -> None:
    loop = OperatingLoop()
    locations = ["Birmingham", "London", "Manchester"]
    results = {}
    for loc in locations:
        results[loc] = loop.verify_recent(loc, variable="temperature_2m", hours=48)
    out = StorageLayout().path("METRICS", "benchmark_stage1.json")
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v.get("scores", {}).get("ensemble_mean", {}) for k, v in results.items()}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
