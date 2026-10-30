"""Run the full evaluation suite and print a summary.

Usage: python -m nearzero.eval.all
"""
from __future__ import annotations

import json
import logging
import os

from .compare import RESULTS, main as run_compare


def run() -> dict:
    result = run_compare()
    print("\n=== NEAR-ZERO X evaluation complete ===")
    print(f"artifacts -> {RESULTS}")
    for row in result["comparison"]["rows"]:
        if row.get("scenario") == "*":
            print(
                f"  {row['policy']:<34} recall={_fmt(row['recall'])} "
                f"lead={_fmt(row['mean_lead_time'])}s false={row['false_alarms']} "
                f"success={_fmt(row['intervention_success_rate'])} red={_fmt(row['mean_risk_reduction'])}"
            )
    print(f"detector: {_fmt(result['benchmark']['detector_inference_ms'])} ms "
          f"| core: {_fmt(result['benchmark']['core_pipeline_ms'])} ms")
    return result


def _fmt(v):
    return "NULL" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()