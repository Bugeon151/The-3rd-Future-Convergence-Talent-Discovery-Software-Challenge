"""NEAR-ZERO X demo launcher.

Usage:
    python -m nearzero.demo                      # simulated demo (default)
    python -m nearzero.demo --mode webcam        # live webcam
    python -m nearzero.demo --mode video --video clip.mp4
    python -m nearzero.demo --scenario head_on --headless --steps 60
"""
from __future__ import annotations

import argparse
import logging
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="NEAR-ZERO X demo")
    ap.add_argument("--mode", choices=["sim", "webcam", "video"], default="sim")
    ap.add_argument("--scenario", default="demo_avoided")
    ap.add_argument("--video", default=None)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--scenario-file", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from ..config import load_config
    from ..events.store import EventStore

    cfg = load_config(args.scenario_file)
    cfg.server.port = args.port

    if os.path.isfile(cfg.events.db_path):
        os.remove(cfg.events.db_path)
    os.makedirs(os.path.dirname(cfg.events.db_path), exist_ok=True)
    store = EventStore(cfg.events.db_path)

    if args.mode == "sim":
        from ..server.runner import SimRunner

        runner = SimRunner(cfg, args.scenario, store=store)
    elif args.mode == "webcam":
        from ..server.runner import CameraRunner

        runner = CameraRunner(cfg, 0, store=store)
    else:
        if not args.video:
            sys.exit("--video path required with --mode video")
        from ..server.runner import CameraRunner

        runner = CameraRunner(cfg, args.video, loop=True, store=store)

    runner.pipeline._store = store

    if args.headless:
        from ..server.runner import SimRunner as _SR

        if not isinstance(runner, _SR):
            sys.exit("--headless requires --mode sim")
        peak = None
        history = []
        for k in range(args.steps):
            if not runner.tick_for(k * cfg.prediction.dt_s):
                break
            snap = runner.latest_snapshot
            if peak is None or (snap and snap.risk.score > peak.risk.score):
                peak = snap
            if snap is not None:
                history.append((snap.t, snap.risk.score, snap.risk.collision_probability))
        if peak is None:
            sys.exit("no frames produced")
        print("\n=== NEAR-ZERO X headless run ===")
        print(f"steps={args.steps} scenario={runner.scenario}")
        print("risk timeline (t, risk, collision_prob):")
        for k in range(0, len(history), max(1, len(history) // 12)):
            t, r, cp = history[k]
            bar = "#" * int(r / 5)
            print(f"   t={t:4.1f}  risk={r:5.0f}  col={cp*100:3.0f}%  {bar}")
        print(f"\npeak risk snapshot (t={peak.t:.1f})")
        print(f"  state={peak.risk.state}  risk={peak.risk.score:.0f}/100")
        print(f"  collision_probability={peak.risk.collision_probability*100:.0f}%  "
              f"ttc={peak.risk.ttc if peak.risk.ttc is not None else '-'}s "
              f"pmd={peak.risk.predicted_min_distance if peak.risk.predicted_min_distance is not None else '-'}m")
        print(f"  tracks={len(peak.tracks)}  events={store.summary()}")
        print("  counterfactual table:")
        for row in sorted(peak.interventions, key=lambda r: r.resulting_risk):
            mark = " <-- BEST" if peak.best_intervention and row.intervention == peak.best_intervention.intervention else ""
            print(f"   {row.intervention:<16} risk={row.resulting_risk:5.0f}"
                  f"   reduction={row.risk_reduction:5.0f}{mark}")
        if runner._executed:
            print(f"  executed maneuver: {runner._executed} at t={runner._executed_at * cfg.prediction.dt_s:.1f}s")
        if peak.verify and peak.verify.get("last_event"):
            le = peak.verify["last_event"]
            print(f"  verification: {le.get('state')} on event #{le.get('event_id')} "
                  f"(recommended={le.get('recommended') or '-'}, peak_risk={le.get('peak_risk', 0):.0f})")
        store.close()
        return

    from ..server.server import run_server

    run_server(cfg, runner, _static_dir())


def _static_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    return os.path.join(root, "frontend", "static")


if __name__ == "__main__":
    main()