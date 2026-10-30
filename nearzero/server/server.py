"""Flask dashboard server: single-screen industrial monitoring UI."""
from __future__ import annotations

import json
import logging
import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

from ..app.state import snapshot_to_state
from ..config import Config
from ..core.geometry import Homography, save_homography
from ..events.store import EventStore
from ..rules import natural
from ..rules.schema import SafetyRule, save_rules, validate_rule

logger = logging.getLogger("nearzero.server")


def build_app(cfg: Config, runner, static_dir: str) -> Flask:
    app = Flask(__name__, static_folder=static_dir, static_url_path="/static")
    store = getattr(runner, "_store", None) or getattr(runner.pipeline, "_store", None)
    events: EventStore = store
    cache = {"last_state": None}

    @app.route("/")
    def index():
        return send_from_directory(static_dir, "index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "mode": runner.mode})

    @app.get("/api/state")
    def state():
        snap = runner.latest_snapshot
        if snap is None:
            return jsonify({"ready": False})
        st = snapshot_to_state(snap, runner.mode)
        st["ready"] = True
        st["event_summary"] = events.summary() if events else {}
        cache["last_state"] = st
        return jsonify(st)

    @app.get("/api/events")
    def api_events():
        if events is None:
            return jsonify([])
        return jsonify(events.query(limit=int(request.args.get("limit", 50))))

    @app.get("/api/scenarios")
    def api_scenarios():
        from ..sim.scenarios import SCENARIO_NAMES

        return jsonify(list(SCENARIO_NAMES))

    @app.post("/api/scenario")
    def api_set_scenario():
        payload = request.get_json(force=True) or {}
        name = payload.get("scenario", "demo_avoided")
        if isinstance(runner, _sim_class()):
            runner.scenario = name
            runner.reset()
            return jsonify({"ok": True, "scenario": name})
        return jsonify({"ok": False, "error": "not sim mode"}), 400

    @app.get("/api/rules")
    def api_rules():
        return jsonify([r.to_dict() for r in runner.pipeline.rules])

    @app.post("/api/rules")
    def api_add_rule():
        payload = request.get_json(force=True) or {}
        text = payload.get("text", "")
        rule = natural.parse_rule(text) if text else None
        if rule is None:
            candidate = payload.get("rule")
            ok, errors = validate_rule(candidate or {})
            if not ok:
                return jsonify({"ok": False, "error": errors}), 400
            rule = SafetyRule.from_dict(candidate)
        runner.pipeline.set_rules(runner.pipeline.rules + [rule])
        save_rules(runner.pipeline.rules, cfg.rules_path)
        return jsonify({"ok": True, "rule": rule.to_dict()})

    @app.post("/api/calibration")
    def api_calibration():
        payload = request.get_json(force=True) or {}
        try:
            pts = payload["pixel"]
            world = payload["world"]
            h = Homography.from_points(pts, world)
            runner.pipeline.homography = h
            save_homography(h, cfg.perception.homography_json)
            return jsonify({"ok": True})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/intervention")
    def api_intervention():
        payload = request.get_json(force=True) or {}
        name = payload.get("intervention", "NO_ACTION")
        result = _apply_manual(runner, name)
        return jsonify(result)

    @app.get("/video_feed")
    def video_feed():
        def gen():
            while True:
                jpeg = runner.latest_jpeg
                if jpeg:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                else:
                    time.sleep(0.05)
                time.sleep(0.02)

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def _sim_class():
    from .runner import SimRunner

    return SimRunner


def _apply_manual(runner, name: str) -> dict:
    snap = runner.latest_snapshot
    if snap is None or not snap.interventions:
        return {"ok": False, "error": "no interventions available"}
    for row in snap.interventions:
        if row.intervention == name:
            return {"ok": True, "selected": name, "resulting_risk": row.resulting_risk,
                    "risk_reduction": row.risk_reduction}
    return {"ok": False, "error": f"intervention {name} not available"}


import time  # noqa: E402


def run_server(cfg: Config, runner, static_dir: str) -> None:
    import sys

    from flask_cors import CORS

    app = build_app(cfg, runner, static_dir)
    CORS(app)
    import logging as _l

    _l.basicConfig(
        level=_l.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    runner.start()
    host = cfg.server.host
    port = cfg.server.port
    print(f"\nNEAR-ZERO X dashboard -> http://localhost:{port}  (mode={runner.mode})")
    print("Stop with Ctrl+C\n")
    try:
        app.run(host=host, port=port, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        runner.stop()
    finally:
        runner.stop()
        if getattr(runner, "_store", None):
            runner._store.close()