# REPO AUDIT

Date: 2026-10-02
Auditor: NEAR-ZERO X dev team

## Current State

| Item            | Finding                                                        |
|-----------------|----------------------------------------------------------------|
| Git repository  | No                                                            |
| Source code     | None (empty project root)                                     |
| Python project  | No, greenfield build                                          |
| Node project    | No                                                           |
| Existing deps   | Global Python 3.13.15, pip 26.2.1, numpy 2.5.3               |
| Model files     | None                                                         |
| Data            | None                                                         |
| Tests           | None                                                         |
| TODO/incomplete | None (no code)                                               |

## Hardware / Runtime Environment

- OS: Windows 10/11 (win32), PowerShell available
- Python: 3.13.15
- GPU: NVIDIA GeForce RTX 5070 Laptop GPU, 8 GB VRAM (Blackwell, sm_120)
- CUDA driver: 13.1
- Internet: assumed available for package install

### GPU constraint (important)

RTX 5070 is **Blackwell (compute capability 12.0 / sm_120)**. Official stable PyTorch
wheels for sm_120 only exist starting at torch 2.7 (CUDA 12.8 build). `torch.cuda.is_available()`
on older wheels returns False on this GPU. Decision: install the newest stable torch with CUDA
support and fall back to CPU automatically when CUDA is unavailable.

## Reusable assets

None in-repo. Project starts from scratch.

## Reuse strategy from external sources

- Detection + tracking: reuse pretrained YOLO weights via `ultralytics` (with ByteTrack),
  downloaded once at setup time. If download/GPU fails, a deterministic **simulated perception**
  path keeps the whole P0 stack testable offline.
- The safety intelligence (prediction, risk, counterfactual planner, verification) is written
  from scratch. It must not depend on the vision stack so it can be unit-tested and batch-evaluated
  with the synthetic simulator.

## Technical debt to avoid

- No camera/core logic entanglement: risk engine must be vision-agnostic.
- No magic metric "risk score" without documented formula -> `docs/RISK_MODEL.md`.
- No undocumented performance numbers -> only measured values, else `NOT MEASURED`.

## Implementation strategy

Priority ladder from the brief: **Working Demo > Measured Evidence > Robust Architecture > Advanced Features**

1. P0 core (geometry, kalman, prediction, risk, planner, verification, events) as pure Python.
2. Synthetic 2D scenario engine (offline tests + guaranteed demo input).
3. Vision pipeline (YOLO + ByteTrack) with CPU/GPU and webcam/file inputs.
4. Flask dashboard (MJPEG + JSON state) — single screen demo.
5. Baseline & ablation evaluation, tests, docs, demo scripts.
6. P1 features where stable: NL rules, privacy blur, restricted zones, uncertainty band.