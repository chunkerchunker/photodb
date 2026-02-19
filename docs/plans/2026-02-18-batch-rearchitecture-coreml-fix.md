# Batch Rearchitecture: CoreML SIGSEGV Fix Plan

## Context

Branch `feature/batch-rearchitecture` upgraded ultralytics from 8.1.0 to >=8.3.206
per [ultralytics/ultralytics#22300](https://github.com/ultralytics/ultralytics/pull/22300)
to enable dynamic batch CoreML inference. **Do NOT revert ultralytics version.**

## Problem

SIGSEGV (exit 139) when running the full pipeline. Root cause: **ultralytics 8.3.206+
CoreML model loading corrupts process state, causing any subsequent MPS or PyTorch
model loading/inference to segfault.**

### Findings

1. **Main branch (ultralytics 8.1.0)**: CoreML YOLO + MobileCLIP MPS coexist fine
2. **This branch (ultralytics 8.4.14)**: CoreML YOLO loaded → any subsequent `torch.load`
   or MPS model operation → SIGSEGV
3. Init order matters: loading CoreML YOLO **last** avoids SIGSEGV during init
4. But even with correct init order, **runtime inference still crashes** when CoreML
   and MPS models run in the same process
5. Each stage works fine in isolation (detection alone, age_gender alone, scene_analysis alone)
6. `dynamic=True` CoreML export also causes SIGSEGV independently (coremltools 9.0 + PyTorch 2.8)
7. Non-dynamic CoreML export (nms=False) works for single-image inference in isolation

### Platform

- coremltools 9.0, PyTorch 2.8.0, ultralytics 8.4.14, Apple M1 Max
- coremltools warns: "Torch version 2.8.0 has not been tested with coremltools"

## Resolution: PyTorch MPS Batch (Option D+)

Instead of fighting CoreML, we skip it entirely and use **PyTorch YOLO on MPS** with
batch inference. This avoids the CoreML SIGSEGV and lets MobileCLIP also use MPS.

### Benchmark (83 photos, detection + scene_analysis, parallel 10)

| Configuration | Wall Time | Detection avg | Scene Analysis avg |
|---|---|---|---|
| CoreML YOLO + CPU MobileCLIP (Option A) | **2m39s** | 0.42s/photo | 18.11s/photo |
| **MPS YOLO batch + MPS MobileCLIP** | **0m33s** | 1.31s/photo | 2.59s/photo |

**4.8x faster overall.** Scene analysis 7x faster (MPS vs CPU MobileCLIP).

### Throughput benchmarks (isolated)

| Mode | Throughput |
|---|---|
| CoreML single-image (ANE) | 24.3 img/s |
| PyTorch MPS single | 8.0 img/s |
| PyTorch MPS batch(8) | 16.2 img/s |
| PyTorch MPS batch(16) | 17.9 img/s |

CoreML is faster per-image, but detection was never the bottleneck (0.42s avg).
The real win is MobileCLIP on MPS instead of CPU.

### Config defaults

```
DETECTION_PREFER_COREML = False   # PyTorch MPS (enables YOLO batch + MPS MobileCLIP)
YOLO_BATCH_ENABLED = True         # YOLO BatchCoordinator active
```

CoreML can still be enabled via `DETECTION_PREFER_COREML=true` env var (falls back to
Option A: MobileCLIP on CPU, YOLO unbatched).

## Previous Options (for reference)

### Option A: Force MobileCLIP to CPU when CoreML YOLO is active
- Still works as fallback (`DETECTION_PREFER_COREML=true`)
- Verified: 83/83 photos, 2m39s, no SIGSEGV
- Slower than MPS batch due to CPU MobileCLIP

### Option B: Run CoreML YOLO in a subprocess
- Not needed — MPS batch is simpler and faster

### Option C: Pin ultralytics to exact 8.3.206
- Not pursued — MPS batch avoids the CoreML issue entirely

## Current State

- [x] BatchCoordinator infrastructure complete (14 tests passing)
- [x] YOLO batch coordinator wired into detection stage
- [x] MobileCLIP batch coordinators wired into scene_analysis
- [x] YOLO_BATCH_ENABLED=True (PyTorch MPS), INSIGHTFACE_BATCH_ENABLED=False
- [x] Init order preserved: age_gender → scene_analysis → detection (safe for CoreML fallback)
- [x] download_models.sh: non-dynamic export (for CoreML fallback)
- [x] CoreML + MPS SIGSEGV resolved via MPS-only path
- [x] Full pipeline e2e test: 83/83 photos, 0m33s, all 3 batch coordinators working
- [x] Unit tests pass (234 passed, 4 skipped)
- [x] Committed to branch
