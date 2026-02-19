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

## Fix Strategy

The CoreML + MPS coexistence issue must be resolved. Options (in preference order):

### Option A: Force MobileCLIP to CPU when CoreML YOLO is active **[IMPLEMENTED]**
- MobileCLIP encoding is fast even on CPU
- CoreML YOLO stays on Neural Engine (the bottleneck)
- Avoids MPS entirely, sidestepping the conflict
- **Verified working**: 83/83 photos, detection + scene_analysis, 0 failures, no SIGSEGV

### Option B: Run CoreML YOLO in a subprocess
- Isolate CoreML from MPS completely via process boundary
- More complex but guaranteed isolation
- BatchCoordinator would submit to subprocess instead of thread

### Option C: Pin ultralytics to exact 8.3.206
- The PR version, not the latest (currently using 8.4.14)
- May have different CoreML behavior
- Test before committing to this

### Option D: Disable CoreML, use PyTorch YOLO on MPS
- Both models use MPS (may still conflict)
- Loses Neural Engine performance advantage

## Current State

- [x] BatchCoordinator infrastructure complete (14 tests passing)
- [x] YOLO batch coordinator wired into detection stage
- [x] MobileCLIP batch coordinators wired into scene_analysis
- [x] Config flags: YOLO_BATCH_ENABLED=False (dynamic CoreML broken), INSIGHTFACE_BATCH_ENABLED=False
- [x] Init order fixed: age_gender → scene_analysis → detection (CoreML last)
- [x] download_models.sh: non-dynamic export (dynamic=True causes separate SIGSEGV)
- [x] **RESOLVED: CoreML + MPS runtime SIGSEGV** — Option A: MobileCLIP forced to CPU
- [x] Full pipeline end-to-end test (83/83 photos, detection + scene_analysis, 2m39s)
- [x] Unit tests pass (234 passed, 4 skipped)
- [ ] Commit changes to branch
