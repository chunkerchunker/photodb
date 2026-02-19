# Batching Rearchitecture Plan

**Date:** 2026-02-18
**Status:** Draft

## Problem Statement

The current pipeline processes photos one-at-a-time per thread. Each thread runs a photo through all stages sequentially (normalize → metadata → detection → age_gender → scene_analysis). While `ThreadPoolExecutor` provides parallelism at the photo level, ML inference calls within each stage process only a single image.

Modern ML accelerators (ANE, MPS, CUDA) achieve much higher throughput with batched inference. The ANE has 16 cores with a hardware scheduler and pipeline architecture — but our current code submits single-image requests, leaving throughput on the table.

## Current Architecture

### Processing Flow

```
ThreadPoolExecutor (N workers)
  └─ per-photo thread
       ├─ normalize (IO: copy + resize)
       ├─ metadata (IO: EXIF parsing)
       ├─ detection
       │    ├─ YOLO person_face (single image → CoreML/ANE)
       │    └─ InsightFace embedding (per-face crop → ONNX/CoreML)
       ├─ age_gender
       │    └─ MiVOLO recognize (single image, serialized with lock)
       └─ scene_analysis
            ├─ Apple Vision classify (single image → ANE)
            ├─ MobileCLIP encode_image (single image → MPS)
            └─ MobileCLIP encode_faces_batch (N faces batched → MPS)
```

### Key Observations

- **Shared models**: ML models are loaded once in `__init__` and shared across threads via `__new__` pattern (no per-thread copies).
- **MobileCLIP already batches faces**: `encode_faces_batch()` stacks face crops into a batch tensor and runs `model.encode_image(batch)` in one call — but only within a single photo.
- **MiVOLO is locked**: All threads serialize through a `threading.Lock`. Batching MiVOLO calls would not help since they're already serialized.
- **Database per-thread**: Each thread gets its own `PhotoRepository` via `ConnectionPool` for thread-safe DB access.

## Model Batch Support Matrix

| Model | Framework | Batch API | Currently Batched | Batching Feasible | Notes |
|-------|-----------|-----------|-------------------|-------------------|-------|
| **YOLO person_face** | Ultralytics + CoreML | `model([img1, img2, ...])` | No (single image) | **Yes** | Fixed in Ultralytics 8.3.206 ([PR #22300](https://github.com/ultralytics/ultralytics/pull/22300)). Requires re-exporting CoreML model with `dynamic=True nms=False`. We're on 8.1.0 — upgrade required. |
| **InsightFace ArcFace** | ONNX Runtime + CoreML EP | `get_feat([face1, face2, ...])` | No (single face) | **Risky** | `get_feat()` accepts list of aligned faces. Uses `cv2.dnn.blobFromImages` internally. However, ONNX Runtime CoreML EP has SIGSEGV with dynamic batch dimensions ([onnxruntime#21227](https://github.com/microsoft/onnxruntime/issues/21227)). May work with fixed batch sizes. |
| **MiVOLO** | PyTorch | Single-image API only | No | **No** | API takes single image + bboxes. Model weights support batch internally but `recognize()` doesn't expose it. Additionally serialized with `threading.Lock` due to race conditions. Not worth the effort. |
| **MobileCLIP (images)** | PyTorch + MPS | `model.encode_image(batch_tensor)` | No (single image) | **Yes** | Pure PyTorch, already works with batch tensors. `encode_faces_batch` proves it. Just need to batch across photos instead of only within a photo. |
| **MobileCLIP (faces)** | PyTorch + MPS | Same as above | Yes (within photo) | **Yes, cross-photo** | Already batches faces within a single photo. Could batch across multiple photos for larger batch sizes. |
| **Apple Vision** | Vision.framework | No batch API | No | **No** | `VNClassifyImageRequest` is per-image. No batch API exists. Each request creates its own `VNImageRequestHandler`. |

## Recommended Approach: Batch Coordinator Pattern

### Architecture

Instead of changing the per-photo threading model, introduce a **batch coordinator** for stages that benefit from batching. The coordinator collects inference requests from multiple worker threads, batches them, runs a single batched inference, and distributes results back.

```
ThreadPoolExecutor (N workers)
  └─ per-photo thread
       ├─ normalize (unchanged)
       ├─ metadata (unchanged)
       ├─ detection
       │    ├─ YOLO → BatchCoordinator → batched detect (CoreML dynamic)
       │    └─ InsightFace (unchanged — risky with CoreML EP)
       ├─ age_gender (unchanged — locked, single API)
       └─ scene_analysis
            ├─ Apple Vision (unchanged — no batch API)
            ├─ MobileCLIP image → BatchCoordinator → batched encode_image
            └─ MobileCLIP faces → BatchCoordinator → batched encode_faces
```

### BatchCoordinator Design

```python
class BatchCoordinator:
    """Collects inference requests from worker threads and runs batched inference."""

    def __init__(self, inference_fn, max_batch_size=32, max_wait_ms=50):
        self._inference_fn = inference_fn  # e.g., model.encode_image
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._batch_loop, daemon=True)
        self._thread.start()

    def submit(self, input_tensor) -> concurrent.futures.Future:
        """Submit a single input and get a Future for the result."""
        future = concurrent.futures.Future()
        self._queue.put((input_tensor, future))
        return future

    def _batch_loop(self):
        """Background thread: collect inputs, batch, infer, distribute."""
        while True:
            batch_items = []
            # Block on first item
            item = self._queue.get()
            if item is None:
                break
            batch_items.append(item)

            # Collect more items up to batch size or timeout
            deadline = time.monotonic() + self._max_wait_ms / 1000
            while len(batch_items) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                    if item is None:
                        break
                    batch_items.append(item)
                except queue.Empty:
                    break

            # Run batched inference
            inputs = torch.cat([item[0] for item in batch_items], dim=0)
            try:
                results = self._inference_fn(inputs)
                # Distribute results
                for i, (_, future) in enumerate(batch_items):
                    future.set_result(results[i:i+1])
            except Exception as e:
                for _, future in batch_items:
                    future.set_exception(e)
```

### How Worker Threads Use It

```python
# In scene_analysis stage, instead of:
image_embedding = self.analyzer.encode_image(str(normalized_path))

# Worker thread does:
preprocessed = self.analyzer.preprocess(str(normalized_path))
future = self.batch_coordinator.submit(preprocessed)
image_embedding = future.result()  # Blocks until batch is processed
```

## Implementation Phases

### Phase 1: YOLO Detection Batching (Highest Impact)

**Scope:** Batch YOLO person_face detection across photos via CoreML dynamic batch.

**Why first:** Detection is typically the most expensive stage per photo. YOLO CoreML batch inference is now supported (Ultralytics 8.3.206+) and runs on the ANE, which has 16 cores and a hardware scheduler designed for throughput.

**Prerequisites:**
1. Upgrade `ultralytics` from 8.1.0 to >= 8.3.206
2. Re-export CoreML model with dynamic batch: `yolo export model=yolov8x_person_face.pt format=coreml dynamic=True nms=False`
3. Update `scripts/download_models.sh` to export with new flags

**Changes:**
1. Create `BatchCoordinator` utility class (shared across all phases)
2. Add `BatchCoordinator` instance to `LocalProcessor.__init__` for YOLO
3. Add `preprocess()` method to `PersonDetector` that loads and preprocesses an image without running inference
4. Modify `PersonDetector.detect()` to support batch mode (accept list of images, return list of results)
5. Modify `DetectionStage.process_photo()` to use coordinator when available
6. Handle NMS post-processing per-image after batched inference (since `nms=False` in export)
7. Pass coordinator reference when creating pooled detection stages

**Expected improvement:** Significant throughput improvement for detection. ANE can pipeline multiple images through its 16 cores. Exact improvement depends on batch size and image dimensions.

**Risks:**
- NMS must be handled in post-processing since `dynamic=True` requires `nms=False`
- Dynamic image sizes in a batch may require padding/resizing to uniform dimensions
- Need to validate detection accuracy matches single-image results

### Phase 2: MobileCLIP Image Batching

**Scope:** Batch `encode_image()` calls across photos in scene_analysis stage.

**Why:** Every photo goes through MobileCLIP image encoding. With N worker threads submitting to the coordinator, we get batch sizes up to N (typically 10-500). MPS/Metal can efficiently process batches of 16-64 images.

**Changes:**
1. Add `preprocess()` method to `MobileCLIPAnalyzer` that returns the preprocessed tensor without running inference
2. Add `BatchCoordinator` instance to `LocalProcessor.__init__` for MobileCLIP
3. Modify `SceneAnalysisStage.process_photo()` to use coordinator when available
4. Pass coordinator reference when creating pooled scene analysis stages

**Expected improvement:** 3-5x throughput improvement for scene_analysis image encoding, depending on batch size and MPS utilization.

### Phase 3: MobileCLIP Cross-Photo Face Batching

**Scope:** Batch face crops across photos, not just within a single photo.

**Why:** Currently `encode_faces_batch()` batches faces within one photo (typically 1-5 faces). Batching across photos yields much larger batches (e.g., 10 photos × 3 faces = 30 face crops).

**Changes:**
1. Add face-specific `BatchCoordinator` instance
2. Modify `_process_face_tags()` to submit individual face crops to the coordinator
3. Collect futures and match results back to detections

**Expected improvement:** 2-3x for face tagging when photos have few faces (most common case). Less improvement for group photos that already batch well.

### Phase 4: InsightFace Embedding Batching (Experimental)

**Scope:** Batch `get_feat()` calls across photos in detection stage.

**Why experimental:** ONNX Runtime CoreML EP has known issues with dynamic batch dimensions. May crash.

**Changes:**
1. Test with fixed batch sizes (e.g., always pad to 8 or 16 faces)
2. If stable, add `BatchCoordinator` for InsightFace embeddings
3. Fall back to single-face if batch inference fails

**Risk mitigation:** Feature-flagged, disabled by default, with automatic fallback.

### Phase 5: Pipeline Overlap (Future)

**Scope:** Overlap IO-bound stages with ML-bound stages.

**Why:** Normalize and metadata are IO-bound. Detection/scene_analysis are ML-bound. Currently they run sequentially per photo. With batching, we could start normalizing the next batch while the current batch is in ML inference.

**This is a larger rearchitecture** and depends on phases 1-4 being stable first.

## What NOT to Batch

| Stage/Model | Reason |
|-------------|--------|
| **Normalize** | IO-bound (file copy + ImageMagick resize). Already parallelized well by threads. |
| **Metadata** | IO-bound (EXIF parsing). Already parallelized well by threads. |
| **MiVOLO** | Serialized with lock due to thread-safety issues. Single-image API. The lock means batching would be pointless — only one thread accesses the model at a time anyway. |
| **Apple Vision** | No batch API in Vision.framework. Each `VNClassifyImageRequest` must be created per-image with its own handler. |

## Configuration

New environment variables:
- `BATCH_COORDINATOR_MAX_SIZE`: Maximum batch size (default: `32`)
- `BATCH_COORDINATOR_MAX_WAIT_MS`: Maximum time to wait for batch to fill (default: `50`)
- `BATCH_COORDINATOR_ENABLED`: Enable/disable batch coordinator (default: `true`)
- `YOLO_BATCH_ENABLED`: Enable YOLO batch inference (default: `true` — requires dynamic CoreML model)
- `INSIGHTFACE_BATCH_ENABLED`: Enable experimental InsightFace batching (default: `false`)

## Performance Measurement

Existing performance tracking (added in recent commits) will automatically capture improvements:
- Per-stage avg/photo timing
- Per-face avg/face timing for detection-related stages
- Overall wall time

The batch coordinator should also log:
- Average batch size achieved
- Batch queue wait time (how long items wait before batched)
- Batch inference time vs sum of individual inference times

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| YOLO dynamic CoreML NMS handling | Medium | High | Ultralytics handles NMS in post-processing when `nms=False` in export; validate detection accuracy matches single-image |
| YOLO batch with variable image sizes | Medium | Medium | Ultralytics AutoBackend adapts inputs for dynamic CoreML models; test with diverse image dimensions |
| MPS batch size too large → OOM | Low | High | Cap batch size at 32, monitor memory |
| Batch wait time adds latency for small runs | Medium | Low | Short timeout (50ms), bypass coordinator for `parallel=1` |
| Thread coordination overhead | Low | Low | Lock-free queue, minimal synchronization |
| InsightFace CoreML crash with batch | High | Medium | Feature-flagged, disabled by default, auto-fallback |
| MobileCLIP results differ between batched and unbatched | Very Low | Medium | Validate numerically — PyTorch batch inference is deterministic |
| Ultralytics upgrade breaks existing code | Low | Medium | Pin to specific version, test all stages before upgrading |

## Decision Log

- **2026-02-18:** Initially decided against YOLO batching due to CoreML crash bug ([#10136](https://github.com/ultralytics/ultralytics/issues/10136))
- **2026-02-18:** Revised — YOLO CoreML batch fix landed in Ultralytics 8.3.206 ([PR #22300](https://github.com/ultralytics/ultralytics/pull/22300)). Requires `dynamic=True nms=False` export. Promoted YOLO batching to Phase 1.
- **2026-02-18:** Decided against MiVOLO batching due to serialized lock + single-image API
- **2026-02-18:** Decided against Apple Vision batching due to no batch API
- **2026-02-18:** Chose batch coordinator pattern over redesigning the per-photo threading model to minimize disruption
