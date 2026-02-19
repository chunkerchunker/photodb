# Remove InsightFace Dependency: Direct ONNX ArcFace Loading

## Goal

Replace the `insightface` Python package with direct ONNX Runtime inference for ArcFace embeddings. This eliminates insightface (and its transitive opencv-python dependency) from our direct imports. The same `w600k_r50.onnx` model file is used — only the loading and preprocessing wrapper changes.

**Not in scope**: Removing opencv-python entirely. MiVOLO and ultralytics still depend on it transitively. This plan removes our *direct* usage of both `insightface` and `cv2`.

## Current State

InsightFace is used in exactly **2 source files** for one purpose: extracting 512-dim ArcFace face embeddings for clustering.

### Call chain

```
DetectionStage.process_photo()
  → PersonDetector.extract_embedding(pil_image, bbox)
    → EmbeddingExtractor.extract(pil_image, bbox)
      → crop + pad bbox from PIL image
      → RGB→BGR via numpy slice
      → cv2.resize to (112, 112)
      → model.get_feat([bgr_array])          ← insightface ArcFaceONNX
        → cv2.dnn.blobFromImages(...)        ← insightface internal cv2 usage
        → session.run(...)                   ← onnxruntime (this is all we need)
      → returns 512-dim float list
```

### What insightface actually does

`ArcFaceONNX.get_feat()` (`arcface_onnx.py:77-85`):

```python
def get_feat(self, imgs):
    blob = cv2.dnn.blobFromImages(
        imgs, 1.0 / self.input_std, input_size,
        (self.input_mean, self.input_mean, self.input_mean), swapRB=True
    )
    net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
    return net_out
```

For the buffalo_l model (`w600k_r50.onnx`), it's an MXNet-trained ArcFace model with `input_mean=0.0`, `input_std=1.0`. So `blobFromImages` does:

1. Resize to (112, 112) — already done by our code before calling `get_feat`
2. Subtract mean (0.0) — no-op
3. Scale by 1/std (1.0) — no-op
4. `swapRB=True` — BGR→RGB channel swap
5. HWC→NCHW transpose
6. Convert to float32

**That's just: BGR→RGB swap + HWC→NCHW transpose + float32 cast.** All trivially done with numpy.

There's also a `forward()` method that takes a pre-built NCHW blob and just does `(blob - mean) / std` + `session.run()`. With mean=0, std=1, this is even simpler.

### What we use from insightface

| Import | Purpose | Replacement |
|---|---|---|
| `insightface.model_zoo.get_model` | Load ONNX model with providers | `onnxruntime.InferenceSession` directly |
| `insightface.utils.storage.ensure_available` | Auto-download model pack | Manual download check + error message |
| `model.prepare(ctx_id=0)` | Set execution provider | Pass providers to `InferenceSession` constructor |
| `model.get_feat([face_img])` | Preprocess + inference | Numpy preprocessing + `session.run()` |

## Implementation Plan

### Task 1: Rewrite EmbeddingExtractor

**File:** `src/photodb/utils/embedding_extractor.py`

Replace the entire class internals. Remove `import cv2` and `from insightface.model_zoo import get_model`.

**`__init__`:**

- Find the ONNX model file (same `_ensure_model_available` logic, minus insightface download fallback)
- Create `onnxruntime.InferenceSession(model_path, providers=providers)`
- Read input/output names and shapes from session
- Determine `input_mean` and `input_std` by inspecting ONNX graph (same logic as `ArcFaceONNX.__init__`: check first 8 nodes for Sub/Mul ops)

**`_preprocess(bgr_array)` → `np.ndarray` (new method):**

```python
# Input: (H, W, 3) uint8 BGR numpy array, already 112x112
# 1. BGR→RGB channel swap
rgb = bgr_array[:, :, ::-1]
# 2. HWC→CHW transpose
chw = rgb.transpose(2, 0, 1)
# 3. Float32 + normalize
blob = ((chw.astype(np.float32) - self.input_mean) / self.input_std)
# 4. Add batch dim → (1, 3, 112, 112)
return blob[np.newaxis]
```

**`get_feat(imgs)` → `np.ndarray` (replaces insightface call):**

```python
blobs = np.concatenate([self._preprocess(img) for img in imgs], axis=0)
return self.session.run(self.output_names, {self.input_name: blobs})[0]
```

**`extract(image, bbox)`:**

- Same crop + pad logic as today
- Replace `cv2.resize(bgr_array, INPUT_SIZE)` with `crop.resize(INPUT_SIZE, Image.LANCZOS)` + numpy BGR conversion
- Call `self.get_feat([face_img])` instead of `self.model.get_feat([face_img])`

**`extract_from_aligned(aligned_face)`:**

- Replace `cv2.resize` with Pillow resize (convert numpy→PIL→resize→numpy if needed, or use numpy-based resize)
- Since this method receives a BGR numpy array and the only caller is our own code, we can change the interface to accept a PIL image if that simplifies things. Check callers first.

### Task 2: Update PersonDetector warmup

**File:** `src/photodb/utils/person_detector.py`

Line 159 calls `self.embedding_extractor.model.get_feat([dummy_face])` directly on the insightface model object. Change to use the new `EmbeddingExtractor.get_feat()` method:

```python
self.embedding_extractor.get_feat([dummy_face])
```

Or add a `warmup()` method to `EmbeddingExtractor` that encapsulates this.

### Task 3: Update model download script

**File:** `scripts/download_models.sh`

The current script downloads buffalo_l by running:

```bash
python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l').prepare(ctx_id=0)"
```

Replace with a direct download of `w600k_r50.onnx`. The buffalo_l model pack is hosted at:

- `https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip`

New script logic:

```bash
ONNX_MODEL="$HOME/.insightface/models/buffalo_l/w600k_r50.onnx"
if [ ! -f "$ONNX_MODEL" ]; then
    mkdir -p "$(dirname "$ONNX_MODEL")"
    curl -L "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" -o /tmp/buffalo_l.zip
    unzip -o /tmp/buffalo_l.zip -d "$HOME/.insightface/models/buffalo_l/"
    rm /tmp/buffalo_l.zip
fi
```

### Task 4: Remove `_ensure_model_available` download fallback

**File:** `src/photodb/utils/embedding_extractor.py`

The current code falls back to `from insightface.utils.storage import ensure_available` to auto-download the model. Replace with a clear error message pointing to the download script:

```python
raise FileNotFoundError(
    f"ArcFace model not found for {model_name}. "
    f"Run: ./scripts/download_models.sh"
)
```

### Task 5: Remove insightface from pyproject.toml

**File:** `pyproject.toml`

Remove `"insightface>=0.7.3"` from `dependencies`. Keep `"onnxruntime>=1.16.0"` (already listed separately).

Run `uv sync` to verify the dependency tree resolves. Note: insightface may still be pulled in transitively by mivolo — check and document if so.

### Task 6: Update tests

**File:** `tests/test_embedding_extractor.py`

- Update mocks: replace `@patch("...get_model")` with `@patch("onnxruntime.InferenceSession")`
- Mock `session.run()` return value instead of `model.get_feat()`
- Update `test_extract_converts_rgb_to_bgr` and `test_extract_adds_padding_to_bbox` for new preprocessing path
- Remove any references to `model.prepare()`

**File:** `tests/test_person_detector.py`

- Update warmup mock to match new `get_feat` path

**File:** `tests/test_insightface_integration.py`

- Rename to `tests/test_arcface_integration.py`
- Update to use new `EmbeddingExtractor` directly (no insightface imports)
- Same skip-if-model-unavailable pattern

**File:** `tests/test_age_gender_stage.py`

- Replace `cv2.imwrite`/`cv2.imread` in `TestMiVOLOThreadSafety` with Pillow equivalents

### Task 7: Update documentation

- **`CLAUDE.md`**: Update Face Embeddings section (remove InsightFace references, document ONNX direct loading)
- **`docs/DESIGN.md`**: Update Stage 3 description and model setup section
- **`CLAUDE.md` Free-threaded Python section**: opencv-python blocker note may need updating if we confirm it's only transitive now

## Execution Order

1. **Task 1** — Core rewrite of EmbeddingExtractor (biggest change)
2. **Task 2** — Update PersonDetector warmup (depends on Task 1 API)
3. **Task 6** — Update tests (depends on Task 1+2 API)
4. **Task 3+4** — Download script + fallback removal (independent)
5. **Task 5** — Remove dependency (after everything works)
6. **Task 7** — Documentation (last)

## Verification

- All 229+ existing tests pass
- Integration test with real model: embedding dimensions = 512, L2-normalized
- Embeddings from new code match embeddings from old code for same input (regression test)
- `import insightface` and `import cv2` not present anywhere in `src/`

## Files Modified

- `src/photodb/utils/embedding_extractor.py` — primary rewrite
- `src/photodb/utils/person_detector.py` — warmup call update
- `scripts/download_models.sh` — direct ONNX download
- `pyproject.toml` — remove insightface dependency
- `tests/test_embedding_extractor.py` — update mocks
- `tests/test_person_detector.py` — update warmup mock
- `tests/test_insightface_integration.py` — rename + update
- `tests/test_age_gender_stage.py` — replace cv2 in thread safety test
- `CLAUDE.md` — update docs
- `docs/DESIGN.md` — update docs

## Risk Assessment

**Low risk**: Same ONNX model, same ONNX Runtime backend, same embedding output. The only change is who does the preprocessing (us vs insightface wrapper). Regression test comparing old vs new embeddings will catch any discrepancies.

**Note on opencv-python**: After this change, `import cv2` will not appear anywhere in `src/`. However, opencv-python remains installed as a transitive dependency of mivolo and ultralytics.

Our import chain is:

  age_gender.py → mivolo.model.mi_volo.MiVOLO
    → mivolo.structures (import cv2 at line 6)
    → mivolo.data.misc (import cv2 at line 6)

  Both are top-level unconditional imports. Even though the cv2 calls are only in methods we don't use (like save() and video-related functions), the import cv2 statement runs at module load time.

  So the answer is no, not without forking mivolo. Here's the full picture:
  ┌──────────────────────┬──────────────────────┬─────────────────────────────────────┬──────────────────────────┐
  │      Dependency      │     cv2 imports      │          On our code path?          │        Removable?        │
  ├──────────────────────┼──────────────────────┼─────────────────────────────────────┼──────────────────────────┤
  │ mivolo structures.py │ Top-level import cv2 │ Yes (we import PersonAndFaceResult) │ Only by forking mivolo   │
  ├──────────────────────┼──────────────────────┼─────────────────────────────────────┼──────────────────────────┤
  │ mivolo data/misc.py  │ Top-level import cv2 │ Yes (imported by structures.py)     │ Only by forking mivolo   │
  ├──────────────────────┼──────────────────────┼─────────────────────────────────────┼──────────────────────────┤
  │ ultralytics          │ 41 files             │ Yes (YOLO detection)                │ No                       │
  ├──────────────────────┼──────────────────────┼─────────────────────────────────────┼──────────────────────────┤
  │ insightface          │ arcface_onnx.py      │ Yes (but plan removes this)         │ Yes (Task 1 of the plan) │
  └──────────────────────┴──────────────────────┴─────────────────────────────────────┴──────────────────────────┘
  ultralytics is the hardest — cv2 is woven through 41 files. mivolo is more surgical — just 2 files on our path, and the cv2 usage is only in plot(), save(), and video methods we never call.

  As for free-threaded Python 3.13t: opencv-python still has <https://github.com/opencv/opencv-python/issues/1029>, and the <https://github.com/opencv/opencv-python/issues/1146> is also unresolved. So even if we forked mivolo to lazy-import cv2, ultralytics would still block us.

  Bottom line: Dropping the GIL requires replacing ultralytics with a YOLO implementation that doesn't depend on opencv. That's a much bigger project than the insightface removal.

  Sources:

- <https://github.com/opencv/opencv-python/issues/1029>
- <https://github.com/opencv/opencv-python/issues/1146>
- <https://forum.opencv.org/t/opencv-python-compatibility-with-python-versions-3-13-experimental-rc-releases-for-multithreading/23902>
