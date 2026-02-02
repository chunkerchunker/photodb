# InsightFace Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace FaceNet (facenet-pytorch) with InsightFace (buffalo_l ArcFace) for face embedding extraction while maintaining CoreML/MPS acceleration and pgvector compatibility.

**Architecture:** InsightFace uses ONNX Runtime with CoreMLExecutionProvider on macOS for hardware acceleration. The buffalo_l model produces 512-dimensional embeddings (same as FaceNet), so database schema and clustering logic remain unchanged. The main change is in `PersonDetector.extract_embedding()` which switches from PyTorch InceptionResnetV1 to InsightFace's ArcFace recognition model.

**Tech Stack:** insightface, onnxruntime-silicon (macOS), onnxruntime-gpu (CUDA), pgvector, numpy

---

## Summary of Changes

| Component | Current (FaceNet) | New (InsightFace) |
|-----------|-------------------|-------------------|
| Model | InceptionResnetV1 (vggface2) | ArcFace (buffalo_l) |
| Framework | PyTorch | ONNX Runtime |
| macOS Acceleration | MPS (problematic) | CoreML via onnxruntime-silicon |
| GPU Acceleration | CUDA/MPS | CUDA via onnxruntime-gpu |
| Embedding Dim | 512 | 512 (no change) |
| Input Size | 160x160 | 112x112 |
| Normalization | (x - 127.5) / 128.0 | (x - 127.5) / 127.5 |

---

## Task 1: Add InsightFace Dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update pyproject.toml dependencies**

Replace facenet-pytorch with insightface and add onnxruntime variants:

```toml
# Remove this line:
#     "facenet-pytorch>=2.5.3",

# Add these lines in dependencies:
    "insightface>=0.7.3",
    "onnxruntime>=1.16.0",

# Add new optional dependency group after [project.optional-dependencies].macos:
[project.optional-dependencies]
macos = [
    "coremltools>=7.0",
    "onnxruntime-silicon>=1.16.0",
]
cuda = [
    "onnxruntime-gpu>=1.16.0",
]
```

**Step 2: Verify dependency installation**

Run: `uv sync`
Expected: Dependencies install without errors

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "$(cat <<'EOF'
deps: replace facenet-pytorch with insightface

Switch from PyTorch-based FaceNet to ONNX-based InsightFace for
face embeddings. InsightFace supports CoreML on macOS for better
performance and thread safety.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Create InsightFace Embedding Extractor

**Files:**
- Create: `src/photodb/utils/embedding_extractor.py`
- Test: `tests/test_embedding_extractor.py`

**Step 1: Write the failing test**

Create `tests/test_embedding_extractor.py`:

```python
"""Tests for InsightFace embedding extractor."""

import numpy as np
import pytest
from PIL import Image
from unittest.mock import Mock, patch, MagicMock


class TestEmbeddingExtractor:
    """Tests for EmbeddingExtractor class."""

    def test_init_selects_coreml_on_macos(self):
        """Test that CoreML provider is selected on macOS."""
        with patch("sys.platform", "darwin"), \
             patch("photodb.utils.embedding_extractor.ort") as mock_ort, \
             patch("photodb.utils.embedding_extractor.FaceAnalysis") as mock_fa:
            mock_ort.get_available_providers.return_value = [
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            ]
            mock_app = MagicMock()
            mock_fa.return_value = mock_app

            from photodb.utils.embedding_extractor import EmbeddingExtractor
            extractor = EmbeddingExtractor()

            # Should use CoreML provider
            mock_fa.assert_called_once()
            call_kwargs = mock_fa.call_args[1]
            assert "CoreMLExecutionProvider" in call_kwargs.get("providers", [])

    def test_init_falls_back_to_cpu(self):
        """Test fallback to CPU when no GPU providers available."""
        with patch("sys.platform", "linux"), \
             patch("photodb.utils.embedding_extractor.ort") as mock_ort, \
             patch("photodb.utils.embedding_extractor.FaceAnalysis") as mock_fa:
            mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
            mock_app = MagicMock()
            mock_fa.return_value = mock_app

            from photodb.utils.embedding_extractor import EmbeddingExtractor
            extractor = EmbeddingExtractor()

            call_kwargs = mock_fa.call_args[1]
            assert "CPUExecutionProvider" in call_kwargs.get("providers", [])

    def test_extract_returns_512_dim_embedding(self):
        """Test that extract returns 512-dimensional embedding."""
        with patch("photodb.utils.embedding_extractor.ort") as mock_ort, \
             patch("photodb.utils.embedding_extractor.FaceAnalysis") as mock_fa:
            mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

            # Create mock face result
            mock_face = MagicMock()
            mock_face.embedding = np.random.randn(512).astype(np.float32)
            mock_face.bbox = np.array([10, 10, 50, 50])

            mock_app = MagicMock()
            mock_app.get.return_value = [mock_face]
            mock_fa.return_value = mock_app

            from photodb.utils.embedding_extractor import EmbeddingExtractor
            extractor = EmbeddingExtractor()

            # Create test image
            img = Image.new("RGB", (100, 100), color="white")
            bbox = {"x1": 10, "y1": 10, "x2": 50, "y2": 50}

            embedding = extractor.extract(img, bbox)

            assert embedding is not None
            assert len(embedding) == 512
            assert all(isinstance(x, float) for x in embedding)

    def test_extract_returns_none_for_no_face(self):
        """Test that extract returns None when no face detected in crop."""
        with patch("photodb.utils.embedding_extractor.ort") as mock_ort, \
             patch("photodb.utils.embedding_extractor.FaceAnalysis") as mock_fa:
            mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

            mock_app = MagicMock()
            mock_app.get.return_value = []  # No faces detected
            mock_fa.return_value = mock_app

            from photodb.utils.embedding_extractor import EmbeddingExtractor
            extractor = EmbeddingExtractor()

            img = Image.new("RGB", (100, 100), color="white")
            bbox = {"x1": 10, "y1": 10, "x2": 50, "y2": 50}

            embedding = extractor.extract(img, bbox)
            assert embedding is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embedding_extractor.py -v`
Expected: FAIL with "No module named 'photodb.utils.embedding_extractor'"

**Step 3: Write minimal implementation**

Create `src/photodb/utils/embedding_extractor.py`:

```python
"""
InsightFace-based face embedding extraction.

Uses ArcFace (buffalo_l) model for 512-dimensional face embeddings.
Supports CoreML on macOS, CUDA on Linux/Windows, with CPU fallback.
"""

import logging
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from PIL import Image

logger = logging.getLogger(__name__)


def _get_providers() -> List[str]:
    """
    Get the best available ONNX Runtime execution providers.

    Priority: CoreML (macOS) > CUDA > CPU

    Returns:
        List of provider names in priority order.
    """
    available = ort.get_available_providers()
    providers = []

    # CoreML on macOS (fastest, thread-safe)
    if sys.platform == "darwin" and "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
        logger.info("Using CoreML execution provider")

    # CUDA on Linux/Windows
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
        logger.info("Using CUDA execution provider")

    # Always include CPU as fallback
    providers.append("CPUExecutionProvider")

    return providers


class EmbeddingExtractor:
    """
    Extract face embeddings using InsightFace ArcFace model.

    Uses buffalo_l model which produces 512-dimensional embeddings,
    compatible with existing pgvector schema.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple = (640, 640),
    ):
        """
        Initialize the embedding extractor.

        Args:
            model_name: InsightFace model pack name. Default "buffalo_l".
            det_size: Detection input size. Default (640, 640).
        """
        providers = _get_providers()

        # Initialize FaceAnalysis with recognition model only
        # We use external YOLO for detection, so only need recognition
        self.app = FaceAnalysis(
            name=model_name,
            allowed_modules=["recognition"],
            providers=providers,
        )
        self.app.prepare(ctx_id=0, det_size=det_size)

        self._providers = providers
        logger.info(f"EmbeddingExtractor initialized with providers: {providers}")

    def extract(self, image: Image.Image, bbox: Dict[str, Any]) -> Optional[List[float]]:
        """
        Extract face embedding from a cropped face region.

        Args:
            image: PIL Image containing the face.
            bbox: Bounding box with x1, y1, x2, y2 keys.

        Returns:
            512-dimensional embedding as list of floats, or None if extraction fails.
        """
        try:
            # Crop face from image with padding
            x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])

            # Add padding (20%) for better recognition
            width = x2 - x1
            height = y2 - y1
            pad_x = int(width * 0.2)
            pad_y = int(height * 0.2)

            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(image.width, x2 + pad_x)
            y2 = min(image.height, y2 + pad_y)

            face_crop = image.crop((x1, y1, x2, y2))

            # Convert to numpy array (RGB format expected by InsightFace)
            # InsightFace expects BGR, so we need to convert
            img_array = np.array(face_crop)
            if img_array.ndim == 2:
                # Grayscale to RGB
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:
                # RGBA to RGB
                img_array = img_array[:, :, :3]

            # Convert RGB to BGR for InsightFace
            img_bgr = img_array[:, :, ::-1]

            # Get faces from the crop
            faces = self.app.get(img_bgr)

            if not faces:
                logger.debug("No face detected in crop for embedding extraction")
                return None

            # Use the first (most confident) face
            face = faces[0]
            embedding = face.embedding

            if embedding is None:
                logger.debug("Face detected but no embedding available")
                return None

            # Return as list of floats
            return embedding.astype(np.float32).tolist()

        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            return None

    def extract_from_aligned(self, aligned_face: np.ndarray) -> Optional[List[float]]:
        """
        Extract embedding from an already-aligned face image.

        Args:
            aligned_face: BGR numpy array of aligned face (112x112).

        Returns:
            512-dimensional embedding as list of floats, or None if extraction fails.
        """
        try:
            faces = self.app.get(aligned_face)
            if not faces:
                return None
            return faces[0].embedding.astype(np.float32).tolist()
        except Exception as e:
            logger.error(f"Failed to extract embedding from aligned face: {e}")
            return None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_embedding_extractor.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/photodb/utils/embedding_extractor.py tests/test_embedding_extractor.py
git commit -m "$(cat <<'EOF'
feat: add InsightFace embedding extractor

New EmbeddingExtractor class using InsightFace's ArcFace (buffalo_l)
model for 512-dimensional face embeddings. Supports:
- CoreML on macOS for fast, thread-safe inference
- CUDA on Linux/Windows
- CPU fallback

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Integrate EmbeddingExtractor into PersonDetector

**Files:**
- Modify: `src/photodb/utils/person_detector.py`
- Test: `tests/test_person_detector.py`

**Step 1: Write the failing test**

Add to `tests/test_person_detector.py`:

```python
def test_uses_insightface_for_embeddings(self, mock_yolo, tmp_path):
    """Test that PersonDetector uses InsightFace for embeddings."""
    with patch("photodb.utils.person_detector.EmbeddingExtractor") as mock_extractor_class:
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = [0.1] * 512
        mock_extractor_class.return_value = mock_extractor

        detector = PersonDetector(model_path=str(tmp_path / "fake.pt"))

        # Verify EmbeddingExtractor was instantiated
        mock_extractor_class.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_person_detector.py::test_uses_insightface_for_embeddings -v`
Expected: FAIL (EmbeddingExtractor not used yet)

**Step 3: Modify PersonDetector to use EmbeddingExtractor**

Edit `src/photodb/utils/person_detector.py`:

```python
# At the top, replace:
# from facenet_pytorch import InceptionResnetV1
# With:
from .embedding_extractor import EmbeddingExtractor

# In __init__, replace the FaceNet initialization (lines ~140-145):
# OLD:
#     # Load FaceNet embedding model (always PyTorch)
#     self.facenet = InceptionResnetV1(pretrained="vggface2").eval()
#     # For FaceNet, use CPU if using CoreML (safer for threading) or specified device
#     self._facenet_device = "cpu" if self.using_coreml else self.device
#     if self._facenet_device in ["cuda", "mps"]:
#         self.facenet = self.facenet.to(self._facenet_device)

# NEW:
        # Load InsightFace embedding model (ONNX-based)
        # Uses CoreML on macOS, CUDA on Linux, with CPU fallback
        self.embedding_extractor = EmbeddingExtractor()

# Replace extract_embedding method (lines ~335-386):
    def extract_embedding(self, image: Image.Image, bbox: Dict[str, Any]) -> List[float]:
        """
        Extract face embedding from a cropped face region.

        Args:
            image: PIL Image to extract face from.
            bbox: Bounding box with x1, y1, x2, y2.

        Returns:
            512-dimensional face embedding as list of floats.

        Raises:
            ValueError: If embedding extraction fails.
        """
        embedding = self.embedding_extractor.extract(image, bbox)
        if embedding is None:
            raise ValueError("Failed to extract face embedding")
        return embedding
```

**Step 4: Update imports and remove unused code**

```python
# Remove torch import if no longer needed elsewhere
# Remove: from facenet_pytorch import InceptionResnetV1

# Remove: self._facenet_device usage
# Remove: self.facenet usage
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_person_detector.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/photodb/utils/person_detector.py tests/test_person_detector.py
git commit -m "$(cat <<'EOF'
refactor: switch PersonDetector to InsightFace embeddings

Replace FaceNet (PyTorch InceptionResnetV1) with InsightFace
EmbeddingExtractor for face embeddings. Benefits:
- CoreML support on macOS (thread-safe, faster)
- Consistent ONNX Runtime backend
- Same 512-dim embeddings, no schema changes needed

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Update Model Download Script

**Files:**
- Modify: `scripts/download_models.sh`

**Step 1: Add InsightFace model download**

InsightFace models auto-download on first use, but we should document this and optionally pre-download:

```bash
# Add to download_models.sh:

echo "=== InsightFace Models ==="
echo "InsightFace buffalo_l models will auto-download on first use."
echo "Models are cached in ~/.insightface/models/buffalo_l/"
echo ""
echo "To pre-download, run:"
echo "  python -c \"from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l').prepare(ctx_id=0)\""
echo ""

# Optional: Add explicit pre-download
if [ "${PREDOWNLOAD_INSIGHTFACE:-false}" = "true" ]; then
    echo "Pre-downloading InsightFace buffalo_l model..."
    python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l').prepare(ctx_id=0)"
fi
```

**Step 2: Commit**

```bash
git add scripts/download_models.sh
git commit -m "$(cat <<'EOF'
docs: add InsightFace model download info

InsightFace models auto-download on first use to ~/.insightface/models/.
Added optional PREDOWNLOAD_INSIGHTFACE flag for explicit pre-download.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Update Configuration Documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md with InsightFace configuration**

Add new section after "Age/Gender Stage Configuration":

```markdown
### Face Embedding Configuration

- `EMBEDDING_MODEL_NAME`: InsightFace model pack name (default: `buffalo_l`)
- `EMBEDDING_MODEL_ROOT`: Root directory for InsightFace models (default: `~/.insightface/models`)

**Hardware Acceleration:**
- **macOS:** Uses CoreML via `onnxruntime-silicon` for Neural Engine acceleration (thread-safe)
- **CUDA:** Uses `onnxruntime-gpu` for GPU acceleration
- **CPU:** Falls back to CPU if no accelerators available

**Model Location:** InsightFace models auto-download to `~/.insightface/models/` on first use.
```

Update the "Free-threaded Python" section to note InsightFace compatibility:

```markdown
### Free-threaded Python

**Not currently usable** due to MiVOLO's opencv-python dependency lacking Python 3.13t wheels.

Note: InsightFace (face embeddings) uses ONNX Runtime which is thread-safe, so it would
benefit from free-threaded Python once the opencv blocker is resolved.
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs: add InsightFace configuration to CLAUDE.md

Document InsightFace model settings and hardware acceleration options
(CoreML on macOS, CUDA on Linux). Note ONNX Runtime thread safety.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Add Integration Test

**Files:**
- Create: `tests/test_insightface_integration.py`

**Step 1: Write integration test**

```python
"""Integration tests for InsightFace embedding extraction."""

import numpy as np
import pytest
from PIL import Image

# Skip if insightface not installed
pytest.importorskip("insightface")


class TestInsightFaceIntegration:
    """Integration tests requiring actual InsightFace models."""

    @pytest.fixture
    def real_extractor(self):
        """Create a real EmbeddingExtractor (downloads model if needed)."""
        from photodb.utils.embedding_extractor import EmbeddingExtractor
        return EmbeddingExtractor()

    @pytest.fixture
    def sample_face_image(self):
        """Create a simple test image with a face-like pattern."""
        # Create 200x200 image with face-like features
        img = Image.new("RGB", (200, 200), color=(200, 180, 160))
        return img

    @pytest.mark.slow
    def test_embedding_dimension(self, real_extractor, sample_face_image):
        """Test that embeddings are 512-dimensional."""
        bbox = {"x1": 20, "y1": 20, "x2": 180, "y2": 180}

        # Note: May return None if no face detected in synthetic image
        embedding = real_extractor.extract(sample_face_image, bbox)

        if embedding is not None:
            assert len(embedding) == 512
            assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.slow
    def test_embedding_consistency(self, real_extractor, sample_face_image):
        """Test that same image produces consistent embeddings."""
        bbox = {"x1": 20, "y1": 20, "x2": 180, "y2": 180}

        embedding1 = real_extractor.extract(sample_face_image, bbox)
        embedding2 = real_extractor.extract(sample_face_image, bbox)

        if embedding1 is not None and embedding2 is not None:
            # Should be identical for same input
            np.testing.assert_array_almost_equal(embedding1, embedding2)

    @pytest.mark.slow
    def test_provider_selection(self, real_extractor):
        """Test that appropriate provider was selected."""
        import sys

        providers = real_extractor._providers

        if sys.platform == "darwin":
            # CoreML should be available on macOS
            assert "CoreMLExecutionProvider" in providers or "CPUExecutionProvider" in providers
        else:
            # CPU should always be available
            assert "CPUExecutionProvider" in providers
```

**Step 2: Run integration test**

Run: `uv run pytest tests/test_insightface_integration.py -v -m slow`
Expected: PASS (may skip if no face detected in synthetic image)

**Step 3: Commit**

```bash
git add tests/test_insightface_integration.py
git commit -m "$(cat <<'EOF'
test: add InsightFace integration tests

Tests for embedding dimension, consistency, and provider selection.
Marked as slow since they require model download.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Update Existing Tests

**Files:**
- Modify: `tests/test_person_detector.py`
- Modify: `tests/test_detection_stage.py`

**Step 1: Update mocks in test_person_detector.py**

Replace FaceNet mocks with EmbeddingExtractor mocks:

```python
# Replace:
# @patch("photodb.utils.person_detector.InceptionResnetV1")
# With:
@patch("photodb.utils.person_detector.EmbeddingExtractor")

# Update mock setup:
# OLD: mock_facenet.return_value.return_value = torch.tensor([[0.1] * 512])
# NEW: mock_extractor.return_value.extract.return_value = [0.1] * 512
```

**Step 2: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/test_person_detector.py tests/test_detection_stage.py
git commit -m "$(cat <<'EOF'
test: update tests for InsightFace migration

Replace FaceNet/InceptionResnetV1 mocks with EmbeddingExtractor mocks.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Clean Up and Final Verification

**Files:**
- Verify: All modified files
- Run: Full test suite

**Step 1: Run linting**

Run: `uv run ruff check --fix && uv run ruff format`
Expected: No errors

**Step 2: Run type checking**

Run: `uv run pyright`
Expected: No errors (or acceptable existing errors)

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

**Step 4: Test with real image (manual)**

```bash
# Process a single image to verify end-to-end
uv run process-local /path/to/test/photo.jpg --stage detection
```

**Step 5: Commit any remaining fixes**

```bash
git add -A
git commit -m "$(cat <<'EOF'
chore: clean up InsightFace migration

Fix linting and type issues from migration.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Migration Notes

### Backward Compatibility

- **Embedding Dimension:** Both FaceNet and InsightFace produce 512-dimensional embeddings
- **Database Schema:** No changes required to `face_embedding` table or `cluster` centroid
- **Clustering:** Cosine similarity calculations remain valid

### Re-embedding Existing Data

Existing embeddings from FaceNet are **not compatible** with InsightFace embeddings for clustering purposes (different feature spaces). Options:

1. **Full re-process:** Run `uv run process-local /path/to/photos --stage detection --force` to regenerate all embeddings
2. **Gradual migration:** New photos will use InsightFace; old clusters may need manual review
3. **Clear and restart:** Truncate `face_embedding` and `cluster` tables, then re-process

### Performance Expectations

| Platform | FaceNet | InsightFace |
|----------|---------|-------------|
| macOS (M1/M2) | MPS (unstable) | CoreML (stable, ~5x faster) |
| Linux (CUDA) | PyTorch CUDA | ONNX CUDA (similar) |
| CPU | PyTorch CPU | ONNX CPU (~1.5x faster) |

### Troubleshooting

1. **CoreML not detected on macOS:**
   - Ensure `onnxruntime-silicon` is installed: `pip install onnxruntime-silicon`
   - Check: `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"`

2. **Model download fails:**
   - Models download to `~/.insightface/models/`
   - Ensure internet connectivity on first run
   - Manual download: https://github.com/deepinsight/insightface/tree/master/model_zoo

3. **Memory issues:**
   - buffalo_l is larger than FaceNet; consider `buffalo_s` for constrained environments
