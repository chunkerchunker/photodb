# Scene Taxonomy & Sentiment Analysis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add scene taxonomy classification (Apple Vision) and sentiment analysis (MobileCLIP) to the existing YOLO+MiVOLO detection pipeline, without replacing current detection capabilities.

**Architecture:** Keep existing YOLO→MiVOLO→InsightFace pipeline for detection/age-gender/embeddings. Add two new parallel analyzers: (1) Apple Vision `VNClassifyImageRequest` for scene taxonomy (1303 built-in labels), (2) MobileCLIP for zero-shot sentiment classification of both full scenes and cropped faces. Store results in new generalizable `analysis_output` table.

**Tech Stack:** PyObjC (`pyobjc-framework-Vision`), MobileCLIP via `open_clip`, CoreML for acceleration, PostgreSQL with JSONB.

---

## Architecture Overview

```
Photo (normalized)
  │
  ├─► YOLO ─────────────────► Face/Body boxes ─┬─► MiVOLO ──────► Age/Gender
  │   (existing)                               │
  │                                            ├─► InsightFace ─► Embeddings
  │                                            │
  │                                            └─► MobileCLIP ──► Face sentiment
  │                                                (face crops)    (per detection)
  │
  ├─► VNClassifyImageRequest ─► Scene taxonomy (1303 labels)
  │   (NEW - Apple Vision)      "outdoor", "beach", "wedding", etc.
  │
  └─► MobileCLIP ─────────────► Scene sentiment
      (NEW - full image)        "joyful", "somber", "peaceful", etc.
```

## What's New vs Existing

| Component | Status | Purpose |
|-----------|--------|---------|
| YOLO detection | Existing | Face/body bounding boxes |
| MiVOLO | Existing | Age/gender estimation |
| InsightFace | Existing | Face embeddings for clustering |
| **VNClassifyImageRequest** | **NEW** | Scene taxonomy (1303 labels) |
| **MobileCLIP** | **NEW** | Sentiment (scene + face) |

---

## Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add new dependencies**

```toml
# In dependencies list, add:
    "pyobjc-framework-Vision>=10.0",
    "pyobjc-framework-Quartz>=10.0",
    "open-clip-torch>=2.24.0",
```

**Step 2: Sync dependencies**

Run: `uv sync`
Expected: Successfully installs packages

**Step 3: Verify imports**

Run: `uv run python -c "import Vision; import open_clip; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add pyobjc-framework-Vision and open-clip-torch"
```

---

## Task 2: Create Database Migration

**Files:**
- Create: `migrations/006_add_scene_sentiment_analysis.sql`

**Step 1: Write migration SQL**

```sql
-- Analysis output: Model-agnostic storage for raw outputs
-- Supports multiple models without schema changes
CREATE TABLE IF NOT EXISTS analysis_output (
    id bigserial PRIMARY KEY,
    photo_id bigint NOT NULL,

    -- Model identification
    model_type text NOT NULL,  -- 'classifier', 'sentiment', 'detector'
    model_name text NOT NULL,  -- 'apple_vision_classify', 'mobileclip', etc.
    model_version text,

    -- Raw output (schema varies by model)
    output jsonb NOT NULL,

    -- Processing metadata
    processing_time_ms integer,
    device text,  -- 'cpu', 'ane', 'gpu'

    created_at timestamptz DEFAULT now(),

    FOREIGN KEY (photo_id) REFERENCES photo(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_analysis_output_photo_model
    ON analysis_output(photo_id, model_type, model_name);

CREATE INDEX IF NOT EXISTS idx_analysis_output_output
    ON analysis_output USING GIN(output);

-- Scene analysis: Photo-level classification and sentiment
CREATE TABLE IF NOT EXISTS scene_analysis (
    id bigserial PRIMARY KEY,
    photo_id bigint NOT NULL UNIQUE,

    -- Apple Vision taxonomy (VNClassifyImageRequest)
    taxonomy_labels text[],       -- Top labels: ["outdoor", "beach", "sunny"]
    taxonomy_confidences real[],  -- Corresponding confidences

    -- MobileCLIP scene sentiment
    scene_sentiment text CHECK (scene_sentiment IN (
        'joyful', 'peaceful', 'somber', 'tense', 'neutral', 'energetic'
    )),
    scene_sentiment_confidence real,
    scene_sentiment_scores jsonb,  -- All sentiment scores for debugging

    -- References to raw outputs
    taxonomy_output_id bigint REFERENCES analysis_output(id) ON DELETE SET NULL,
    sentiment_output_id bigint REFERENCES analysis_output(id) ON DELETE SET NULL,

    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),

    FOREIGN KEY (photo_id) REFERENCES photo(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_scene_analysis_labels
    ON scene_analysis USING GIN(taxonomy_labels);
CREATE INDEX IF NOT EXISTS idx_scene_analysis_sentiment
    ON scene_analysis(scene_sentiment);

-- Face sentiment: Per-detection sentiment from MobileCLIP
-- Links to existing person_detection table
ALTER TABLE person_detection
    ADD COLUMN IF NOT EXISTS face_sentiment text CHECK (face_sentiment IN (
        'happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted', 'neutral'
    )),
    ADD COLUMN IF NOT EXISTS face_sentiment_confidence real,
    ADD COLUMN IF NOT EXISTS face_sentiment_scores jsonb,
    ADD COLUMN IF NOT EXISTS sentiment_output_id bigint REFERENCES analysis_output(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_person_detection_face_sentiment
    ON person_detection(face_sentiment);

-- Model registry: Track available analyzers and their capabilities
CREATE TABLE IF NOT EXISTS model_registry (
    id bigserial PRIMARY KEY,
    name text UNIQUE NOT NULL,
    display_name text NOT NULL,
    model_type text NOT NULL,  -- 'detector', 'classifier', 'sentiment', 'embedder'
    capabilities text[] NOT NULL,
    config jsonb,
    is_active boolean DEFAULT true,
    created_at timestamptz DEFAULT now()
);

-- Insert known models
INSERT INTO model_registry (name, display_name, model_type, capabilities, config) VALUES
    ('yolo_person_face', 'YOLO Person+Face', 'detector',
     ARRAY['face_detection', 'body_detection'],
     '{"model": "yolov8x_person_face"}'),
    ('mivolo', 'MiVOLO', 'estimator',
     ARRAY['age_estimation', 'gender_estimation'],
     '{}'),
    ('insightface', 'InsightFace', 'embedder',
     ARRAY['face_embedding'],
     '{"model": "buffalo_l", "dimension": 512}'),
    ('apple_vision_classify', 'Apple Vision Classify', 'classifier',
     ARRAY['scene_taxonomy'],
     '{"classes": 1303}'),
    ('mobileclip', 'MobileCLIP', 'sentiment',
     ARRAY['scene_sentiment', 'face_sentiment', 'zero_shot'],
     '{"model": "MobileCLIP-S2"}')
ON CONFLICT (name) DO NOTHING;
```

**Step 2: Apply migration**

Run: `psql $DATABASE_URL -f migrations/006_add_scene_sentiment_analysis.sql`
Expected: Tables and columns created

**Step 3: Verify schema**

Run: `psql $DATABASE_URL -c "\d scene_analysis"`
Expected: Shows expected columns

**Step 4: Commit**

```bash
git add migrations/006_add_scene_sentiment_analysis.sql
git commit -m "db: add scene_analysis table and face_sentiment columns"
```

---

## Task 3: Add Database Models

**Files:**
- Modify: `src/photodb/database/models.py`

**Step 1: Add AnalysisOutput dataclass**

Add after the `PersonDetection` class:

```python
@dataclass
class AnalysisOutput:
    """Model-agnostic storage for raw analysis outputs."""

    id: Optional[int]
    photo_id: int
    model_type: str  # 'classifier', 'sentiment', 'detector'
    model_name: str  # 'apple_vision_classify', 'mobileclip', etc.
    model_version: Optional[str]
    output: Dict[str, Any]
    processing_time_ms: Optional[int]
    device: Optional[str]
    created_at: Optional[datetime]

    @classmethod
    def create(
        cls,
        photo_id: int,
        model_type: str,
        model_name: str,
        output: Dict[str, Any],
        model_version: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
        device: Optional[str] = None,
    ) -> "AnalysisOutput":
        return cls(
            id=None,
            photo_id=photo_id,
            model_type=model_type,
            model_name=model_name,
            model_version=model_version,
            output=output,
            processing_time_ms=processing_time_ms,
            device=device,
            created_at=datetime.now(timezone.utc),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "photo_id": self.photo_id,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "output": json.dumps(self.output),
            "processing_time_ms": self.processing_time_ms,
            "device": self.device,
            "created_at": self.created_at,
        }


@dataclass
class SceneAnalysis:
    """Photo-level scene classification and sentiment."""

    id: Optional[int]
    photo_id: int
    # Apple Vision taxonomy
    taxonomy_labels: Optional[List[str]]
    taxonomy_confidences: Optional[List[float]]
    # MobileCLIP sentiment
    scene_sentiment: Optional[str]
    scene_sentiment_confidence: Optional[float]
    scene_sentiment_scores: Optional[Dict[str, float]]
    # Output references
    taxonomy_output_id: Optional[int]
    sentiment_output_id: Optional[int]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    @classmethod
    def create(
        cls,
        photo_id: int,
        taxonomy_labels: Optional[List[str]] = None,
        taxonomy_confidences: Optional[List[float]] = None,
        scene_sentiment: Optional[str] = None,
        scene_sentiment_confidence: Optional[float] = None,
        scene_sentiment_scores: Optional[Dict[str, float]] = None,
        taxonomy_output_id: Optional[int] = None,
        sentiment_output_id: Optional[int] = None,
    ) -> "SceneAnalysis":
        now = datetime.now(timezone.utc)
        return cls(
            id=None,
            photo_id=photo_id,
            taxonomy_labels=taxonomy_labels,
            taxonomy_confidences=taxonomy_confidences,
            scene_sentiment=scene_sentiment,
            scene_sentiment_confidence=scene_sentiment_confidence,
            scene_sentiment_scores=scene_sentiment_scores,
            taxonomy_output_id=taxonomy_output_id,
            sentiment_output_id=sentiment_output_id,
            created_at=now,
            updated_at=now,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "photo_id": self.photo_id,
            "taxonomy_labels": self.taxonomy_labels,
            "taxonomy_confidences": self.taxonomy_confidences,
            "scene_sentiment": self.scene_sentiment,
            "scene_sentiment_confidence": self.scene_sentiment_confidence,
            "scene_sentiment_scores": self.scene_sentiment_scores,
            "taxonomy_output_id": self.taxonomy_output_id,
            "sentiment_output_id": self.sentiment_output_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
```

**Step 2: Commit**

```bash
git add src/photodb/database/models.py
git commit -m "models: add AnalysisOutput and SceneAnalysis dataclasses"
```

---

## Task 4: Add Repository Methods

**Files:**
- Modify: `src/photodb/database/pg_repository.py`

**Step 1: Add AnalysisOutput methods**

```python
def create_analysis_output(self, output: AnalysisOutput) -> AnalysisOutput:
    """Insert analysis output record."""
    with self.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO analysis_output
                (photo_id, model_type, model_name, model_version, output,
                 processing_time_ms, device, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    output.photo_id,
                    output.model_type,
                    output.model_name,
                    output.model_version,
                    json.dumps(output.output),
                    output.processing_time_ms,
                    output.device,
                    output.created_at,
                ),
            )
            output.id = cur.fetchone()[0]
            conn.commit()
    return output
```

**Step 2: Add SceneAnalysis methods**

```python
def upsert_scene_analysis(self, analysis: SceneAnalysis) -> SceneAnalysis:
    """Insert or update scene analysis for a photo."""
    with self.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO scene_analysis
                (photo_id, taxonomy_labels, taxonomy_confidences,
                 scene_sentiment, scene_sentiment_confidence, scene_sentiment_scores,
                 taxonomy_output_id, sentiment_output_id, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (photo_id) DO UPDATE SET
                    taxonomy_labels = COALESCE(EXCLUDED.taxonomy_labels, scene_analysis.taxonomy_labels),
                    taxonomy_confidences = COALESCE(EXCLUDED.taxonomy_confidences, scene_analysis.taxonomy_confidences),
                    scene_sentiment = COALESCE(EXCLUDED.scene_sentiment, scene_analysis.scene_sentiment),
                    scene_sentiment_confidence = COALESCE(EXCLUDED.scene_sentiment_confidence, scene_analysis.scene_sentiment_confidence),
                    scene_sentiment_scores = COALESCE(EXCLUDED.scene_sentiment_scores, scene_analysis.scene_sentiment_scores),
                    taxonomy_output_id = COALESCE(EXCLUDED.taxonomy_output_id, scene_analysis.taxonomy_output_id),
                    sentiment_output_id = COALESCE(EXCLUDED.sentiment_output_id, scene_analysis.sentiment_output_id),
                    updated_at = EXCLUDED.updated_at
                RETURNING id
                """,
                (
                    analysis.photo_id,
                    analysis.taxonomy_labels,
                    analysis.taxonomy_confidences,
                    analysis.scene_sentiment,
                    analysis.scene_sentiment_confidence,
                    json.dumps(analysis.scene_sentiment_scores) if analysis.scene_sentiment_scores else None,
                    analysis.taxonomy_output_id,
                    analysis.sentiment_output_id,
                    analysis.created_at,
                    analysis.updated_at,
                ),
            )
            analysis.id = cur.fetchone()[0]
            conn.commit()
    return analysis


def get_scene_analysis(self, photo_id: int) -> Optional[SceneAnalysis]:
    """Get scene analysis for a photo."""
    with self.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM scene_analysis WHERE photo_id = %s",
                (photo_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return SceneAnalysis(
                id=row[0],
                photo_id=row[1],
                taxonomy_labels=row[2],
                taxonomy_confidences=row[3],
                scene_sentiment=row[4],
                scene_sentiment_confidence=row[5],
                scene_sentiment_scores=row[6] if isinstance(row[6], dict) else json.loads(row[6]) if row[6] else None,
                taxonomy_output_id=row[7],
                sentiment_output_id=row[8],
                created_at=row[9],
                updated_at=row[10],
            )
```

**Step 3: Add face sentiment update method**

```python
def update_detection_face_sentiment(
    self,
    detection_id: int,
    face_sentiment: str,
    face_sentiment_confidence: float,
    face_sentiment_scores: Optional[Dict[str, float]] = None,
    sentiment_output_id: Optional[int] = None,
) -> None:
    """Update face sentiment for a person detection."""
    with self.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE person_detection
                SET face_sentiment = %s,
                    face_sentiment_confidence = %s,
                    face_sentiment_scores = %s,
                    sentiment_output_id = %s
                WHERE id = %s
                """,
                (
                    face_sentiment,
                    face_sentiment_confidence,
                    json.dumps(face_sentiment_scores) if face_sentiment_scores else None,
                    sentiment_output_id,
                    detection_id,
                ),
            )
            conn.commit()
```

**Step 4: Commit**

```bash
git add src/photodb/database/pg_repository.py
git commit -m "repo: add methods for AnalysisOutput and SceneAnalysis"
```

---

## Task 5: Create Apple Vision Scene Classifier

**Files:**
- Create: `src/photodb/utils/apple_vision_classifier.py`
- Test: `tests/test_apple_vision_classifier.py`

**Step 1: Write failing test**

```python
# tests/test_apple_vision_classifier.py
"""Tests for Apple Vision scene classifier."""
import sys
import pytest
from pathlib import Path

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="Apple Vision only available on macOS"
)


class TestAppleVisionClassifier:
    """Test Apple Vision scene classification."""

    def test_classify_returns_labels(self, test_image_path):
        """Classification should return labels with confidences."""
        from photodb.utils.apple_vision_classifier import AppleVisionClassifier

        classifier = AppleVisionClassifier()
        result = classifier.classify(str(test_image_path))

        assert result["status"] in ("success", "error")
        if result["status"] == "success":
            assert "classifications" in result
            assert len(result["classifications"]) > 0
            assert "identifier" in result["classifications"][0]
            assert "confidence" in result["classifications"][0]

    def test_classify_returns_top_k(self, test_image_path):
        """Should return requested number of top labels."""
        from photodb.utils.apple_vision_classifier import AppleVisionClassifier

        classifier = AppleVisionClassifier()
        result = classifier.classify(str(test_image_path), top_k=5)

        if result["status"] == "success":
            assert len(result["classifications"]) <= 5

    def test_classify_includes_timing(self, test_image_path):
        """Result should include processing time."""
        from photodb.utils.apple_vision_classifier import AppleVisionClassifier

        classifier = AppleVisionClassifier()
        result = classifier.classify(str(test_image_path))

        assert "processing_time_ms" in result


@pytest.fixture
def test_image_path():
    return Path(__file__).parent.parent / "test_photos" / "test.jpg"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_apple_vision_classifier.py -v`
Expected: `ModuleNotFoundError`

**Step 3: Implement AppleVisionClassifier**

```python
# src/photodb/utils/apple_vision_classifier.py
"""
Apple Vision scene classifier using VNClassifyImageRequest.

Provides 1303 scene taxonomy labels. Only available on macOS.
"""
import logging
import sys
import time
from typing import Any, Dict, List

if sys.platform != "darwin":
    raise ImportError("Apple Vision only available on macOS")

import Quartz
import Vision
from Foundation import NSURL

logger = logging.getLogger(__name__)


class AppleVisionClassifier:
    """Classify scene content using Apple Vision Framework.

    Uses VNClassifyImageRequest which provides 1303 built-in classification
    labels. Runs on Neural Engine when available.
    """

    def __init__(self):
        """Initialize classifier."""
        logger.info("AppleVisionClassifier initialized")

    def classify(
        self, image_path: str, top_k: int = 10, min_confidence: float = 0.01
    ) -> Dict[str, Any]:
        """
        Classify scene content in an image.

        Args:
            image_path: Path to the image file.
            top_k: Number of top classifications to return.
            min_confidence: Minimum confidence threshold.

        Returns:
            Dictionary with:
                - status: 'success' or 'error'
                - classifications: List of {identifier, confidence}
                - processing_time_ms: Time taken for classification
                - error: Error message (only if status is 'error')
        """
        start_time = time.time()

        try:
            # Load image
            image_url = NSURL.fileURLWithPath_(image_path)
            ci_image = Quartz.CIImage.imageWithContentsOfURL_(image_url)

            if ci_image is None:
                return {
                    "status": "error",
                    "classifications": [],
                    "error": f"Failed to load image: {image_path}",
                }

            # Create request handler
            handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
                ci_image, None
            )

            # Create classification request
            request = Vision.VNClassifyImageRequest.alloc().init()

            # Perform request
            success, error = handler.performRequests_error_([request], None)

            if not success:
                return {
                    "status": "error",
                    "classifications": [],
                    "error": str(error) if error else "Classification failed",
                    "processing_time_ms": int((time.time() - start_time) * 1000),
                }

            # Parse results
            results = request.results() or []

            classifications = []
            for observation in results:
                conf = float(observation.confidence())
                if conf >= min_confidence:
                    classifications.append({
                        "identifier": str(observation.identifier()),
                        "confidence": conf,
                    })

            # Sort by confidence and take top_k
            classifications.sort(key=lambda x: x["confidence"], reverse=True)
            classifications = classifications[:top_k]

            return {
                "status": "success",
                "classifications": classifications,
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {
                "status": "error",
                "classifications": [],
                "error": str(e),
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_apple_vision_classifier.py -v`
Expected: PASS (on macOS)

**Step 5: Commit**

```bash
git add src/photodb/utils/apple_vision_classifier.py tests/test_apple_vision_classifier.py
git commit -m "feat: add AppleVisionClassifier for scene taxonomy"
```

---

## Task 6: Create MobileCLIP Analyzer

**Files:**
- Create: `src/photodb/utils/mobileclip_analyzer.py`
- Test: `tests/test_mobileclip_analyzer.py`

**Step 1: Write failing test**

```python
# tests/test_mobileclip_analyzer.py
"""Tests for MobileCLIP sentiment analyzer."""
import pytest
from pathlib import Path
from PIL import Image
import numpy as np


class TestMobileCLIPAnalyzer:
    """Test MobileCLIP zero-shot sentiment analysis."""

    def test_analyze_scene_sentiment(self, test_image_path):
        """Scene sentiment should return valid sentiment."""
        from photodb.utils.mobileclip_analyzer import MobileCLIPAnalyzer

        analyzer = MobileCLIPAnalyzer()
        result = analyzer.analyze_scene_sentiment(str(test_image_path))

        assert "sentiment" in result
        assert result["sentiment"] in (
            "joyful", "peaceful", "somber", "tense", "neutral", "energetic"
        )
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1
        assert "scores" in result

    def test_analyze_face_sentiment(self, face_crop):
        """Face sentiment should return valid emotion."""
        from photodb.utils.mobileclip_analyzer import MobileCLIPAnalyzer

        analyzer = MobileCLIPAnalyzer()
        result = analyzer.analyze_face_sentiment(face_crop)

        assert "sentiment" in result
        assert result["sentiment"] in (
            "happy", "sad", "angry", "surprised", "fearful", "disgusted", "neutral"
        )
        assert "confidence" in result
        assert "scores" in result

    def test_analyze_face_sentiment_from_bbox(self, test_image_path):
        """Should extract and analyze face from bbox."""
        from photodb.utils.mobileclip_analyzer import MobileCLIPAnalyzer

        analyzer = MobileCLIPAnalyzer()
        # Mock bbox covering center of image
        bbox = {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
        result = analyzer.analyze_face_sentiment_from_image(
            str(test_image_path), bbox
        )

        assert "sentiment" in result
        assert "confidence" in result

    def test_batch_face_sentiments(self, test_image_path):
        """Should analyze multiple faces efficiently."""
        from photodb.utils.mobileclip_analyzer import MobileCLIPAnalyzer

        analyzer = MobileCLIPAnalyzer()
        bboxes = [
            {"x1": 50, "y1": 50, "x2": 150, "y2": 150},
            {"x1": 200, "y1": 50, "x2": 300, "y2": 150},
        ]
        results = analyzer.analyze_faces_batch(str(test_image_path), bboxes)

        assert len(results) == 2
        for result in results:
            assert "sentiment" in result


@pytest.fixture
def test_image_path():
    return Path(__file__).parent.parent / "test_photos" / "test.jpg"


@pytest.fixture
def face_crop():
    """Create a simple test face crop."""
    # 160x160 RGB image
    return Image.fromarray(
        np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mobileclip_analyzer.py -v`
Expected: `ModuleNotFoundError`

**Step 3: Implement MobileCLIPAnalyzer**

```python
# src/photodb/utils/mobileclip_analyzer.py
"""
MobileCLIP-based sentiment analyzer using zero-shot classification.

Analyzes both scene-level and face-level sentiment using text prompts.
Uses MobileCLIP-S2 for efficient inference.
"""
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy-load model to avoid import-time overhead
_model = None
_preprocess = None
_tokenizer = None


def _load_model():
    """Lazy-load MobileCLIP model."""
    global _model, _preprocess, _tokenizer

    if _model is not None:
        return _model, _preprocess, _tokenizer

    import open_clip

    logger.info("Loading MobileCLIP-S2 model...")
    _model, _, _preprocess = open_clip.create_model_and_transforms(
        "MobileCLIP-S2", pretrained="datacompdr"
    )
    _tokenizer = open_clip.get_tokenizer("MobileCLIP-S2")

    # Set to eval mode
    _model.eval()

    # Try to use MPS on macOS, else CPU
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    _model = _model.to(device)
    logger.info(f"MobileCLIP loaded on {device}")

    return _model, _preprocess, _tokenizer


# Scene sentiment prompts
SCENE_PROMPTS = {
    "joyful": "a joyful happy celebratory scene",
    "peaceful": "a peaceful calm serene scene",
    "somber": "a somber sad melancholic scene",
    "tense": "a tense dramatic intense scene",
    "neutral": "an ordinary everyday neutral scene",
    "energetic": "an energetic exciting dynamic scene",
}

# Face sentiment prompts
FACE_PROMPTS = {
    "happy": "a photo of a happy smiling person",
    "sad": "a photo of a sad unhappy person",
    "angry": "a photo of an angry frustrated person",
    "surprised": "a photo of a surprised shocked person",
    "fearful": "a photo of a fearful scared person",
    "disgusted": "a photo of a disgusted person",
    "neutral": "a photo of a person with neutral expression",
}


class MobileCLIPAnalyzer:
    """Zero-shot sentiment analysis using MobileCLIP.

    Provides:
    - Scene sentiment: joyful, peaceful, somber, tense, neutral, energetic
    - Face sentiment: happy, sad, angry, surprised, fearful, disgusted, neutral
    """

    def __init__(self):
        """Initialize analyzer (model loaded lazily on first use)."""
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = None

        # Pre-encoded text features (computed on first use)
        self._scene_text_features = None
        self._face_text_features = None

    def _ensure_loaded(self):
        """Ensure model is loaded."""
        if self._model is None:
            self._model, self._preprocess, self._tokenizer = _load_model()
            self._device = next(self._model.parameters()).device

    def _get_scene_text_features(self):
        """Get pre-encoded scene prompt features."""
        if self._scene_text_features is None:
            self._ensure_loaded()
            prompts = list(SCENE_PROMPTS.values())
            text_tokens = self._tokenizer(prompts).to(self._device)
            with torch.no_grad():
                self._scene_text_features = self._model.encode_text(text_tokens)
                self._scene_text_features /= self._scene_text_features.norm(
                    dim=-1, keepdim=True
                )
        return self._scene_text_features

    def _get_face_text_features(self):
        """Get pre-encoded face prompt features."""
        if self._face_text_features is None:
            self._ensure_loaded()
            prompts = list(FACE_PROMPTS.values())
            text_tokens = self._tokenizer(prompts).to(self._device)
            with torch.no_grad():
                self._face_text_features = self._model.encode_text(text_tokens)
                self._face_text_features /= self._face_text_features.norm(
                    dim=-1, keepdim=True
                )
        return self._face_text_features

    def analyze_scene_sentiment(
        self, image_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Analyze scene-level sentiment.

        Args:
            image_path: Path to image file.

        Returns:
            Dict with sentiment, confidence, scores, and processing_time_ms.
        """
        start_time = time.time()
        self._ensure_loaded()

        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)

            # Encode image
            with torch.no_grad():
                image_features = self._model.encode_image(image_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compute similarity with scene prompts
            text_features = self._get_scene_text_features()
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            scores_tensor = similarity[0].cpu()

            # Build scores dict
            labels = list(SCENE_PROMPTS.keys())
            scores = {label: float(scores_tensor[i]) for i, label in enumerate(labels)}

            # Get top sentiment
            top_idx = scores_tensor.argmax().item()
            sentiment = labels[top_idx]
            confidence = float(scores_tensor[top_idx])

            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "scores": scores,
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }

        except Exception as e:
            logger.error(f"Scene sentiment analysis failed: {e}")
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "scores": {},
                "error": str(e),
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }

    def analyze_face_sentiment(self, face_image: Image.Image) -> Dict[str, Any]:
        """
        Analyze face sentiment from a cropped face image.

        Args:
            face_image: PIL Image of cropped face.

        Returns:
            Dict with sentiment, confidence, scores.
        """
        start_time = time.time()
        self._ensure_loaded()

        try:
            # Preprocess
            if face_image.mode != "RGB":
                face_image = face_image.convert("RGB")
            image_tensor = self._preprocess(face_image).unsqueeze(0).to(self._device)

            # Encode image
            with torch.no_grad():
                image_features = self._model.encode_image(image_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compute similarity with face prompts
            text_features = self._get_face_text_features()
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            scores_tensor = similarity[0].cpu()

            # Build scores dict
            labels = list(FACE_PROMPTS.keys())
            scores = {label: float(scores_tensor[i]) for i, label in enumerate(labels)}

            # Get top sentiment
            top_idx = scores_tensor.argmax().item()
            sentiment = labels[top_idx]
            confidence = float(scores_tensor[top_idx])

            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "scores": scores,
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }

        except Exception as e:
            logger.error(f"Face sentiment analysis failed: {e}")
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "scores": {},
                "error": str(e),
            }

    def analyze_face_sentiment_from_image(
        self, image_path: Union[str, Path], bbox: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Analyze face sentiment by cropping from full image.

        Args:
            image_path: Path to full image.
            bbox: Bounding box with x1, y1, x2, y2.

        Returns:
            Dict with sentiment, confidence, scores.
        """
        try:
            image = Image.open(image_path).convert("RGB")

            # Crop face
            x1 = max(0, int(bbox["x1"]))
            y1 = max(0, int(bbox["y1"]))
            x2 = min(image.width, int(bbox["x2"]))
            y2 = min(image.height, int(bbox["y2"]))

            face_crop = image.crop((x1, y1, x2, y2))

            return self.analyze_face_sentiment(face_crop)

        except Exception as e:
            logger.error(f"Face sentiment from image failed: {e}")
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "scores": {},
                "error": str(e),
            }

    def analyze_faces_batch(
        self, image_path: Union[str, Path], bboxes: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple faces efficiently with batched inference.

        Args:
            image_path: Path to full image.
            bboxes: List of bounding boxes.

        Returns:
            List of sentiment results, one per bbox.
        """
        if not bboxes:
            return []

        start_time = time.time()
        self._ensure_loaded()

        try:
            image = Image.open(image_path).convert("RGB")

            # Crop all faces
            face_tensors = []
            for bbox in bboxes:
                x1 = max(0, int(bbox["x1"]))
                y1 = max(0, int(bbox["y1"]))
                x2 = min(image.width, int(bbox["x2"]))
                y2 = min(image.height, int(bbox["y2"]))
                face_crop = image.crop((x1, y1, x2, y2))
                face_tensors.append(self._preprocess(face_crop))

            # Stack into batch
            batch = torch.stack(face_tensors).to(self._device)

            # Batch inference
            with torch.no_grad():
                image_features = self._model.encode_image(batch)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compute similarities
            text_features = self._get_face_text_features()
            similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            # Parse results
            labels = list(FACE_PROMPTS.keys())
            results = []
            for i in range(len(bboxes)):
                scores_tensor = similarities[i].cpu()
                scores = {label: float(scores_tensor[j]) for j, label in enumerate(labels)}
                top_idx = scores_tensor.argmax().item()

                results.append({
                    "sentiment": labels[top_idx],
                    "confidence": float(scores_tensor[top_idx]),
                    "scores": scores,
                })

            total_time = int((time.time() - start_time) * 1000)
            logger.debug(f"Batch analyzed {len(bboxes)} faces in {total_time}ms")

            return results

        except Exception as e:
            logger.error(f"Batch face analysis failed: {e}")
            return [
                {"sentiment": "neutral", "confidence": 0.0, "scores": {}, "error": str(e)}
                for _ in bboxes
            ]
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_mobileclip_analyzer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/photodb/utils/mobileclip_analyzer.py tests/test_mobileclip_analyzer.py
git commit -m "feat: add MobileCLIPAnalyzer for zero-shot sentiment"
```

---

## Task 7: Create Scene Analysis Stage

**Files:**
- Create: `src/photodb/stages/scene_analysis.py`
- Test: `tests/test_scene_analysis_stage.py`

**Step 1: Write failing test**

```python
# tests/test_scene_analysis_stage.py
"""Tests for scene analysis stage."""
import sys
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone


class TestSceneAnalysisStage:
    """Test scene analysis stage."""

    def test_stage_name(self):
        """Stage name should be 'scene_analysis'."""
        from photodb.stages.scene_analysis import SceneAnalysisStage

        mock_repo = MagicMock()
        stage = SceneAnalysisStage(mock_repo, {"IMG_PATH": "/tmp"})

        assert stage.stage_name == "scene_analysis"

    @pytest.mark.skipif(sys.platform != "darwin", reason="Apple Vision requires macOS")
    def test_process_creates_scene_analysis(self, mock_repository, mock_photo):
        """Processing should create scene analysis record."""
        from photodb.stages.scene_analysis import SceneAnalysisStage

        with patch("photodb.stages.scene_analysis.MobileCLIPAnalyzer") as MockCLIP:
            mock_clip = MockCLIP.return_value
            mock_clip.analyze_scene_sentiment.return_value = {
                "sentiment": "peaceful",
                "confidence": 0.8,
                "scores": {"peaceful": 0.8, "joyful": 0.1},
                "processing_time_ms": 50,
            }
            mock_clip.analyze_faces_batch.return_value = []

            with patch("photodb.stages.scene_analysis.AppleVisionClassifier") as MockVision:
                mock_vision = MockVision.return_value
                mock_vision.classify.return_value = {
                    "status": "success",
                    "classifications": [
                        {"identifier": "outdoor", "confidence": 0.9},
                        {"identifier": "nature", "confidence": 0.7},
                    ],
                    "processing_time_ms": 30,
                }

                stage = SceneAnalysisStage(mock_repository, {"IMG_PATH": "/tmp"})

                from pathlib import Path
                result = stage.process_photo(mock_photo, Path("/tmp/test.jpg"))

                assert result is True
                assert mock_repository.upsert_scene_analysis.called


@pytest.fixture
def mock_repository():
    repo = MagicMock()
    repo.get_detections_for_photo.return_value = []
    return repo


@pytest.fixture
def mock_photo():
    from photodb.database.models import Photo

    return Photo(
        id=1,
        filename="test.jpg",
        normalized_path="2024/01/test.jpg",
        width=1920,
        height=1080,
        normalized_width=1920,
        normalized_height=1080,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scene_analysis_stage.py -v`
Expected: `ModuleNotFoundError`

**Step 3: Implement SceneAnalysisStage**

```python
# src/photodb/stages/scene_analysis.py
"""
Scene analysis stage: Scene taxonomy and sentiment analysis.

Uses Apple Vision (VNClassifyImageRequest) for scene taxonomy and
MobileCLIP for zero-shot sentiment analysis of scenes and faces.

This stage runs AFTER the detection stage and uses existing face
bounding boxes from person_detection table.
"""
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .base import BaseStage
from ..database.models import Photo, AnalysisOutput, SceneAnalysis
from ..utils.mobileclip_analyzer import MobileCLIPAnalyzer

logger = logging.getLogger(__name__)

# Apple Vision only available on macOS
_apple_vision_available = sys.platform == "darwin"
if _apple_vision_available:
    from ..utils.apple_vision_classifier import AppleVisionClassifier


class SceneAnalysisStage(BaseStage):
    """Stage for scene taxonomy and sentiment analysis.

    Performs:
    - Apple Vision scene classification (1303 labels) - macOS only
    - MobileCLIP scene sentiment (joyful, peaceful, somber, etc.)
    - MobileCLIP face sentiment for existing detections

    Prerequisites: Detection stage must have run first (for face bboxes).
    """

    stage_name = "scene_analysis"

    def __init__(self, repository, config: dict):
        super().__init__(repository, config)

        # Initialize analyzers
        self.mobileclip = MobileCLIPAnalyzer()

        if _apple_vision_available:
            self.apple_classifier = AppleVisionClassifier()
        else:
            self.apple_classifier = None
            logger.warning("Apple Vision not available (not macOS), skipping taxonomy")

        logger.info("SceneAnalysisStage initialized")

    def process_photo(self, photo: Photo, file_path: Path) -> bool:
        """Process scene taxonomy and sentiment for a photo."""
        try:
            # Check normalized image exists
            if not photo.normalized_path:
                logger.warning(f"No normalized path for photo {photo.id}, skipping")
                return False

            normalized_path = Path(self.config["IMG_PATH"]) / photo.normalized_path
            if not normalized_path.exists():
                logger.error(f"Normalized image not found: {normalized_path}")
                return False

            # 1. Apple Vision scene taxonomy (macOS only)
            taxonomy_output_id = None
            taxonomy_labels = None
            taxonomy_confidences = None

            if self.apple_classifier:
                taxonomy_result = self.apple_classifier.classify(
                    str(normalized_path), top_k=10
                )

                # Store raw output
                taxonomy_output = AnalysisOutput.create(
                    photo_id=photo.id,
                    model_type="classifier",
                    model_name="apple_vision_classify",
                    output=taxonomy_result,
                    processing_time_ms=taxonomy_result.get("processing_time_ms"),
                    device="ane",
                )
                self.repository.create_analysis_output(taxonomy_output)
                taxonomy_output_id = taxonomy_output.id

                if taxonomy_result["status"] == "success":
                    taxonomy_labels = [
                        c["identifier"] for c in taxonomy_result["classifications"]
                    ]
                    taxonomy_confidences = [
                        c["confidence"] for c in taxonomy_result["classifications"]
                    ]

            # 2. MobileCLIP scene sentiment
            scene_result = self.mobileclip.analyze_scene_sentiment(str(normalized_path))

            sentiment_output = AnalysisOutput.create(
                photo_id=photo.id,
                model_type="sentiment",
                model_name="mobileclip",
                output={"scene": scene_result},
                processing_time_ms=scene_result.get("processing_time_ms"),
                device="mps" if sys.platform == "darwin" else "cpu",
            )
            self.repository.create_analysis_output(sentiment_output)

            # 3. Save scene analysis
            scene_analysis = SceneAnalysis.create(
                photo_id=photo.id,
                taxonomy_labels=taxonomy_labels,
                taxonomy_confidences=taxonomy_confidences,
                scene_sentiment=scene_result["sentiment"],
                scene_sentiment_confidence=scene_result["confidence"],
                scene_sentiment_scores=scene_result.get("scores"),
                taxonomy_output_id=taxonomy_output_id,
                sentiment_output_id=sentiment_output.id,
            )
            self.repository.upsert_scene_analysis(scene_analysis)

            # 4. Face sentiment for existing detections
            detections = self.repository.get_detections_for_photo(photo.id)
            face_detections = [d for d in detections if d.has_face()]

            if face_detections:
                # Build bboxes for batch processing
                bboxes = []
                for det in face_detections:
                    bboxes.append({
                        "x1": det.face_bbox_x,
                        "y1": det.face_bbox_y,
                        "x2": det.face_bbox_x + det.face_bbox_width,
                        "y2": det.face_bbox_y + det.face_bbox_height,
                    })

                # Batch analyze faces
                face_results = self.mobileclip.analyze_faces_batch(
                    str(normalized_path), bboxes
                )

                # Store face output
                face_output = AnalysisOutput.create(
                    photo_id=photo.id,
                    model_type="sentiment",
                    model_name="mobileclip",
                    output={"faces": face_results},
                    device="mps" if sys.platform == "darwin" else "cpu",
                )
                self.repository.create_analysis_output(face_output)

                # Update each detection
                for det, result in zip(face_detections, face_results):
                    if det.id:
                        self.repository.update_detection_face_sentiment(
                            detection_id=det.id,
                            face_sentiment=result["sentiment"],
                            face_sentiment_confidence=result["confidence"],
                            face_sentiment_scores=result.get("scores"),
                            sentiment_output_id=face_output.id,
                        )

                logger.info(
                    f"Analyzed {len(face_detections)} faces for {file_path}"
                )

            logger.info(
                f"Scene analysis complete for {file_path}: "
                f"sentiment={scene_result['sentiment']}, "
                f"taxonomy={taxonomy_labels[:3] if taxonomy_labels else 'N/A'}"
            )
            return True

        except Exception as e:
            logger.error(f"Scene analysis failed for {file_path}: {e}")
            return False
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_scene_analysis_stage.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/photodb/stages/scene_analysis.py tests/test_scene_analysis_stage.py
git commit -m "feat: add SceneAnalysisStage with taxonomy and sentiment"
```

---

## Task 8: Register Stage in CLI

**Files:**
- Modify: `src/photodb/cli_local.py`

**Step 1: Import and register the stage**

Add to imports:

```python
from .stages.scene_analysis import SceneAnalysisStage
```

Add to STAGES dict or stage selection:

```python
STAGES["scene_analysis"] = SceneAnalysisStage
```

**Step 2: Update stage choices**

Update the `--stage` option:

```python
@click.option(
    "--stage",
    type=click.Choice([
        "normalize", "metadata", "detection", "age_gender", "scene_analysis"
    ]),
    help="Run only a specific stage",
)
```

**Step 3: Verify CLI**

Run: `uv run process-local --help`
Expected: Shows `scene_analysis` in stage choices

**Step 4: Commit**

```bash
git add src/photodb/cli_local.py
git commit -m "cli: register SceneAnalysisStage in process-local"
```

---

## Task 9: Update Documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add scene analysis configuration**

Add after the Face Embedding Configuration section:

```markdown
### Scene Analysis Stage Configuration

The scene analysis stage provides scene taxonomy and sentiment analysis.

**Capabilities:**
- **Apple Vision taxonomy** (macOS only): 1303 scene classification labels
- **MobileCLIP scene sentiment**: joyful, peaceful, somber, tense, neutral, energetic
- **MobileCLIP face sentiment**: happy, sad, angry, surprised, fearful, disgusted, neutral

**Usage:**
```bash
# Run scene analysis (requires detection stage to have run first)
uv run process-local /path/to/photos --stage scene_analysis

# Full pipeline including scene analysis
uv run process-local /path/to/photos  # Runs all stages in order
```

**Model Download:**
MobileCLIP-S2 model (~100MB) downloads automatically on first use from HuggingFace.

**Hardware:**
- MobileCLIP uses MPS (Metal) on macOS, CUDA on Linux/Windows with GPU, or CPU fallback
- Apple Vision uses Neural Engine automatically (macOS only)

**Note:** Scene analysis runs AFTER detection stage. It uses existing face bounding
boxes from `person_detection` table to analyze face sentiments.
```

**Step 2: Update pipeline documentation**

Update the Processing Pipeline section to include:

```markdown
5. **Scene Analysis**: Scene taxonomy (Apple Vision) and sentiment (MobileCLIP)
```

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add scene analysis stage documentation"
```

---

## Task 10: Integration Test

**Files:**
- Create: `tests/test_integration_scene_analysis.py`

**Step 1: Write integration test**

```python
# tests/test_integration_scene_analysis.py
"""Integration tests for scene analysis pipeline."""
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Full integration requires macOS for Apple Vision"
)
class TestSceneAnalysisIntegration:
    """Integration tests for scene analysis."""

    def test_full_pipeline_flow(self, test_image_path):
        """Test the full scene analysis flow."""
        from photodb.utils.apple_vision_classifier import AppleVisionClassifier
        from photodb.utils.mobileclip_analyzer import MobileCLIPAnalyzer

        # Test Apple Vision classification
        classifier = AppleVisionClassifier()
        taxonomy = classifier.classify(str(test_image_path))

        assert taxonomy["status"] == "success"
        assert len(taxonomy["classifications"]) > 0

        # Test MobileCLIP scene sentiment
        analyzer = MobileCLIPAnalyzer()
        scene_sentiment = analyzer.analyze_scene_sentiment(str(test_image_path))

        assert scene_sentiment["sentiment"] in (
            "joyful", "peaceful", "somber", "tense", "neutral", "energetic"
        )
        assert scene_sentiment["confidence"] > 0

        # Test MobileCLIP face sentiment (with mock bbox)
        bbox = {"x1": 100, "y1": 100, "x2": 300, "y2": 300}
        face_sentiment = analyzer.analyze_face_sentiment_from_image(
            str(test_image_path), bbox
        )

        assert face_sentiment["sentiment"] in (
            "happy", "sad", "angry", "surprised", "fearful", "disgusted", "neutral"
        )


class TestMobileCLIPOnly:
    """Tests that work without macOS."""

    def test_mobileclip_loads(self):
        """MobileCLIP should load successfully."""
        from photodb.utils.mobileclip_analyzer import MobileCLIPAnalyzer

        analyzer = MobileCLIPAnalyzer()
        # Trigger lazy load
        analyzer._ensure_loaded()

        assert analyzer._model is not None

    def test_scene_prompts_defined(self):
        """Scene sentiment prompts should be defined."""
        from photodb.utils.mobileclip_analyzer import SCENE_PROMPTS, FACE_PROMPTS

        assert len(SCENE_PROMPTS) == 6
        assert len(FACE_PROMPTS) == 7
        assert "joyful" in SCENE_PROMPTS
        assert "happy" in FACE_PROMPTS


@pytest.fixture
def test_image_path():
    return Path(__file__).parent.parent / "test_photos" / "test.jpg"
```

**Step 2: Run integration tests**

Run: `uv run pytest tests/test_integration_scene_analysis.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_integration_scene_analysis.py
git commit -m "test: add integration tests for scene analysis"
```

---

## Summary

This plan adds scene taxonomy and sentiment analysis to the existing pipeline:

| New Component | Purpose | Platform |
|---------------|---------|----------|
| `AppleVisionClassifier` | Scene taxonomy (1303 labels) | macOS only |
| `MobileCLIPAnalyzer` | Scene + face sentiment | All platforms |
| `SceneAnalysisStage` | Orchestrates both | All platforms |

**Database additions:**
- `analysis_output` - Raw model outputs (generalizable)
- `scene_analysis` - Photo-level taxonomy + sentiment
- `person_detection.face_sentiment*` - Per-face sentiment columns

**Pipeline flow:**
```
normalize → metadata → detection → age_gender → scene_analysis
                          ↓              ↓
                     (face bboxes)  (age/gender)
                          ↓
                    scene_analysis
                          ↓
            (taxonomy + scene sentiment + face sentiment)
```

**Existing components unchanged:**
- YOLO detection
- MiVOLO age/gender
- InsightFace embeddings

---

Plan complete and saved to `docs/plans/2026-02-02-scene-sentiment-analysis.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
