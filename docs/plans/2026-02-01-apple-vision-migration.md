# Apple Vision Framework Migration Implementation Plan

NOTE: This plan was archived and not implemented, in favor of (2026-02-02-scene-sentiment-analysis.md).

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate face/body detection from YOLO+FaceNet to Apple Vision Framework, adding sentiment analysis for scenes and detected people, with a generalizable schema supporting multiple detection models/pipelines.

**Architecture:** Replace `PersonDetector` (YOLO+FaceNet) with `AppleVisionDetector` using PyObjC bindings to macOS Vision framework. Add new `analysis_output` table for model-agnostic results storage. Keep existing `person_detection` table for detection-specific fields while extending it to reference analysis outputs. Face embeddings remain in `face_embedding` table but dimension becomes configurable via a new `embedding_model` registry.

**Tech Stack:** PyObjC (pyobjc-framework-Vision, pyobjc-framework-Quartz), Apple Vision Framework APIs (VNDetectFaceRectanglesRequest, VNDetectHumanRectanglesRequest, VNDetectFaceLandmarksRequest, VNClassifyImageRequest), PostgreSQL with JSONB.

---

## Overview

### What Apple Vision Provides

| Request | Purpose | Output |
|---------|---------|--------|
| `VNDetectFaceRectanglesRequest` | Face bounding boxes | `VNFaceObservation` with `boundingBox` |
| `VNDetectHumanRectanglesRequest` | Body bounding boxes | `VNHumanObservation` with `boundingBox` |
| `VNDetectFaceLandmarksRequest` | 76 facial landmarks | `VNFaceLandmarks2D` with eyes, nose, mouth, etc. |
| `VNClassifyImageRequest` | Scene classification (1303 classes) | `VNClassificationObservation` with identifiers/confidence |

### Sentiment Analysis Strategy

Apple Vision doesn't provide direct emotion/sentiment APIs. We'll implement sentiment analysis by:

1. **Scene Sentiment**: Use `VNClassifyImageRequest` to classify scene type, then map to sentiment categories
2. **Face Sentiment**: Use `VNDetectFaceLandmarksRequest` to extract 76 facial landmarks, then apply heuristics based on landmark positions (mouth curvature, eye openness, eyebrow position)

### Schema Generalization Strategy

Current schema tightly couples `person_detection` to YOLO+FaceNet. New design:

1. **`analysis_output`** - Model-agnostic raw output storage (JSONB)
2. **`detector_registry`** - Track available detectors and their capabilities
3. **`embedding_config`** - Track embedding dimensions per model
4. **Extend `person_detection`** - Add `analysis_output_id` FK for linking

---

## Task 1: Add PyObjC Dependencies

**Files:**

- Modify: `pyproject.toml:6-28`

**Step 1: Add pyobjc dependencies**

```toml
# In dependencies list, add:
    "pyobjc-framework-Vision>=10.0",
    "pyobjc-framework-Quartz>=10.0",
```

**Step 2: Run dependency sync**

Run: `uv sync`
Expected: Successfully installs pyobjc packages

**Step 3: Verify import works**

Run: `uv run python -c "import Vision; import Quartz; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add pyobjc-framework-Vision and Quartz for Apple Vision"
```

---

## Task 2: Create Database Migration for Generalizable Schema

**Files:**

- Create: `migrations/006_add_analysis_output.sql`

**Step 1: Write migration SQL**

```sql
-- Model-agnostic analysis output storage
-- Stores raw model outputs in JSONB for any detector/analyzer
CREATE TABLE IF NOT EXISTS analysis_output (
    id bigserial PRIMARY KEY,
    photo_id bigint NOT NULL,

    -- Model identification
    model_type text NOT NULL,  -- 'detector', 'classifier', 'embedder', 'sentiment'
    model_name text NOT NULL,  -- 'apple_vision', 'yolo_facenet', 'mivolo', etc.
    model_version text,

    -- Raw output (schema varies by model)
    output jsonb NOT NULL,

    -- Processing metadata
    processing_time_ms integer,
    device text,  -- 'cpu', 'gpu', 'ane' (Apple Neural Engine)

    created_at timestamptz DEFAULT now(),

    FOREIGN KEY (photo_id) REFERENCES photo(id) ON DELETE CASCADE
);

-- Index for querying by photo and model
CREATE INDEX IF NOT EXISTS idx_analysis_output_photo_model
    ON analysis_output(photo_id, model_type, model_name);

-- GIN index for JSONB queries
CREATE INDEX IF NOT EXISTS idx_analysis_output_output
    ON analysis_output USING GIN(output);

-- Detector registry: track available detection models
CREATE TABLE IF NOT EXISTS detector_registry (
    id bigserial PRIMARY KEY,
    name text UNIQUE NOT NULL,  -- 'apple_vision', 'yolo_facenet', 'mivolo'
    display_name text NOT NULL,
    capabilities text[] NOT NULL,  -- ['face_detection', 'body_detection', 'landmarks', 'sentiment']
    embedding_dimension integer,  -- NULL if doesn't produce embeddings
    is_active boolean DEFAULT true,
    config jsonb,  -- Model-specific configuration
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

-- Insert known detectors
INSERT INTO detector_registry (name, display_name, capabilities, embedding_dimension, config) VALUES
    ('yolo_facenet', 'YOLO + FaceNet', ARRAY['face_detection', 'body_detection', 'face_embedding'], 512, '{"yolo_model": "yolov8x_person_face"}'),
    ('apple_vision', 'Apple Vision Framework', ARRAY['face_detection', 'body_detection', 'face_landmarks', 'scene_classification'], NULL, '{}'),
    ('mivolo', 'MiVOLO', ARRAY['age_estimation', 'gender_estimation'], NULL, '{}')
ON CONFLICT (name) DO NOTHING;

-- Extend person_detection with analysis_output reference
ALTER TABLE person_detection
    ADD COLUMN IF NOT EXISTS analysis_output_id bigint REFERENCES analysis_output(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_person_detection_analysis_output
    ON person_detection(analysis_output_id);

-- Scene analysis table for photo-level classifications
CREATE TABLE IF NOT EXISTS scene_analysis (
    id bigserial PRIMARY KEY,
    photo_id bigint NOT NULL UNIQUE,
    analysis_output_id bigint REFERENCES analysis_output(id) ON DELETE SET NULL,

    -- Top classifications (extracted from output for indexing)
    top_labels text[],  -- Top 5 scene labels
    top_confidences real[],  -- Corresponding confidences

    -- Derived sentiment
    scene_sentiment text CHECK (scene_sentiment IN ('positive', 'negative', 'neutral', 'mixed')),
    scene_sentiment_confidence real,

    created_at timestamptz DEFAULT now(),

    FOREIGN KEY (photo_id) REFERENCES photo(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_scene_analysis_labels ON scene_analysis USING GIN(top_labels);
CREATE INDEX IF NOT EXISTS idx_scene_analysis_sentiment ON scene_analysis(scene_sentiment);

-- Face sentiment (linked to person_detection)
ALTER TABLE person_detection
    ADD COLUMN IF NOT EXISTS face_sentiment text CHECK (face_sentiment IN ('happy', 'sad', 'angry', 'surprised', 'neutral', 'fearful', 'disgusted')),
    ADD COLUMN IF NOT EXISTS face_sentiment_confidence real,
    ADD COLUMN IF NOT EXISTS landmarks_output_id bigint REFERENCES analysis_output(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_person_detection_sentiment ON person_detection(face_sentiment);
```

**Step 2: Apply migration**

Run: `psql $DATABASE_URL -f migrations/006_add_analysis_output.sql`
Expected: Tables and columns created successfully

**Step 3: Verify schema**

Run: `psql $DATABASE_URL -c "\d analysis_output" && psql $DATABASE_URL -c "\d detector_registry"`
Expected: Tables show expected columns

**Step 4: Commit**

```bash
git add migrations/006_add_analysis_output.sql
git commit -m "db: add generalizable analysis_output schema for multi-model support"
```

---

## Task 3: Add New Database Models

**Files:**

- Modify: `src/photodb/database/models.py`

**Step 1: Add AnalysisOutput dataclass**

Add after `PersonDetection` class:

```python
@dataclass
class AnalysisOutput:
    """Model-agnostic storage for raw analysis outputs."""

    id: Optional[int]
    photo_id: int
    model_type: str  # 'detector', 'classifier', 'embedder', 'sentiment'
    model_name: str  # 'apple_vision', 'yolo_facenet', etc.
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
    analysis_output_id: Optional[int]
    top_labels: Optional[List[str]]
    top_confidences: Optional[List[float]]
    scene_sentiment: Optional[str]  # 'positive', 'negative', 'neutral', 'mixed'
    scene_sentiment_confidence: Optional[float]
    created_at: Optional[datetime]

    @classmethod
    def create(
        cls,
        photo_id: int,
        top_labels: Optional[List[str]] = None,
        top_confidences: Optional[List[float]] = None,
        scene_sentiment: Optional[str] = None,
        scene_sentiment_confidence: Optional[float] = None,
        analysis_output_id: Optional[int] = None,
    ) -> "SceneAnalysis":
        return cls(
            id=None,
            photo_id=photo_id,
            analysis_output_id=analysis_output_id,
            top_labels=top_labels,
            top_confidences=top_confidences,
            scene_sentiment=scene_sentiment,
            scene_sentiment_confidence=scene_sentiment_confidence,
            created_at=datetime.now(timezone.utc),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "photo_id": self.photo_id,
            "analysis_output_id": self.analysis_output_id,
            "top_labels": self.top_labels,
            "top_confidences": self.top_confidences,
            "scene_sentiment": self.scene_sentiment,
            "scene_sentiment_confidence": self.scene_sentiment_confidence,
            "created_at": self.created_at,
        }


@dataclass
class DetectorRegistry:
    """Registry of available detection models."""

    id: Optional[int]
    name: str
    display_name: str
    capabilities: List[str]
    embedding_dimension: Optional[int]
    is_active: bool
    config: Optional[Dict[str, Any]]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    def has_capability(self, capability: str) -> bool:
        return capability in self.capabilities

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "capabilities": self.capabilities,
            "embedding_dimension": self.embedding_dimension,
            "is_active": self.is_active,
            "config": self.config,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
```

**Step 2: Update PersonDetection with new fields**

Add to `PersonDetection` dataclass after `detector_version`:

```python
    # Analysis output reference (for raw model data)
    analysis_output_id: Optional[int] = None
    # Face sentiment from landmarks analysis
    face_sentiment: Optional[str] = None  # 'happy', 'sad', 'angry', etc.
    face_sentiment_confidence: Optional[float] = None
    landmarks_output_id: Optional[int] = None
```

**Step 3: Commit**

```bash
git add src/photodb/database/models.py
git commit -m "models: add AnalysisOutput, SceneAnalysis, DetectorRegistry"
```

---

## Task 4: Add Repository Methods for New Tables

**Files:**

- Modify: `src/photodb/database/pg_repository.py`

**Step 1: Add AnalysisOutput repository methods**

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


def get_analysis_outputs_for_photo(
    self, photo_id: int, model_type: Optional[str] = None, model_name: Optional[str] = None
) -> List[AnalysisOutput]:
    """Get analysis outputs for a photo, optionally filtered by model."""
    with self.pool.connection() as conn:
        with conn.cursor() as cur:
            query = "SELECT * FROM analysis_output WHERE photo_id = %s"
            params = [photo_id]

            if model_type:
                query += " AND model_type = %s"
                params.append(model_type)
            if model_name:
                query += " AND model_name = %s"
                params.append(model_name)

            query += " ORDER BY created_at DESC"
            cur.execute(query, params)

            rows = cur.fetchall()
            return [self._row_to_analysis_output(row) for row in rows]


def _row_to_analysis_output(self, row) -> AnalysisOutput:
    """Convert database row to AnalysisOutput."""
    return AnalysisOutput(
        id=row[0],
        photo_id=row[1],
        model_type=row[2],
        model_name=row[3],
        model_version=row[4],
        output=row[5] if isinstance(row[5], dict) else json.loads(row[5]),
        processing_time_ms=row[6],
        device=row[7],
        created_at=row[8],
    )
```

**Step 2: Add SceneAnalysis repository methods**

```python
def create_scene_analysis(self, analysis: SceneAnalysis) -> SceneAnalysis:
    """Insert or update scene analysis record."""
    with self.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO scene_analysis
                (photo_id, analysis_output_id, top_labels, top_confidences,
                 scene_sentiment, scene_sentiment_confidence, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (photo_id) DO UPDATE SET
                    analysis_output_id = EXCLUDED.analysis_output_id,
                    top_labels = EXCLUDED.top_labels,
                    top_confidences = EXCLUDED.top_confidences,
                    scene_sentiment = EXCLUDED.scene_sentiment,
                    scene_sentiment_confidence = EXCLUDED.scene_sentiment_confidence,
                    created_at = EXCLUDED.created_at
                RETURNING id
                """,
                (
                    analysis.photo_id,
                    analysis.analysis_output_id,
                    analysis.top_labels,
                    analysis.top_confidences,
                    analysis.scene_sentiment,
                    analysis.scene_sentiment_confidence,
                    analysis.created_at,
                ),
            )
            analysis.id = cur.fetchone()[0]
            conn.commit()
    return analysis


def get_scene_analysis(self, photo_id: int) -> Optional[SceneAnalysis]:
    """Get scene analysis for a photo."""
    with self.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM scene_analysis WHERE photo_id = %s", (photo_id,))
            row = cur.fetchone()
            if row:
                return SceneAnalysis(
                    id=row[0],
                    photo_id=row[1],
                    analysis_output_id=row[2],
                    top_labels=row[3],
                    top_confidences=row[4],
                    scene_sentiment=row[5],
                    scene_sentiment_confidence=row[6],
                    created_at=row[7],
                )
            return None
```

**Step 3: Add detector update method for sentiment**

```python
def update_detection_sentiment(
    self,
    detection_id: int,
    face_sentiment: str,
    face_sentiment_confidence: float,
    landmarks_output_id: Optional[int] = None,
) -> None:
    """Update face sentiment for a detection."""
    with self.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE person_detection
                SET face_sentiment = %s,
                    face_sentiment_confidence = %s,
                    landmarks_output_id = %s
                WHERE id = %s
                """,
                (face_sentiment, face_sentiment_confidence, landmarks_output_id, detection_id),
            )
            conn.commit()
```

**Step 4: Commit**

```bash
git add src/photodb/database/pg_repository.py
git commit -m "repo: add methods for AnalysisOutput and SceneAnalysis"
```

---

## Task 5: Create Apple Vision Detector Utility

**Files:**

- Create: `src/photodb/utils/apple_vision_detector.py`
- Test: `tests/test_apple_vision_detector.py`

**Step 1: Write failing test**

```python
# tests/test_apple_vision_detector.py
"""Tests for Apple Vision detector."""
import sys
import pytest
from pathlib import Path

# Skip all tests if not on macOS
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="Apple Vision only available on macOS"
)


class TestAppleVisionDetector:
    """Test Apple Vision Framework detector."""

    def test_detect_returns_dict_with_status(self, test_image_path):
        """Detection result should have status field."""
        from photodb.utils.apple_vision_detector import AppleVisionDetector

        detector = AppleVisionDetector()
        result = detector.detect(str(test_image_path))

        assert "status" in result
        assert result["status"] in ("success", "no_detections", "error")

    def test_detect_returns_face_detections(self, test_image_with_face):
        """Should detect faces in image with face."""
        from photodb.utils.apple_vision_detector import AppleVisionDetector

        detector = AppleVisionDetector()
        result = detector.detect(str(test_image_with_face))

        assert result["status"] == "success"
        assert len(result["detections"]) > 0

        # Check face detection structure
        detection = result["detections"][0]
        assert "face" in detection
        if detection["face"]:
            bbox = detection["face"]["bbox"]
            assert "x1" in bbox
            assert "y1" in bbox
            assert "x2" in bbox
            assert "y2" in bbox

    def test_classify_scene_returns_labels(self, test_image_path):
        """Scene classification should return labels with confidence."""
        from photodb.utils.apple_vision_detector import AppleVisionDetector

        detector = AppleVisionDetector()
        result = detector.classify_scene(str(test_image_path))

        assert "status" in result
        if result["status"] == "success":
            assert "classifications" in result
            assert len(result["classifications"]) > 0

            classification = result["classifications"][0]
            assert "identifier" in classification
            assert "confidence" in classification

    def test_detect_landmarks_returns_points(self, test_image_with_face):
        """Landmark detection should return facial points."""
        from photodb.utils.apple_vision_detector import AppleVisionDetector

        detector = AppleVisionDetector()
        result = detector.detect_landmarks(str(test_image_with_face))

        assert "status" in result
        if result["status"] == "success":
            assert "faces" in result
            if result["faces"]:
                face = result["faces"][0]
                assert "landmarks" in face
                # Vision framework returns 76 landmark points
                assert "leftEye" in face["landmarks"] or "all_points" in face["landmarks"]


@pytest.fixture
def test_image_path():
    """Path to a test image."""
    return Path(__file__).parent.parent / "test_photos" / "test.jpg"


@pytest.fixture
def test_image_with_face():
    """Path to test image containing a face."""
    # Use same test image - adjust if you have a specific face test image
    return Path(__file__).parent.parent / "test_photos" / "test.jpg"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_apple_vision_detector.py -v`
Expected: `ModuleNotFoundError: No module named 'photodb.utils.apple_vision_detector'`

**Step 3: Implement AppleVisionDetector**

```python
# src/photodb/utils/apple_vision_detector.py
"""
Apple Vision Framework detector for face/body detection and scene classification.

Uses PyObjC bindings to access macOS Vision APIs. Only works on macOS.
"""
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.platform != "darwin":
    raise ImportError("Apple Vision Framework only available on macOS")

import Quartz
import Vision
from Foundation import NSURL

logger = logging.getLogger(__name__)


class AppleVisionDetector:
    """Detect faces, bodies, and classify scenes using Apple Vision Framework.

    Uses Neural Engine when available for fast inference.
    Thread-safe - Vision framework handles concurrency internally.
    """

    def __init__(self):
        """Initialize Apple Vision detector."""
        logger.info("AppleVisionDetector initialized (using Apple Vision Framework)")

    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Detect faces and bodies in an image.

        Args:
            image_path: Path to the image file.

        Returns:
            Dictionary with:
                - status: 'success', 'no_detections', or 'error'
                - detections: List of person detections with face/body info
                - image_dimensions: Dict with width and height
                - error: Error message (only if status is 'error')
                - processing_time_ms: Time taken for detection
        """
        start_time = time.time()

        try:
            # Load image
            image_url = NSURL.fileURLWithPath_(image_path)
            ci_image = Quartz.CIImage.imageWithContentsOfURL_(image_url)

            if ci_image is None:
                return {
                    "status": "error",
                    "detections": [],
                    "image_dimensions": {"width": 0, "height": 0},
                    "error": f"Failed to load image: {image_path}",
                }

            # Get image dimensions
            extent = ci_image.extent()
            img_width = int(extent.size.width)
            img_height = int(extent.size.height)

            # Create request handler
            handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
                ci_image, None
            )

            # Create face and body detection requests
            face_request = Vision.VNDetectFaceRectanglesRequest.alloc().init()
            body_request = Vision.VNDetectHumanRectanglesRequest.alloc().init()

            # Perform requests
            success, error = handler.performRequests_error_([face_request, body_request], None)

            if not success:
                return {
                    "status": "error",
                    "detections": [],
                    "image_dimensions": {"width": img_width, "height": img_height},
                    "error": str(error) if error else "Unknown Vision error",
                }

            # Parse face results
            faces = []
            face_results = face_request.results() or []
            for observation in face_results:
                bbox = observation.boundingBox()
                # Vision uses normalized coordinates (0-1), origin at bottom-left
                # Convert to pixel coordinates with origin at top-left
                faces.append({
                    "bbox": self._convert_bbox(bbox, img_width, img_height),
                    "confidence": float(observation.confidence()),
                })

            # Parse body results
            bodies = []
            body_results = body_request.results() or []
            for observation in body_results:
                bbox = observation.boundingBox()
                bodies.append({
                    "bbox": self._convert_bbox(bbox, img_width, img_height),
                    "confidence": float(observation.confidence()),
                })

            # Match faces to bodies
            if not faces and not bodies:
                return {
                    "status": "no_detections",
                    "detections": [],
                    "image_dimensions": {"width": img_width, "height": img_height},
                    "processing_time_ms": int((time.time() - start_time) * 1000),
                }

            matched_detections = self._match_faces_to_bodies(faces, bodies)

            return {
                "status": "success",
                "detections": matched_detections,
                "image_dimensions": {"width": img_width, "height": img_height},
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }

        except Exception as e:
            logger.error(f"Apple Vision detection failed: {e}")
            return {
                "status": "error",
                "detections": [],
                "image_dimensions": {"width": 0, "height": 0},
                "error": str(e),
            }

    def classify_scene(self, image_path: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Classify scene content in an image.

        Args:
            image_path: Path to the image file.
            top_k: Number of top classifications to return.

        Returns:
            Dictionary with:
                - status: 'success' or 'error'
                - classifications: List of {identifier, confidence}
                - processing_time_ms: Time taken
        """
        start_time = time.time()

        try:
            image_url = NSURL.fileURLWithPath_(image_path)
            ci_image = Quartz.CIImage.imageWithContentsOfURL_(image_url)

            if ci_image is None:
                return {
                    "status": "error",
                    "classifications": [],
                    "error": f"Failed to load image: {image_path}",
                }

            handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
                ci_image, None
            )

            classify_request = Vision.VNClassifyImageRequest.alloc().init()
            success, error = handler.performRequests_error_([classify_request], None)

            if not success:
                return {
                    "status": "error",
                    "classifications": [],
                    "error": str(error) if error else "Classification failed",
                }

            results = classify_request.results() or []

            # Sort by confidence and take top_k
            classifications = []
            for observation in results:
                classifications.append({
                    "identifier": str(observation.identifier()),
                    "confidence": float(observation.confidence()),
                })

            # Sort by confidence descending
            classifications.sort(key=lambda x: x["confidence"], reverse=True)
            classifications = classifications[:top_k]

            return {
                "status": "success",
                "classifications": classifications,
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }

        except Exception as e:
            logger.error(f"Scene classification failed: {e}")
            return {
                "status": "error",
                "classifications": [],
                "error": str(e),
            }

    def detect_landmarks(self, image_path: str) -> Dict[str, Any]:
        """
        Detect facial landmarks in an image.

        Args:
            image_path: Path to the image file.

        Returns:
            Dictionary with:
                - status: 'success', 'no_faces', or 'error'
                - faces: List of face landmark data
                - processing_time_ms: Time taken
        """
        start_time = time.time()

        try:
            image_url = NSURL.fileURLWithPath_(image_path)
            ci_image = Quartz.CIImage.imageWithContentsOfURL_(image_url)

            if ci_image is None:
                return {
                    "status": "error",
                    "faces": [],
                    "error": f"Failed to load image: {image_path}",
                }

            extent = ci_image.extent()
            img_width = int(extent.size.width)
            img_height = int(extent.size.height)

            handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
                ci_image, None
            )

            landmarks_request = Vision.VNDetectFaceLandmarksRequest.alloc().init()
            success, error = handler.performRequests_error_([landmarks_request], None)

            if not success:
                return {
                    "status": "error",
                    "faces": [],
                    "error": str(error) if error else "Landmarks detection failed",
                }

            results = landmarks_request.results() or []

            if not results:
                return {
                    "status": "no_faces",
                    "faces": [],
                    "processing_time_ms": int((time.time() - start_time) * 1000),
                }

            faces = []
            for observation in results:
                bbox = observation.boundingBox()
                face_data = {
                    "bbox": self._convert_bbox(bbox, img_width, img_height),
                    "confidence": float(observation.confidence()),
                    "landmarks": {},
                }

                landmarks = observation.landmarks()
                if landmarks:
                    face_data["landmarks"] = self._extract_landmarks(
                        landmarks, bbox, img_width, img_height
                    )

                faces.append(face_data)

            return {
                "status": "success",
                "faces": faces,
                "image_dimensions": {"width": img_width, "height": img_height},
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }

        except Exception as e:
            logger.error(f"Landmarks detection failed: {e}")
            return {
                "status": "error",
                "faces": [],
                "error": str(e),
            }

    def _convert_bbox(
        self, vision_bbox, img_width: int, img_height: int
    ) -> Dict[str, float]:
        """
        Convert Vision normalized bbox to pixel coordinates.

        Vision uses normalized coordinates (0-1) with origin at bottom-left.
        We convert to pixel coordinates with origin at top-left (standard image coords).
        """
        x = float(vision_bbox.origin.x) * img_width
        y = (1.0 - float(vision_bbox.origin.y) - float(vision_bbox.size.height)) * img_height
        width = float(vision_bbox.size.width) * img_width
        height = float(vision_bbox.size.height) * img_height

        return {
            "x1": x,
            "y1": y,
            "x2": x + width,
            "y2": y + height,
            "width": width,
            "height": height,
        }

    def _extract_landmarks(
        self, landmarks, face_bbox, img_width: int, img_height: int
    ) -> Dict[str, List[Dict[str, float]]]:
        """Extract landmark points from VNFaceLandmarks2D."""
        result = {}

        # Map landmark regions to their accessors
        landmark_regions = [
            ("leftEye", landmarks.leftEye),
            ("rightEye", landmarks.rightEye),
            ("leftEyebrow", landmarks.leftEyebrow),
            ("rightEyebrow", landmarks.rightEyebrow),
            ("nose", landmarks.nose),
            ("noseCrest", landmarks.noseCrest),
            ("medianLine", landmarks.medianLine),
            ("outerLips", landmarks.outerLips),
            ("innerLips", landmarks.innerLips),
            ("leftPupil", landmarks.leftPupil),
            ("rightPupil", landmarks.rightPupil),
            ("faceContour", landmarks.faceContour),
        ]

        for name, region_func in landmark_regions:
            try:
                region = region_func()
                if region and region.pointCount() > 0:
                    points = []
                    # Get normalized points (relative to face bbox)
                    normalized_points = region.normalizedPoints()
                    for i in range(region.pointCount()):
                        # Convert from face-relative to image coordinates
                        nx = normalized_points[i].x
                        ny = normalized_points[i].y

                        # Scale to face bbox, then to image
                        fx = float(face_bbox.origin.x) + nx * float(face_bbox.size.width)
                        fy = float(face_bbox.origin.y) + ny * float(face_bbox.size.height)

                        # Convert to pixel coords (flip y)
                        px = fx * img_width
                        py = (1.0 - fy) * img_height

                        points.append({"x": px, "y": py})

                    result[name] = points
            except Exception:
                # Some landmarks may not be available
                continue

        return result

    def _match_faces_to_bodies(
        self, faces: List[Dict], bodies: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Match faces to containing bodies based on spatial overlap."""
        matched = []
        used_bodies = set()

        for face in faces:
            best_body = None
            best_containment = 0.0

            for i, body in enumerate(bodies):
                if i in used_bodies:
                    continue

                containment = self._compute_containment(face["bbox"], body["bbox"])
                if containment > best_containment:
                    best_containment = containment
                    best_body = (i, body)

            if best_body and best_containment > 0.3:
                used_bodies.add(best_body[0])
                matched.append({"face": face, "body": best_body[1]})
            else:
                matched.append({"face": face, "body": None})

        # Add unmatched bodies
        for i, body in enumerate(bodies):
            if i not in used_bodies:
                matched.append({"face": None, "body": body})

        return matched

    def _compute_containment(self, face_bbox: Dict, body_bbox: Dict) -> float:
        """Compute how much of face is contained within body."""
        x1 = max(face_bbox["x1"], body_bbox["x1"])
        y1 = max(face_bbox["y1"], body_bbox["y1"])
        x2 = min(face_bbox["x2"], body_bbox["x2"])
        y2 = min(face_bbox["y2"], body_bbox["y2"])

        if x1 >= x2 or y1 >= y2:
            return 0.0

        intersection_area = (x2 - x1) * (y2 - y1)
        face_area = face_bbox["width"] * face_bbox["height"]

        if face_area == 0:
            return 0.0

        return intersection_area / face_area
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_apple_vision_detector.py -v`
Expected: Tests pass (or skip if not on macOS)

**Step 5: Commit**

```bash
git add src/photodb/utils/apple_vision_detector.py tests/test_apple_vision_detector.py
git commit -m "feat: add AppleVisionDetector using macOS Vision Framework"
```

---

## Task 6: Create Sentiment Analyzer

**Files:**

- Create: `src/photodb/utils/sentiment_analyzer.py`
- Test: `tests/test_sentiment_analyzer.py`

**Step 1: Write failing test**

```python
# tests/test_sentiment_analyzer.py
"""Tests for sentiment analyzer."""
import pytest


class TestSceneSentiment:
    """Test scene sentiment classification."""

    def test_positive_scene_labels(self):
        """Positive labels should map to positive sentiment."""
        from photodb.utils.sentiment_analyzer import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        result = analyzer.classify_scene_sentiment(
            [("celebration", 0.9), ("party", 0.8), ("happiness", 0.7)]
        )

        assert result["sentiment"] == "positive"
        assert result["confidence"] > 0.5

    def test_negative_scene_labels(self):
        """Negative labels should map to negative sentiment."""
        from photodb.utils.sentiment_analyzer import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        result = analyzer.classify_scene_sentiment(
            [("funeral", 0.9), ("sadness", 0.8), ("grief", 0.7)]
        )

        assert result["sentiment"] == "negative"
        assert result["confidence"] > 0.5

    def test_neutral_scene_labels(self):
        """Neutral labels should map to neutral sentiment."""
        from photodb.utils.sentiment_analyzer import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        result = analyzer.classify_scene_sentiment(
            [("office", 0.9), ("desk", 0.8), ("computer", 0.7)]
        )

        assert result["sentiment"] == "neutral"


class TestFaceSentiment:
    """Test face sentiment from landmarks."""

    def test_happy_face_landmarks(self):
        """Upward mouth curve should indicate happiness."""
        from photodb.utils.sentiment_analyzer import SentimentAnalyzer

        analyzer = SentimentAnalyzer()

        # Mock landmarks with upward mouth curve
        landmarks = {
            "outerLips": [
                {"x": 100, "y": 200},  # left corner
                {"x": 120, "y": 190},  # left upper
                {"x": 140, "y": 185},  # center upper (smile curves up)
                {"x": 160, "y": 190},  # right upper
                {"x": 180, "y": 200},  # right corner
                {"x": 160, "y": 210},  # right lower
                {"x": 140, "y": 205},  # center lower
                {"x": 120, "y": 210},  # left lower
            ],
            "leftEye": [{"x": 110, "y": 150}],
            "rightEye": [{"x": 170, "y": 150}],
        }

        result = analyzer.classify_face_sentiment(landmarks)

        assert result["sentiment"] in ("happy", "neutral")
        assert "confidence" in result

    def test_empty_landmarks_returns_neutral(self):
        """Empty landmarks should return neutral."""
        from photodb.utils.sentiment_analyzer import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        result = analyzer.classify_face_sentiment({})

        assert result["sentiment"] == "neutral"
        assert result["confidence"] < 0.5
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sentiment_analyzer.py -v`
Expected: `ModuleNotFoundError: No module named 'photodb.utils.sentiment_analyzer'`

**Step 3: Implement SentimentAnalyzer**

```python
# src/photodb/utils/sentiment_analyzer.py
"""
Sentiment analysis from scene classifications and facial landmarks.

Scene sentiment: Maps VNClassifyImageRequest labels to sentiment categories.
Face sentiment: Analyzes facial landmark geometry to infer emotion.
"""
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Scene labels mapped to sentiment (partial list - Vision has 1303 classes)
POSITIVE_SCENE_LABELS = {
    "celebration", "party", "wedding", "birthday", "happiness", "joy",
    "smile", "laughing", "fun", "vacation", "beach", "sunset", "sunrise",
    "family", "friends", "love", "romance", "graduation", "achievement",
    "success", "victory", "christmas", "holiday", "festival", "carnival",
    "playground", "amusement_park", "fireworks", "rainbow", "flower",
    "garden", "park", "nature", "mountain", "lake", "ocean", "tropical",
}

NEGATIVE_SCENE_LABELS = {
    "funeral", "cemetery", "grief", "sadness", "crying", "tears",
    "hospital", "accident", "disaster", "storm", "destruction",
    "war", "conflict", "violence", "danger", "emergency", "fire",
    "flood", "earthquake", "abandoned", "ruins", "poverty", "homeless",
    "prison", "jail", "court", "police", "arrest",
}

NEUTRAL_SCENE_LABELS = {
    "office", "desk", "computer", "meeting", "conference", "classroom",
    "school", "library", "building", "street", "road", "car", "traffic",
    "store", "shop", "restaurant", "kitchen", "bedroom", "bathroom",
    "living_room", "hallway", "stairs", "elevator", "parking",
}


class SentimentAnalyzer:
    """Analyze sentiment from scene classifications and facial landmarks."""

    def classify_scene_sentiment(
        self, classifications: List[Tuple[str, float]]
    ) -> Dict[str, Any]:
        """
        Classify scene sentiment from image classification labels.

        Args:
            classifications: List of (label, confidence) tuples.

        Returns:
            Dict with sentiment ('positive', 'negative', 'neutral', 'mixed')
            and confidence score.
        """
        if not classifications:
            return {"sentiment": "neutral", "confidence": 0.0}

        positive_score = 0.0
        negative_score = 0.0
        neutral_score = 0.0
        total_weight = 0.0

        for label, confidence in classifications:
            label_lower = label.lower().replace(" ", "_")
            weight = confidence

            if label_lower in POSITIVE_SCENE_LABELS or any(
                pos in label_lower for pos in ["happy", "joy", "fun", "celebrat"]
            ):
                positive_score += weight
            elif label_lower in NEGATIVE_SCENE_LABELS or any(
                neg in label_lower for neg in ["sad", "grief", "danger", "disaster"]
            ):
                negative_score += weight
            else:
                neutral_score += weight

            total_weight += weight

        if total_weight == 0:
            return {"sentiment": "neutral", "confidence": 0.0}

        # Normalize scores
        positive_score /= total_weight
        negative_score /= total_weight
        neutral_score /= total_weight

        # Determine sentiment
        if positive_score > 0.4 and negative_score > 0.3:
            sentiment = "mixed"
            confidence = min(positive_score, negative_score)
        elif positive_score > negative_score and positive_score > neutral_score:
            sentiment = "positive"
            confidence = positive_score
        elif negative_score > positive_score and negative_score > neutral_score:
            sentiment = "negative"
            confidence = negative_score
        else:
            sentiment = "neutral"
            confidence = neutral_score

        return {"sentiment": sentiment, "confidence": float(confidence)}

    def classify_face_sentiment(
        self, landmarks: Dict[str, List[Dict[str, float]]]
    ) -> Dict[str, Any]:
        """
        Classify face sentiment from facial landmarks.

        Uses geometry of mouth, eyes, and eyebrows to infer emotion:
        - Mouth curvature (smile vs frown)
        - Eye openness
        - Eyebrow position

        Args:
            landmarks: Dict mapping landmark regions to point lists.

        Returns:
            Dict with sentiment and confidence.
        """
        if not landmarks:
            return {"sentiment": "neutral", "confidence": 0.3}

        scores = {
            "happy": 0.0,
            "sad": 0.0,
            "angry": 0.0,
            "surprised": 0.0,
            "neutral": 0.5,  # Base neutral score
        }

        # Analyze mouth curvature
        mouth_score = self._analyze_mouth(landmarks)
        if mouth_score > 0:
            scores["happy"] += mouth_score * 0.5
        elif mouth_score < 0:
            scores["sad"] += abs(mouth_score) * 0.4
            scores["angry"] += abs(mouth_score) * 0.2

        # Analyze eye openness
        eye_score = self._analyze_eyes(landmarks)
        if eye_score > 0.3:  # Wide eyes
            scores["surprised"] += eye_score * 0.4
        elif eye_score < -0.2:  # Squinted eyes
            scores["angry"] += abs(eye_score) * 0.3

        # Analyze eyebrow position
        brow_score = self._analyze_eyebrows(landmarks)
        if brow_score > 0:  # Raised eyebrows
            scores["surprised"] += brow_score * 0.3
        elif brow_score < 0:  # Furrowed eyebrows
            scores["angry"] += abs(brow_score) * 0.4

        # Find dominant sentiment
        max_sentiment = max(scores, key=scores.get)
        confidence = scores[max_sentiment]

        # Normalize confidence to 0-1
        total = sum(scores.values())
        if total > 0:
            confidence = scores[max_sentiment] / total

        return {"sentiment": max_sentiment, "confidence": float(confidence)}

    def _analyze_mouth(self, landmarks: Dict) -> float:
        """
        Analyze mouth curvature.

        Returns positive for smile, negative for frown, 0 for neutral.
        """
        outer_lips = landmarks.get("outerLips", [])
        if len(outer_lips) < 5:
            return 0.0

        # Get mouth corners and center
        try:
            # Approximate: first and last-ish points are corners
            left_corner = outer_lips[0]
            right_corner = outer_lips[len(outer_lips) // 2]

            # Center upper lip (approximately middle of upper points)
            upper_idx = len(outer_lips) // 4
            upper_center = outer_lips[upper_idx]

            # Calculate curvature: if center is above corner line, it's a smile
            corner_avg_y = (left_corner["y"] + right_corner["y"]) / 2
            curvature = corner_avg_y - upper_center["y"]  # Positive = smile (y increases downward)

            # Normalize by mouth width
            mouth_width = abs(right_corner["x"] - left_corner["x"])
            if mouth_width > 0:
                return curvature / mouth_width

        except (IndexError, KeyError):
            pass

        return 0.0

    def _analyze_eyes(self, landmarks: Dict) -> float:
        """
        Analyze eye openness.

        Returns positive for wide open, negative for squinted.
        """
        left_eye = landmarks.get("leftEye", [])
        right_eye = landmarks.get("rightEye", [])

        if len(left_eye) < 4 or len(right_eye) < 4:
            return 0.0

        try:
            # Estimate eye height/width ratio
            def eye_openness(eye_points):
                ys = [p["y"] for p in eye_points]
                xs = [p["x"] for p in eye_points]
                height = max(ys) - min(ys)
                width = max(xs) - min(xs)
                if width > 0:
                    return height / width
                return 0.0

            left_ratio = eye_openness(left_eye)
            right_ratio = eye_openness(right_eye)
            avg_ratio = (left_ratio + right_ratio) / 2

            # Typical eye ratio is around 0.3-0.4
            return (avg_ratio - 0.35) * 3  # Scale to roughly -1 to 1

        except (IndexError, KeyError):
            pass

        return 0.0

    def _analyze_eyebrows(self, landmarks: Dict) -> float:
        """
        Analyze eyebrow position relative to eyes.

        Returns positive for raised, negative for furrowed.
        """
        left_eye = landmarks.get("leftEye", [])
        right_eye = landmarks.get("rightEye", [])
        left_brow = landmarks.get("leftEyebrow", [])
        right_brow = landmarks.get("rightEyebrow", [])

        if not (left_eye and right_eye and left_brow and right_brow):
            return 0.0

        try:
            # Calculate average eye Y position
            eye_y = sum(p["y"] for p in left_eye + right_eye) / (len(left_eye) + len(right_eye))

            # Calculate average eyebrow Y position
            brow_y = sum(p["y"] for p in left_brow + right_brow) / (len(left_brow) + len(right_brow))

            # Distance from eyes to brows (positive = brows above eyes)
            # Note: y increases downward in image coords
            distance = eye_y - brow_y

            # Normalize by approximate face height (rough estimate)
            face_height = max(p["y"] for p in left_eye + right_eye) - min(p["y"] for p in left_brow + right_brow)
            if face_height > 0:
                normalized = distance / face_height
                # Typical is around 0.15-0.25
                return (normalized - 0.2) * 5  # Scale to roughly -1 to 1

        except (IndexError, KeyError, ZeroDivisionError):
            pass

        return 0.0
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_sentiment_analyzer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/photodb/utils/sentiment_analyzer.py tests/test_sentiment_analyzer.py
git commit -m "feat: add SentimentAnalyzer for scene and face sentiment"
```

---

## Task 7: Create Apple Vision Detection Stage

**Files:**

- Create: `src/photodb/stages/apple_vision_detection.py`
- Test: `tests/test_apple_vision_detection_stage.py`

**Step 1: Write failing test**

```python
# tests/test_apple_vision_detection_stage.py
"""Tests for Apple Vision detection stage."""
import sys
import pytest
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="Apple Vision only available on macOS"
)


class TestAppleVisionDetectionStage:
    """Test Apple Vision detection stage."""

    def test_stage_name(self):
        """Stage name should be 'apple_vision_detection'."""
        from photodb.stages.apple_vision_detection import AppleVisionDetectionStage

        mock_repo = MagicMock()
        stage = AppleVisionDetectionStage(mock_repo, {"IMG_PATH": "/tmp"})

        assert stage.stage_name == "apple_vision_detection"

    def test_process_photo_saves_detections(self, mock_repository, mock_photo):
        """Processing should save detections to repository."""
        from photodb.stages.apple_vision_detection import AppleVisionDetectionStage

        with patch("photodb.stages.apple_vision_detection.AppleVisionDetector") as MockDetector:
            mock_detector = MockDetector.return_value
            mock_detector.detect.return_value = {
                "status": "success",
                "detections": [
                    {
                        "face": {"bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}, "confidence": 0.95},
                        "body": {"bbox": {"x1": 5, "y1": 5, "x2": 100, "y2": 200}, "confidence": 0.9},
                    }
                ],
                "processing_time_ms": 50,
            }
            mock_detector.classify_scene.return_value = {
                "status": "success",
                "classifications": [("outdoor", 0.9), ("nature", 0.8)],
            }
            mock_detector.detect_landmarks.return_value = {
                "status": "success",
                "faces": [],
            }

            stage = AppleVisionDetectionStage(mock_repository, {"IMG_PATH": "/tmp"})

            from pathlib import Path
            result = stage.process_photo(mock_photo, Path("/tmp/test.jpg"))

            assert result is True
            assert mock_repository.create_person_detection.called


@pytest.fixture
def mock_repository():
    repo = MagicMock()
    repo.get_detections_for_photo.return_value = []
    return repo


@pytest.fixture
def mock_photo():
    from photodb.database.models import Photo
    from datetime import datetime, timezone

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

Run: `uv run pytest tests/test_apple_vision_detection_stage.py -v`
Expected: `ModuleNotFoundError`

**Step 3: Implement AppleVisionDetectionStage**

```python
# src/photodb/stages/apple_vision_detection.py
"""
Apple Vision detection stage: Face/body detection, scene classification, and sentiment analysis.

Uses Apple Vision Framework via PyObjC for native macOS performance.
"""
import logging
import sys
from pathlib import Path
from typing import Optional

from .base import BaseStage
from ..database.models import (
    Photo,
    PersonDetection,
    AnalysisOutput,
    SceneAnalysis,
)

if sys.platform != "darwin":
    raise ImportError("Apple Vision detection stage only available on macOS")

from ..utils.apple_vision_detector import AppleVisionDetector
from ..utils.sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)


class AppleVisionDetectionStage(BaseStage):
    """Stage for detection using Apple Vision Framework.

    Performs:
    - Face and body detection
    - Scene classification
    - Facial landmark detection
    - Sentiment analysis (scene and face)

    Stores raw outputs in analysis_output table for auditability.
    """

    stage_name = "apple_vision_detection"

    def __init__(self, repository, config: dict):
        super().__init__(repository, config)
        self.detector = AppleVisionDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        logger.info("AppleVisionDetectionStage initialized")

    def process_photo(self, photo: Photo, file_path: Path) -> bool:
        """Process detection, classification, and sentiment for a photo."""
        try:
            # Check normalized image exists
            if not photo.normalized_path:
                logger.warning(f"No normalized path for photo {photo.id}, skipping")
                return False

            normalized_path = Path(self.config["IMG_PATH"]) / photo.normalized_path
            if not normalized_path.exists():
                logger.error(f"Normalized image not found: {normalized_path}")
                return False

            # Clear existing detections if reprocessing
            existing = self.repository.get_detections_for_photo(photo.id)
            if existing:
                logger.debug(f"Clearing {len(existing)} existing detections")
                self.repository.delete_detections_for_photo(photo.id)

            # 1. Face and body detection
            detection_result = self.detector.detect(str(normalized_path))

            # Store raw detection output
            detection_output = AnalysisOutput.create(
                photo_id=photo.id,
                model_type="detector",
                model_name="apple_vision",
                output=detection_result,
                processing_time_ms=detection_result.get("processing_time_ms"),
                device="ane",  # Apple Neural Engine
            )
            self.repository.create_analysis_output(detection_output)

            if detection_result["status"] == "error":
                logger.error(f"Detection failed: {detection_result.get('error')}")
                return False

            # 2. Scene classification
            scene_result = self.detector.classify_scene(str(normalized_path))

            scene_output = AnalysisOutput.create(
                photo_id=photo.id,
                model_type="classifier",
                model_name="apple_vision",
                output=scene_result,
                processing_time_ms=scene_result.get("processing_time_ms"),
                device="ane",
            )
            self.repository.create_analysis_output(scene_output)

            # 3. Analyze scene sentiment
            if scene_result["status"] == "success":
                classifications = [
                    (c["identifier"], c["confidence"])
                    for c in scene_result["classifications"]
                ]
                scene_sentiment = self.sentiment_analyzer.classify_scene_sentiment(classifications)

                scene_analysis = SceneAnalysis.create(
                    photo_id=photo.id,
                    top_labels=[c["identifier"] for c in scene_result["classifications"][:5]],
                    top_confidences=[c["confidence"] for c in scene_result["classifications"][:5]],
                    scene_sentiment=scene_sentiment["sentiment"],
                    scene_sentiment_confidence=scene_sentiment["confidence"],
                    analysis_output_id=scene_output.id,
                )
                self.repository.create_scene_analysis(scene_analysis)

            # 4. Landmark detection for face sentiment
            landmark_result = self.detector.detect_landmarks(str(normalized_path))

            landmarks_output: Optional[AnalysisOutput] = None
            if landmark_result["status"] == "success":
                landmarks_output = AnalysisOutput.create(
                    photo_id=photo.id,
                    model_type="landmarks",
                    model_name="apple_vision",
                    output=landmark_result,
                    processing_time_ms=landmark_result.get("processing_time_ms"),
                    device="ane",
                )
                self.repository.create_analysis_output(landmarks_output)

            # 5. Create person detection records
            if detection_result["status"] == "no_detections":
                logger.debug(f"No detections in {file_path}")
                return True

            detections_saved = 0
            for i, det in enumerate(detection_result["detections"]):
                face_data = det.get("face")
                body_data = det.get("body")

                detection = self._create_detection_record(
                    photo.id, face_data, body_data, detection_output.id
                )
                self.repository.create_person_detection(detection)
                detections_saved += 1

                # Analyze face sentiment if landmarks available
                if (
                    face_data
                    and landmark_result["status"] == "success"
                    and i < len(landmark_result.get("faces", []))
                ):
                    face_landmarks = landmark_result["faces"][i].get("landmarks", {})
                    face_sentiment = self.sentiment_analyzer.classify_face_sentiment(face_landmarks)

                    if detection.id:
                        self.repository.update_detection_sentiment(
                            detection.id,
                            face_sentiment["sentiment"],
                            face_sentiment["confidence"],
                            landmarks_output.id if landmarks_output else None,
                        )

            logger.info(f"Saved {detections_saved} detections for {file_path}")
            return True

        except Exception as e:
            logger.error(f"Apple Vision detection failed for {file_path}: {e}")
            return False

    def _create_detection_record(
        self,
        photo_id: int,
        face_data: dict | None,
        body_data: dict | None,
        analysis_output_id: int | None,
    ) -> PersonDetection:
        """Create PersonDetection from Apple Vision output."""
        face_bbox_x = None
        face_bbox_y = None
        face_bbox_width = None
        face_bbox_height = None
        face_confidence = None

        if face_data:
            bbox = face_data["bbox"]
            face_bbox_x = bbox["x1"]
            face_bbox_y = bbox["y1"]
            face_bbox_width = bbox["x2"] - bbox["x1"]
            face_bbox_height = bbox["y2"] - bbox["y1"]
            face_confidence = face_data["confidence"]

        body_bbox_x = None
        body_bbox_y = None
        body_bbox_width = None
        body_bbox_height = None
        body_confidence = None

        if body_data:
            bbox = body_data["bbox"]
            body_bbox_x = bbox["x1"]
            body_bbox_y = bbox["y1"]
            body_bbox_width = bbox["x2"] - bbox["x1"]
            body_bbox_height = bbox["y2"] - bbox["y1"]
            body_confidence = body_data["confidence"]

        return PersonDetection.create(
            photo_id=photo_id,
            face_bbox_x=face_bbox_x,
            face_bbox_y=face_bbox_y,
            face_bbox_width=face_bbox_width,
            face_bbox_height=face_bbox_height,
            face_confidence=face_confidence,
            body_bbox_x=body_bbox_x,
            body_bbox_y=body_bbox_y,
            body_bbox_width=body_bbox_width,
            body_bbox_height=body_bbox_height,
            body_confidence=body_confidence,
            detector_model="apple_vision",
            analysis_output_id=analysis_output_id,
        )
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_apple_vision_detection_stage.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/photodb/stages/apple_vision_detection.py tests/test_apple_vision_detection_stage.py
git commit -m "feat: add AppleVisionDetectionStage with sentiment analysis"
```

---

## Task 8: Register Stage in CLI

**Files:**

- Modify: `src/photodb/cli_local.py`

**Step 1: Add apple_vision_detection to available stages**

Find the stage registration section and add:

```python
# In the STAGES dict or stage selection logic:

# Check if running on macOS for Apple Vision
import sys
if sys.platform == "darwin":
    from .stages.apple_vision_detection import AppleVisionDetectionStage
    STAGES["apple_vision_detection"] = AppleVisionDetectionStage
```

**Step 2: Update stage help text**

Update the `--stage` option help to include `apple_vision_detection`:

```python
@click.option(
    "--stage",
    type=click.Choice(["normalize", "metadata", "detection", "age_gender", "apple_vision_detection"]),
    help="Run only a specific stage",
)
```

**Step 3: Test CLI help**

Run: `uv run process-local --help`
Expected: Shows `apple_vision_detection` in stage choices (on macOS)

**Step 4: Commit**

```bash
git add src/photodb/cli_local.py
git commit -m "cli: register AppleVisionDetectionStage in process-local"
```

---

## Task 9: Update Configuration Documentation

**Files:**

- Modify: `CLAUDE.md`

**Step 1: Add Apple Vision configuration section**

Add after the Age/Gender Stage Configuration section:

```markdown
### Apple Vision Detection Stage Configuration (macOS only)

The Apple Vision detection stage uses native macOS Vision Framework APIs and requires no additional model downloads.

**Capabilities:**
- Face detection with bounding boxes
- Body/human detection with bounding boxes
- Facial landmark detection (76 points)
- Scene classification (1303 categories)
- Derived sentiment analysis (scene and face)

**Usage:**
```bash
# Run Apple Vision detection only
uv run process-local /path/to/photos --stage apple_vision_detection

# Force reprocessing with Apple Vision
uv run process-local /path/to/photos --stage apple_vision_detection --force
```

**Note:** This stage is only available on macOS. It uses the Apple Neural Engine when available for fast, efficient inference. Unlike YOLO+FaceNet, no GPU or model downloads required.

```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add Apple Vision detection configuration to CLAUDE.md"
```

---

## Task 10: Create Migration Guide

**Files:**

- Create: `docs/APPLE_VISION_MIGRATION.md`

**Step 1: Write migration guide**

```markdown
# Migrating from YOLO+FaceNet to Apple Vision Framework

This guide explains how to migrate your PhotoDB installation from YOLO-based detection to Apple Vision Framework detection.

## Prerequisites

- macOS 11.0 or later
- Python 3.13+
- PhotoDB with latest schema migrations applied

## Why Migrate?

| Feature | YOLO+FaceNet | Apple Vision |
|---------|--------------|--------------|
| Model downloads | Required (~300MB) | None (built-in) |
| GPU requirement | Recommended | Uses Neural Engine |
| Thread safety | CoreML only | Native |
| Face embeddings | 512-D vectors | Not provided* |
| Body detection | Yes | Yes |
| Scene classification | No | Yes (1303 classes) |
| Facial landmarks | No | Yes (76 points) |
| Sentiment analysis | No | Yes (derived) |

*Note: Apple Vision doesn't provide face embeddings. For face clustering, you'll need to:
1. Continue using FaceNet alongside Apple Vision, or
2. Use a different embedding model, or
3. Rely on other clustering approaches

## Migration Steps

### 1. Apply Database Migration

```bash
psql $DATABASE_URL -f migrations/006_add_analysis_output.sql
```

This adds:

- `analysis_output` table for model-agnostic raw outputs
- `detector_registry` table for tracking available models
- `scene_analysis` table for photo-level classifications
- New columns on `person_detection` for sentiment

### 2. Run Apple Vision Detection

```bash
# Process all photos with Apple Vision
uv run process-local /path/to/photos --stage apple_vision_detection

# Or reprocess specific photos
uv run process-local /path/to/photos --stage apple_vision_detection --force
```

### 3. Verify Results

Query the new tables:

```sql
-- Check scene analysis
SELECT p.filename, sa.top_labels, sa.scene_sentiment
FROM scene_analysis sa
JOIN photo p ON sa.photo_id = p.id
LIMIT 10;

-- Check face sentiments
SELECT p.filename, pd.face_sentiment, pd.face_sentiment_confidence
FROM person_detection pd
JOIN photo p ON pd.photo_id = p.id
WHERE pd.face_sentiment IS NOT NULL
LIMIT 10;

-- View raw model outputs
SELECT model_name, model_type, COUNT(*)
FROM analysis_output
GROUP BY model_name, model_type;
```

### 4. Keep Both Detectors (Optional)

You can run both detection pipelines:

```bash
# Run YOLO detection (for embeddings)
uv run process-local /path/to/photos --stage detection

# Run Apple Vision detection (for sentiment/classification)
uv run process-local /path/to/photos --stage apple_vision_detection
```

Results are stored with different `detector_model` values:

- `"YOLO+FaceNet"` for the original detector
- `"apple_vision"` for Apple Vision Framework

## Schema Design Philosophy

The new schema is designed to support multiple detection models:

1. **Raw outputs preserved**: All model outputs stored in `analysis_output` as JSONB
2. **Model-agnostic**: `detector_registry` tracks capabilities of each model
3. **Linked records**: `person_detection.analysis_output_id` links to raw data
4. **Extensible**: Add new models without schema changes

## Troubleshooting

### "Apple Vision only available on macOS"

The Apple Vision stage only works on macOS. On Linux/Windows, continue using YOLO+FaceNet.

### No face sentiment detected

Face sentiment requires successful landmark detection. Check:

```sql
SELECT COUNT(*) FROM analysis_output
WHERE model_type = 'landmarks' AND model_name = 'apple_vision';
```

### Performance issues

Apple Vision uses the Neural Engine automatically. If slow:

1. Ensure photos are normalized first
2. Process in smaller batches
3. Check Activity Monitor for "nsurlsessiond" (shouldn't be running during local processing)

```

**Step 2: Commit**

```bash
git add docs/APPLE_VISION_MIGRATION.md
git commit -m "docs: add Apple Vision migration guide"
```

---

## Summary

This plan migrates PhotoDB from YOLO+FaceNet to Apple Vision Framework with:

1. **New dependencies**: PyObjC for macOS Vision bindings
2. **Generalizable schema**: `analysis_output` stores raw model outputs, `detector_registry` tracks model capabilities
3. **Apple Vision detector**: Face/body detection, scene classification, landmarks
4. **Sentiment analysis**: Scene sentiment from classifications, face sentiment from landmarks
5. **New stage**: `AppleVisionDetectionStage` integrates all components
6. **Backward compatibility**: Original YOLO detection remains available

Key architectural decisions:

- Raw model outputs preserved for auditability and debugging
- Schema supports adding new models without migrations
- Sentiment derived from Vision outputs rather than separate model
- Face embeddings not replaced (requires separate decision for clustering)

---

Plan complete and saved to `docs/plans/2026-02-01-apple-vision-migration.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
