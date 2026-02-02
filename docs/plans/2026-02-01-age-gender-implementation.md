# Age/Gender Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add age/gender estimation using MiVOLO with body detection, replacing MTCNN with YOLOv8x person_face detector.

**Architecture:** Two new pipeline stages (`detection` and `age_gender`) replace the existing `faces` stage. A new `person_detection` table replaces the `face` table, storing both face and body bboxes plus age/gender data. Person-level aggregates (birth year, gender) are added to the `person` table.

**Tech Stack:** YOLOv8x (person_face variant), MiVOLO, FaceNet (embeddings), PostgreSQL, pgvector

---

## Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add ultralytics package**

```bash
cd /Users/andrewchoi/dev/home/photodb/.worktrees/age-gender-detection
uv add ultralytics
```

Expected: Package added to pyproject.toml

**Step 2: Add MiVOLO from GitHub**

```bash
uv add git+https://github.com/WildChlamydia/MiVOLO.git
```

Expected: Package added to pyproject.toml

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add ultralytics and MiVOLO for age/gender detection"
```

---

## Task 2: Create Database Migration Script

**Files:**
- Create: `migrations/005_add_person_detection.sql`
- Create: `migrations/005_add_person_detection_rollback.sql`

**Step 1: Write the migration SQL**

Create `migrations/005_add_person_detection.sql`:

```sql
-- Migration: Add person_detection table and migrate from face table
-- This replaces the face table with a unified person_detection table

BEGIN;

-- 1. Create person_detection table
CREATE TABLE IF NOT EXISTS person_detection (
    id BIGSERIAL PRIMARY KEY,
    photo_id BIGINT NOT NULL REFERENCES photo(id) ON DELETE CASCADE,

    -- Face bounding box (nullable - may have body only)
    face_bbox_x REAL,
    face_bbox_y REAL,
    face_bbox_width REAL,
    face_bbox_height REAL,
    face_confidence REAL,

    -- Body bounding box (nullable - may have face only)
    body_bbox_x REAL,
    body_bbox_y REAL,
    body_bbox_width REAL,
    body_bbox_height REAL,
    body_confidence REAL,

    -- Age/gender from MiVOLO (nullable until age_gender stage runs)
    age_estimate REAL,
    gender CHAR(1) CHECK (gender IN ('M', 'F', 'U')),
    gender_confidence REAL,
    mivolo_output JSONB,

    -- Clustering (carried over from face table)
    person_id BIGINT REFERENCES person(id) ON DELETE SET NULL,
    cluster_status TEXT CHECK (cluster_status IN ('auto', 'pending', 'manual', 'unassigned', 'constrained')),
    cluster_id BIGINT,
    cluster_confidence REAL,

    -- Detector metadata
    detector_model TEXT,
    detector_version TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Create indexes for person_detection
CREATE INDEX idx_person_detection_photo_id ON person_detection(photo_id);
CREATE INDEX idx_person_detection_person_id ON person_detection(person_id);
CREATE INDEX idx_person_detection_cluster_id ON person_detection(cluster_id);
CREATE INDEX idx_person_detection_cluster_status ON person_detection(cluster_status);
CREATE INDEX idx_person_detection_age ON person_detection(age_estimate) WHERE age_estimate IS NOT NULL;
CREATE INDEX idx_person_detection_gender ON person_detection(gender) WHERE gender IS NOT NULL;

-- 3. Migrate existing face data
INSERT INTO person_detection (
    id, photo_id,
    face_bbox_x, face_bbox_y, face_bbox_width, face_bbox_height,
    face_confidence, person_id, cluster_status, cluster_id,
    cluster_confidence, detector_model, created_at
)
SELECT
    id, photo_id,
    bbox_x, bbox_y, bbox_width, bbox_height,
    confidence, person_id, cluster_status, cluster_id,
    cluster_confidence, 'mtcnn-legacy', NOW()
FROM face;

-- 4. Update face_embedding foreign key
ALTER TABLE face_embedding DROP CONSTRAINT face_embedding_face_id_fkey;
ALTER TABLE face_embedding RENAME COLUMN face_id TO person_detection_id;
ALTER TABLE face_embedding ADD CONSTRAINT face_embedding_person_detection_fkey
    FOREIGN KEY (person_detection_id) REFERENCES person_detection(id) ON DELETE CASCADE;

-- 5. Update cluster table references
ALTER TABLE "cluster" DROP CONSTRAINT IF EXISTS cluster_representative_face_id_fkey;
ALTER TABLE "cluster" DROP CONSTRAINT IF EXISTS cluster_medoid_face_id_fkey;
ALTER TABLE "cluster" RENAME COLUMN representative_face_id TO representative_detection_id;
ALTER TABLE "cluster" RENAME COLUMN medoid_face_id TO medoid_detection_id;
ALTER TABLE "cluster" ADD CONSTRAINT cluster_representative_detection_fkey
    FOREIGN KEY (representative_detection_id) REFERENCES person_detection(id) ON DELETE SET NULL;
ALTER TABLE "cluster" ADD CONSTRAINT cluster_medoid_detection_fkey
    FOREIGN KEY (medoid_detection_id) REFERENCES person_detection(id) ON DELETE SET NULL;

-- 6. Add cluster_id foreign key to person_detection (after cluster exists)
ALTER TABLE person_detection ADD CONSTRAINT person_detection_cluster_fkey
    FOREIGN KEY (cluster_id) REFERENCES "cluster"(id) ON DELETE SET NULL;

-- 7. Update must_link table
ALTER TABLE must_link DROP CONSTRAINT must_link_face_id_1_fkey;
ALTER TABLE must_link DROP CONSTRAINT must_link_face_id_2_fkey;
ALTER TABLE must_link RENAME COLUMN face_id_1 TO detection_id_1;
ALTER TABLE must_link RENAME COLUMN face_id_2 TO detection_id_2;
ALTER TABLE must_link ADD CONSTRAINT must_link_detection_id_1_fkey
    FOREIGN KEY (detection_id_1) REFERENCES person_detection(id) ON DELETE CASCADE;
ALTER TABLE must_link ADD CONSTRAINT must_link_detection_id_2_fkey
    FOREIGN KEY (detection_id_2) REFERENCES person_detection(id) ON DELETE CASCADE;

-- 8. Update cannot_link table
ALTER TABLE cannot_link DROP CONSTRAINT cannot_link_face_id_1_fkey;
ALTER TABLE cannot_link DROP CONSTRAINT cannot_link_face_id_2_fkey;
ALTER TABLE cannot_link RENAME COLUMN face_id_1 TO detection_id_1;
ALTER TABLE cannot_link RENAME COLUMN face_id_2 TO detection_id_2;
ALTER TABLE cannot_link ADD CONSTRAINT cannot_link_detection_id_1_fkey
    FOREIGN KEY (detection_id_1) REFERENCES person_detection(id) ON DELETE CASCADE;
ALTER TABLE cannot_link ADD CONSTRAINT cannot_link_detection_id_2_fkey
    FOREIGN KEY (detection_id_2) REFERENCES person_detection(id) ON DELETE CASCADE;

-- 9. Update face_match_candidate table
ALTER TABLE face_match_candidate DROP CONSTRAINT face_match_candidate_face_id_fkey;
ALTER TABLE face_match_candidate RENAME COLUMN face_id TO detection_id;
ALTER TABLE face_match_candidate ADD CONSTRAINT face_match_candidate_detection_fkey
    FOREIGN KEY (detection_id) REFERENCES person_detection(id) ON DELETE CASCADE;

-- 10. Add age/gender columns to person table
ALTER TABLE person ADD COLUMN estimated_birth_year INT;
ALTER TABLE person ADD COLUMN birth_year_stddev REAL;
ALTER TABLE person ADD COLUMN gender CHAR(1) CHECK (gender IN ('M', 'F', 'U'));
ALTER TABLE person ADD COLUMN gender_confidence REAL;
ALTER TABLE person ADD COLUMN age_gender_sample_count INT DEFAULT 0;
ALTER TABLE person ADD COLUMN age_gender_updated_at TIMESTAMPTZ;

-- 11. Reset sequence
SELECT setval('person_detection_id_seq', COALESCE((SELECT MAX(id) FROM person_detection), 1));

-- 12. Drop old face table
DROP TABLE face;

-- 13. Record migration
INSERT INTO schema_migrations (version, description)
VALUES ('005', 'Add person_detection table, migrate from face table, add age/gender to person')
ON CONFLICT (version) DO NOTHING;

COMMIT;
```

**Step 2: Write the rollback SQL**

Create `migrations/005_add_person_detection_rollback.sql`:

```sql
-- Rollback: Restore face table from person_detection
-- WARNING: This will lose body detection and age/gender data

BEGIN;

-- 1. Recreate face table
CREATE TABLE IF NOT EXISTS face (
    id BIGSERIAL PRIMARY KEY,
    photo_id BIGINT NOT NULL,
    bbox_x REAL NOT NULL,
    bbox_y REAL NOT NULL,
    bbox_width REAL NOT NULL,
    bbox_height REAL NOT NULL,
    confidence DECIMAL(3, 2) NOT NULL DEFAULT 0,
    person_id BIGINT,
    cluster_status TEXT CHECK (cluster_status IN ('auto', 'pending', 'manual', 'unassigned', 'constrained')),
    cluster_id BIGINT,
    cluster_confidence DECIMAL(3, 2) DEFAULT 0,
    unassigned_since TIMESTAMPTZ DEFAULT NULL,
    FOREIGN KEY (photo_id) REFERENCES photo(id) ON DELETE CASCADE,
    FOREIGN KEY (person_id) REFERENCES person(id) ON DELETE SET NULL
);

-- 2. Migrate data back (only detections with faces)
INSERT INTO face (id, photo_id, bbox_x, bbox_y, bbox_width, bbox_height, confidence, person_id, cluster_status, cluster_id, cluster_confidence)
SELECT id, photo_id, face_bbox_x, face_bbox_y, face_bbox_width, face_bbox_height, face_confidence, person_id, cluster_status, cluster_id, cluster_confidence
FROM person_detection
WHERE face_bbox_x IS NOT NULL;

-- 3. Restore face_embedding FK
ALTER TABLE face_embedding DROP CONSTRAINT face_embedding_person_detection_fkey;
ALTER TABLE face_embedding RENAME COLUMN person_detection_id TO face_id;
ALTER TABLE face_embedding ADD CONSTRAINT face_embedding_face_id_fkey
    FOREIGN KEY (face_id) REFERENCES face(id) ON DELETE CASCADE;

-- 4. Restore cluster FKs
ALTER TABLE "cluster" DROP CONSTRAINT cluster_representative_detection_fkey;
ALTER TABLE "cluster" DROP CONSTRAINT cluster_medoid_detection_fkey;
ALTER TABLE "cluster" RENAME COLUMN representative_detection_id TO representative_face_id;
ALTER TABLE "cluster" RENAME COLUMN medoid_detection_id TO medoid_face_id;

-- 5. Add back face FK to cluster
ALTER TABLE face ADD CONSTRAINT face_cluster_fkey
    FOREIGN KEY (cluster_id) REFERENCES "cluster"(id) ON DELETE SET NULL;
ALTER TABLE "cluster" ADD CONSTRAINT cluster_representative_face_id_fkey
    FOREIGN KEY (representative_face_id) REFERENCES face(id) ON DELETE SET NULL;
ALTER TABLE "cluster" ADD CONSTRAINT cluster_medoid_face_id_fkey
    FOREIGN KEY (medoid_face_id) REFERENCES face(id) ON DELETE SET NULL;

-- 6. Restore must_link
ALTER TABLE must_link DROP CONSTRAINT must_link_detection_id_1_fkey;
ALTER TABLE must_link DROP CONSTRAINT must_link_detection_id_2_fkey;
ALTER TABLE must_link RENAME COLUMN detection_id_1 TO face_id_1;
ALTER TABLE must_link RENAME COLUMN detection_id_2 TO face_id_2;
ALTER TABLE must_link ADD CONSTRAINT must_link_face_id_1_fkey
    FOREIGN KEY (face_id_1) REFERENCES face(id) ON DELETE CASCADE;
ALTER TABLE must_link ADD CONSTRAINT must_link_face_id_2_fkey
    FOREIGN KEY (face_id_2) REFERENCES face(id) ON DELETE CASCADE;

-- 7. Restore cannot_link
ALTER TABLE cannot_link DROP CONSTRAINT cannot_link_detection_id_1_fkey;
ALTER TABLE cannot_link DROP CONSTRAINT cannot_link_detection_id_2_fkey;
ALTER TABLE cannot_link RENAME COLUMN detection_id_1 TO face_id_1;
ALTER TABLE cannot_link RENAME COLUMN detection_id_2 TO face_id_2;
ALTER TABLE cannot_link ADD CONSTRAINT cannot_link_face_id_1_fkey
    FOREIGN KEY (face_id_1) REFERENCES face(id) ON DELETE CASCADE;
ALTER TABLE cannot_link ADD CONSTRAINT cannot_link_face_id_2_fkey
    FOREIGN KEY (face_id_2) REFERENCES face(id) ON DELETE CASCADE;

-- 8. Restore face_match_candidate
ALTER TABLE face_match_candidate DROP CONSTRAINT face_match_candidate_detection_fkey;
ALTER TABLE face_match_candidate RENAME COLUMN detection_id TO face_id;
ALTER TABLE face_match_candidate ADD CONSTRAINT face_match_candidate_face_id_fkey
    FOREIGN KEY (face_id) REFERENCES face(id) ON DELETE CASCADE;

-- 9. Remove person age/gender columns
ALTER TABLE person DROP COLUMN IF EXISTS estimated_birth_year;
ALTER TABLE person DROP COLUMN IF EXISTS birth_year_stddev;
ALTER TABLE person DROP COLUMN IF EXISTS gender;
ALTER TABLE person DROP COLUMN IF EXISTS gender_confidence;
ALTER TABLE person DROP COLUMN IF EXISTS age_gender_sample_count;
ALTER TABLE person DROP COLUMN IF EXISTS age_gender_updated_at;

-- 10. Drop person_detection
DROP TABLE person_detection;

-- 11. Remove migration record
DELETE FROM schema_migrations WHERE version = '005';

COMMIT;
```

**Step 3: Commit**

```bash
git add migrations/005_add_person_detection.sql migrations/005_add_person_detection_rollback.sql
git commit -m "migration: add person_detection table replacing face table"
```

---

## Task 3: Update Database Models

**Files:**
- Modify: `src/photodb/database/models.py`

**Step 1: Add PersonDetection dataclass**

Add after the `Face` class (around line 279):

```python
@dataclass
class PersonDetection:
    id: Optional[int]
    photo_id: int
    # Face bounding box (nullable)
    face_bbox_x: Optional[float]
    face_bbox_y: Optional[float]
    face_bbox_width: Optional[float]
    face_bbox_height: Optional[float]
    face_confidence: Optional[float]
    # Body bounding box (nullable)
    body_bbox_x: Optional[float]
    body_bbox_y: Optional[float]
    body_bbox_width: Optional[float]
    body_bbox_height: Optional[float]
    body_confidence: Optional[float]
    # Age/gender (nullable until age_gender stage runs)
    age_estimate: Optional[float]
    gender: Optional[str]  # 'M', 'F', 'U'
    gender_confidence: Optional[float]
    mivolo_output: Optional[Dict[str, Any]]
    # Clustering
    person_id: Optional[int]
    cluster_status: Optional[str]
    cluster_id: Optional[int]
    cluster_confidence: Optional[float]
    # Detector metadata
    detector_model: Optional[str]
    detector_version: Optional[str]
    created_at: Optional[datetime] = None

    @classmethod
    def create(
        cls,
        photo_id: int,
        face_bbox_x: Optional[float] = None,
        face_bbox_y: Optional[float] = None,
        face_bbox_width: Optional[float] = None,
        face_bbox_height: Optional[float] = None,
        face_confidence: Optional[float] = None,
        body_bbox_x: Optional[float] = None,
        body_bbox_y: Optional[float] = None,
        body_bbox_width: Optional[float] = None,
        body_bbox_height: Optional[float] = None,
        body_confidence: Optional[float] = None,
        detector_model: Optional[str] = None,
        detector_version: Optional[str] = None,
    ) -> "PersonDetection":
        """Create a new person detection record."""
        return cls(
            id=None,
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
            age_estimate=None,
            gender=None,
            gender_confidence=None,
            mivolo_output=None,
            person_id=None,
            cluster_status=None,
            cluster_id=None,
            cluster_confidence=None,
            detector_model=detector_model,
            detector_version=detector_version,
            created_at=datetime.now(timezone.utc),
        )

    def has_face(self) -> bool:
        """Check if detection includes a face."""
        return self.face_bbox_x is not None

    def has_body(self) -> bool:
        """Check if detection includes a body."""
        return self.body_bbox_x is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "photo_id": self.photo_id,
            "face_bbox_x": self.face_bbox_x,
            "face_bbox_y": self.face_bbox_y,
            "face_bbox_width": self.face_bbox_width,
            "face_bbox_height": self.face_bbox_height,
            "face_confidence": self.face_confidence,
            "body_bbox_x": self.body_bbox_x,
            "body_bbox_y": self.body_bbox_y,
            "body_bbox_width": self.body_bbox_width,
            "body_bbox_height": self.body_bbox_height,
            "body_confidence": self.body_confidence,
            "age_estimate": self.age_estimate,
            "gender": self.gender,
            "gender_confidence": self.gender_confidence,
            "mivolo_output": self.mivolo_output,
            "person_id": self.person_id,
            "cluster_status": self.cluster_status,
            "cluster_id": self.cluster_id,
            "cluster_confidence": self.cluster_confidence,
            "detector_model": self.detector_model,
            "detector_version": self.detector_version,
            "created_at": self.created_at,
        }
```

**Step 2: Update Person dataclass**

Add age/gender fields to the Person class (after line 188):

```python
@dataclass
class Person:
    id: Optional[int]
    first_name: str
    last_name: Optional[str]
    # Age/gender aggregates
    estimated_birth_year: Optional[int] = None
    birth_year_stddev: Optional[float] = None
    gender: Optional[str] = None  # 'M', 'F', 'U'
    gender_confidence: Optional[float] = None
    age_gender_sample_count: int = 0
    age_gender_updated_at: Optional[datetime] = None
    created_at: datetime = None
    updated_at: datetime = None
```

Update the `create` and `to_dict` methods accordingly.

**Step 3: Remove Face class**

Delete the `Face` class (lines 222-278) as it's replaced by `PersonDetection`.

**Step 4: Update Cluster class**

Change field names from `representative_face_id` to `representative_detection_id` and `medoid_face_id` to `medoid_detection_id`.

**Step 5: Commit**

```bash
git add src/photodb/database/models.py
git commit -m "feat: add PersonDetection model, update Person with age/gender fields"
```

---

## Task 4: Update Repository with PersonDetection Methods

**Files:**
- Modify: `src/photodb/database/repository.py`

**Step 1: Add PersonDetection imports and methods**

Add to imports:
```python
from .models import PersonDetection
```

Add repository methods:

```python
def create_person_detection(self, detection: PersonDetection) -> None:
    """Insert a new person detection record."""
    with self.pool.transaction() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """INSERT INTO person_detection (
                    photo_id, face_bbox_x, face_bbox_y, face_bbox_width, face_bbox_height,
                    face_confidence, body_bbox_x, body_bbox_y, body_bbox_width, body_bbox_height,
                    body_confidence, detector_model, detector_version, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                (
                    detection.photo_id,
                    detection.face_bbox_x, detection.face_bbox_y,
                    detection.face_bbox_width, detection.face_bbox_height,
                    detection.face_confidence,
                    detection.body_bbox_x, detection.body_bbox_y,
                    detection.body_bbox_width, detection.body_bbox_height,
                    detection.body_confidence,
                    detection.detector_model, detection.detector_version,
                    detection.created_at,
                ),
            )
            detection.id = cursor.fetchone()[0]

def get_detections_for_photo(self, photo_id: int) -> List[PersonDetection]:
    """Get all person detections for a photo."""
    with self.pool.get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SELECT * FROM person_detection WHERE photo_id = %s",
                (photo_id,)
            )
            rows = cursor.fetchall()
            return [PersonDetection(**dict(row)) for row in rows]

def delete_detections_for_photo(self, photo_id: int) -> int:
    """Delete all detections for a photo. Returns count deleted."""
    with self.pool.transaction() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM person_detection WHERE photo_id = %s",
                (photo_id,)
            )
            return cursor.rowcount

def save_detection_embedding(self, detection_id: int, embedding: List[float]) -> None:
    """Save face embedding for a person detection."""
    with self.pool.transaction() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """INSERT INTO face_embedding (person_detection_id, embedding)
                   VALUES (%s, %s)
                   ON CONFLICT (person_detection_id) DO UPDATE SET embedding = EXCLUDED.embedding""",
                (detection_id, embedding),
            )

def update_detection_age_gender(
    self,
    detection_id: int,
    age_estimate: float,
    gender: str,
    gender_confidence: float,
    mivolo_output: Dict[str, Any],
) -> None:
    """Update age/gender fields for a person detection."""
    with self.pool.transaction() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """UPDATE person_detection
                   SET age_estimate = %s, gender = %s, gender_confidence = %s, mivolo_output = %s
                   WHERE id = %s""",
                (age_estimate, gender, gender_confidence, json.dumps(mivolo_output), detection_id),
            )

def get_detections_for_person(self, person_id: int) -> List[PersonDetection]:
    """Get all detections for a person (via cluster assignments)."""
    with self.pool.get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SELECT * FROM person_detection WHERE person_id = %s",
                (person_id,)
            )
            rows = cursor.fetchall()
            return [PersonDetection(**dict(row)) for row in rows]

def update_person_age_gender(
    self,
    person_id: int,
    estimated_birth_year: Optional[int],
    birth_year_stddev: Optional[float],
    gender: Optional[str],
    gender_confidence: Optional[float],
    sample_count: int,
) -> None:
    """Update aggregated age/gender statistics for a person."""
    with self.pool.transaction() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """UPDATE person SET
                    estimated_birth_year = %s,
                    birth_year_stddev = %s,
                    gender = %s,
                    gender_confidence = %s,
                    age_gender_sample_count = %s,
                    age_gender_updated_at = NOW(),
                    updated_at = NOW()
                   WHERE id = %s""",
                (estimated_birth_year, birth_year_stddev, gender, gender_confidence, sample_count, person_id),
            )
```

**Step 2: Update existing methods**

Rename methods referencing `face` to `detection`:
- `get_faces_for_photo` → keep as alias or update callers
- `save_face_embedding` → `save_detection_embedding`

**Step 3: Commit**

```bash
git add src/photodb/database/repository.py
git commit -m "feat: add PersonDetection repository methods"
```

---

## Task 5: Create PersonDetector Utility

**Files:**
- Create: `src/photodb/utils/person_detector.py`
- Create: `tests/test_person_detector.py`

**Step 1: Write the test**

Create `tests/test_person_detector.py`:

```python
"""Tests for PersonDetector utility."""

import pytest
from pathlib import Path
from PIL import Image
import tempfile

from photodb.utils.person_detector import PersonDetector


class TestPersonDetector:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    def sample_image(self, temp_dir):
        """Create a simple test image."""
        img_path = temp_dir / "test.jpg"
        img = Image.new("RGB", (640, 480), color="white")
        img.save(img_path, "JPEG")
        return img_path

    def test_detector_initialization(self):
        """Test detector initializes without error."""
        detector = PersonDetector(force_cpu=True)
        assert detector is not None
        assert detector.device == "cpu"

    def test_detect_returns_dict_structure(self, sample_image):
        """Test detection returns expected structure."""
        detector = PersonDetector(force_cpu=True)
        result = detector.detect(str(sample_image))

        assert "status" in result
        assert "detections" in result
        assert "image_dimensions" in result
        assert isinstance(result["detections"], list)

    def test_detect_empty_image(self, sample_image):
        """Test detection on image with no people."""
        detector = PersonDetector(force_cpu=True)
        result = detector.detect(str(sample_image))

        # Empty image should have no detections
        assert result["status"] in ["success", "no_detections"]
        assert len(result["detections"]) == 0
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/andrewchoi/dev/home/photodb/.worktrees/age-gender-detection
uv run pytest tests/test_person_detector.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'photodb.utils.person_detector'"

**Step 3: Write minimal implementation**

Create `src/photodb/utils/person_detector.py`:

```python
"""
Person detection using YOLO for face and body detection.
Replaces the previous MTCNN-based face detection.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
from PIL import Image
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import numpy as np


class PersonDetector:
    """Detect faces and bodies in images using YOLO, extract face embeddings."""

    # Class IDs from yolov8x_person_face model
    FACE_CLASS_ID = 1
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        force_cpu: bool = False,
    ):
        """
        Initialize person detector.

        Args:
            model_path: Path to YOLO model weights. Defaults to yolov8x_person_face.pt
            device: Device to use ('cuda', 'mps', 'cpu'). Auto-detects if None.
            force_cpu: Force CPU mode.
        """
        if force_cpu:
            device = "cpu"
        elif device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        # Load YOLO model
        model_path = model_path or os.getenv(
            "DETECTION_MODEL_PATH", "models/yolov8x_person_face.pt"
        )
        self.yolo = YOLO(model_path)

        # Face embedding model (same as before)
        self.embedding_model = InceptionResnetV1(pretrained="vggface2").eval()
        if device in ["cuda", "mps"]:
            self.embedding_model = self.embedding_model.to(device)

        self.min_confidence = float(os.getenv("DETECTION_CONFIDENCE", "0.5"))

    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Detect faces and bodies in an image.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with detection results
        """
        img = Image.open(image_path)
        if img.mode == "RGBA":
            img = img.convert("RGB")

        img_width, img_height = img.size

        # Run YOLO detection
        results = self.yolo(img, conf=self.min_confidence, verbose=False)

        if not results or len(results) == 0:
            return {
                "status": "no_detections",
                "detections": [],
                "image_dimensions": {"width": img_width, "height": img_height},
            }

        # Parse detections
        detections = []
        faces = []
        bodies = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()

                detection = {
                    "class_id": cls_id,
                    "confidence": conf,
                    "bbox": {
                        "x": float(xyxy[0]),
                        "y": float(xyxy[1]),
                        "width": float(xyxy[2] - xyxy[0]),
                        "height": float(xyxy[3] - xyxy[1]),
                    },
                }

                if cls_id == self.FACE_CLASS_ID:
                    faces.append(detection)
                elif cls_id == self.PERSON_CLASS_ID:
                    bodies.append(detection)

        # Match faces to bodies based on overlap
        matched_detections = self._match_faces_to_bodies(faces, bodies)

        return {
            "status": "success" if matched_detections else "no_detections",
            "detections": matched_detections,
            "image_dimensions": {"width": img_width, "height": img_height},
        }

    def _match_faces_to_bodies(
        self, faces: List[Dict], bodies: List[Dict]
    ) -> List[Dict]:
        """Match face detections to body detections based on spatial overlap."""
        matched = []
        used_bodies = set()

        for face in faces:
            best_body = None
            best_iou = 0.0

            for i, body in enumerate(bodies):
                if i in used_bodies:
                    continue
                iou = self._compute_containment(face["bbox"], body["bbox"])
                if iou > best_iou and iou > 0.3:  # Face should be inside body
                    best_iou = iou
                    best_body = (i, body)

            detection = {
                "face": face,
                "body": best_body[1] if best_body else None,
            }
            if best_body:
                used_bodies.add(best_body[0])
            matched.append(detection)

        # Add unmatched bodies as body-only detections
        for i, body in enumerate(bodies):
            if i not in used_bodies:
                matched.append({"face": None, "body": body})

        return matched

    def _compute_containment(self, face_bbox: Dict, body_bbox: Dict) -> float:
        """Compute how much of the face is contained within the body bbox."""
        fx, fy = face_bbox["x"], face_bbox["y"]
        fw, fh = face_bbox["width"], face_bbox["height"]
        bx, by = body_bbox["x"], body_bbox["y"]
        bw, bh = body_bbox["width"], body_bbox["height"]

        # Intersection
        ix = max(fx, bx)
        iy = max(fy, by)
        iw = min(fx + fw, bx + bw) - ix
        ih = min(fy + fh, by + bh) - iy

        if iw <= 0 or ih <= 0:
            return 0.0

        intersection = iw * ih
        face_area = fw * fh
        return intersection / face_area if face_area > 0 else 0.0

    def extract_embedding(self, image: Image.Image, bbox: Dict) -> List[float]:
        """Extract face embedding from a bounding box region."""
        # Crop and resize face
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        face_crop = image.crop((x, y, x + w, y + h))
        face_crop = face_crop.resize((160, 160))

        # Convert to tensor
        face_tensor = torch.from_numpy(np.array(face_crop)).permute(2, 0, 1).float()
        face_tensor = (face_tensor - 127.5) / 128.0  # Normalize
        face_tensor = face_tensor.unsqueeze(0)

        if self.device in ["cuda", "mps"]:
            face_tensor = face_tensor.to(self.device)

        with torch.no_grad():
            embedding = self.embedding_model(face_tensor)

        return embedding.cpu().numpy().tolist()[0]
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_person_detector.py -v
```

Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/photodb/utils/person_detector.py tests/test_person_detector.py
git commit -m "feat: add PersonDetector utility with YOLO face+body detection"
```

---

## Task 6: Create Detection Stage

**Files:**
- Create: `src/photodb/stages/detection.py`
- Create: `tests/test_detection_stage.py`

**Step 1: Write the test**

Create `tests/test_detection_stage.py`:

```python
"""Tests for DetectionStage."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from photodb.stages.detection import DetectionStage
from photodb.database.models import Photo, PersonDetection


class TestDetectionStage:
    @pytest.fixture
    def mock_repository(self):
        repo = MagicMock()
        repo.get_detections_for_photo.return_value = []
        return repo

    @pytest.fixture
    def config(self):
        return {
            "IMG_PATH": "/tmp/photos",
            "DETECTION_MODEL_PATH": "models/yolov8x_person_face.pt",
        }

    def test_stage_name(self, mock_repository, config):
        """Test stage has correct name."""
        with patch("photodb.stages.detection.PersonDetector"):
            stage = DetectionStage(mock_repository, config)
            assert stage.stage_name == "detection"

    def test_process_photo_clears_existing(self, mock_repository, config):
        """Test that reprocessing clears existing detections."""
        with patch("photodb.stages.detection.PersonDetector") as MockDetector:
            mock_detector = MockDetector.return_value
            mock_detector.detect.return_value = {
                "status": "no_detections",
                "detections": [],
                "image_dimensions": {"width": 640, "height": 480},
            }

            stage = DetectionStage(mock_repository, config)

            photo = Photo.create(filename="/test/photo.jpg")
            photo.id = 1
            photo.normalized_path = "2024/01/photo.jpg"

            # Simulate existing detections
            mock_repository.get_detections_for_photo.return_value = [
                PersonDetection.create(photo_id=1)
            ]

            with patch.object(Path, "exists", return_value=True):
                stage.process_photo(photo, Path("/test/photo.jpg"))

            mock_repository.delete_detections_for_photo.assert_called_once_with(1)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_detection_stage.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

Create `src/photodb/stages/detection.py`:

```python
"""
Stage: Person detection (faces and bodies) with embedding extraction.
Replaces the previous FacesStage.
"""

import os
from pathlib import Path
import logging

from PIL import Image

from .base import BaseStage
from ..database.models import Photo, PersonDetection
from ..utils.person_detector import PersonDetector

logger = logging.getLogger(__name__)


class DetectionStage(BaseStage):
    """Stage for detecting faces and bodies, extracting face embeddings."""

    def __init__(self, repository, config: dict):
        super().__init__(repository, config)
        self.stage_name = "detection"  # Override auto-generated name

        force_cpu = os.getenv("DETECTION_FORCE_CPU", "false").lower() == "true"
        model_path = config.get(
            "DETECTION_MODEL_PATH",
            os.getenv("DETECTION_MODEL_PATH", "models/yolov8x_person_face.pt"),
        )

        self.detector = PersonDetector(model_path=model_path, force_cpu=force_cpu)
        logger.debug(f"DetectionStage initialized with device: {self.detector.device}")

    def process_photo(self, photo: Photo, file_path: Path) -> bool:
        """Process person detection for a single photo."""
        try:
            if not photo.normalized_path:
                logger.warning(
                    f"No normalized path for photo {photo.id}, skipping detection"
                )
                return False

            normalized_path = Path(self.config["IMG_PATH"]) / photo.normalized_path
            if not normalized_path.exists():
                logger.error(f"Normalized image not found: {normalized_path}")
                return False

            logger.debug(f"Processing detection for {file_path} -> {normalized_path}")

            # Clear existing detections if reprocessing
            existing = self.repository.get_detections_for_photo(photo.id)
            if existing:
                logger.debug(f"Clearing {len(existing)} existing detections")
                self.repository.delete_detections_for_photo(photo.id)

            # Run detection
            result = self.detector.detect(str(normalized_path))

            if result["status"] == "no_detections":
                logger.debug(f"No people detected in {file_path}")
                return True

            # Get minimum confidence threshold
            min_confidence = float(os.getenv("DETECTION_MIN_CONFIDENCE", "0.5"))

            # Load image for embedding extraction
            img = Image.open(normalized_path)
            if img.mode == "RGBA":
                img = img.convert("RGB")

            # Store detections
            detections_saved = 0
            for det in result["detections"]:
                face_data = det.get("face")
                body_data = det.get("body")

                # Skip if both face and body are below threshold
                face_conf = face_data["confidence"] if face_data else 0
                body_conf = body_data["confidence"] if body_data else 0
                if max(face_conf, body_conf) < min_confidence:
                    continue

                # Create PersonDetection record
                detection = PersonDetection.create(
                    photo_id=photo.id,
                    face_bbox_x=face_data["bbox"]["x"] if face_data else None,
                    face_bbox_y=face_data["bbox"]["y"] if face_data else None,
                    face_bbox_width=face_data["bbox"]["width"] if face_data else None,
                    face_bbox_height=face_data["bbox"]["height"] if face_data else None,
                    face_confidence=face_conf if face_data else None,
                    body_bbox_x=body_data["bbox"]["x"] if body_data else None,
                    body_bbox_y=body_data["bbox"]["y"] if body_data else None,
                    body_bbox_width=body_data["bbox"]["width"] if body_data else None,
                    body_bbox_height=body_data["bbox"]["height"] if body_data else None,
                    body_confidence=body_conf if body_data else None,
                    detector_model="yolov8x_person_face",
                    detector_version="1.0",
                )

                self.repository.create_person_detection(detection)

                # Extract and save face embedding if face detected
                if face_data:
                    embedding = self.detector.extract_embedding(img, face_data["bbox"])
                    self.repository.save_detection_embedding(detection.id, embedding)

                detections_saved += 1

            logger.info(f"Saved {detections_saved} detections for {file_path}")
            return True

        except Exception as e:
            logger.error(f"Detection failed for {file_path}: {e}")
            return False
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_detection_stage.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/photodb/stages/detection.py tests/test_detection_stage.py
git commit -m "feat: add DetectionStage for face+body detection"
```

---

## Task 7: Create AgeGender Stage

**Files:**
- Create: `src/photodb/stages/age_gender.py`
- Create: `tests/test_age_gender_stage.py`

**Step 1: Write the test**

Create `tests/test_age_gender_stage.py`:

```python
"""Tests for AgeGenderStage."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from photodb.stages.age_gender import AgeGenderStage
from photodb.database.models import Photo, PersonDetection


class TestAgeGenderStage:
    @pytest.fixture
    def mock_repository(self):
        return MagicMock()

    @pytest.fixture
    def config(self):
        return {
            "IMG_PATH": "/tmp/photos",
            "MIVOLO_MODEL_PATH": "models/mivolo_imdb.pth.tar",
        }

    def test_stage_name(self, mock_repository, config):
        """Test stage has correct name."""
        with patch("photodb.stages.age_gender.MiVOLOPredictor"):
            stage = AgeGenderStage(mock_repository, config)
            assert stage.stage_name == "age_gender"

    def test_process_photo_updates_detections(self, mock_repository, config):
        """Test that age/gender is updated on detections."""
        with patch("photodb.stages.age_gender.MiVOLOPredictor") as MockPredictor:
            mock_predictor = MockPredictor.return_value
            mock_predictor.predict.return_value = {
                "age": 25.5,
                "gender": "M",
                "gender_confidence": 0.95,
            }

            stage = AgeGenderStage(mock_repository, config)

            photo = Photo.create(filename="/test/photo.jpg")
            photo.id = 1
            photo.normalized_path = "2024/01/photo.jpg"

            detection = PersonDetection.create(photo_id=1)
            detection.id = 10
            detection.face_bbox_x = 100
            detection.face_bbox_y = 100
            detection.face_bbox_width = 50
            detection.face_bbox_height = 50

            mock_repository.get_detections_for_photo.return_value = [detection]

            with patch.object(Path, "exists", return_value=True):
                result = stage.process_photo(photo, Path("/test/photo.jpg"))

            assert result is True
            mock_repository.update_detection_age_gender.assert_called_once()
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_age_gender_stage.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

Create `src/photodb/stages/age_gender.py`:

```python
"""
Stage: Age and gender estimation using MiVOLO.
"""

import os
from pathlib import Path
import logging
from typing import Optional, Dict, Any

from .base import BaseStage
from ..database.models import Photo

logger = logging.getLogger(__name__)


class MiVOLOPredictor:
    """Wrapper for MiVOLO age/gender prediction."""

    def __init__(self, checkpoint_path: str, detector_path: str, device: str = "cuda"):
        """
        Initialize MiVOLO predictor.

        Args:
            checkpoint_path: Path to MiVOLO checkpoint
            detector_path: Path to YOLO detector (for internal use by MiVOLO)
            device: Device to use
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.detector_path = detector_path

        # Import MiVOLO components
        try:
            from mivolo.predictor import Predictor
            self.predictor = Predictor(
                config=None,
                ckpt=checkpoint_path,
                device=device,
                with_persons=True,
            )
            self._available = True
        except ImportError:
            logger.warning("MiVOLO not available, age/gender estimation disabled")
            self._available = False

    def predict(
        self,
        image_path: str,
        face_bbox: Optional[tuple] = None,
        body_bbox: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        """
        Predict age and gender for a detection.

        Args:
            image_path: Path to image
            face_bbox: (x, y, width, height) of face
            body_bbox: (x, y, width, height) of body

        Returns:
            Dict with age, gender, gender_confidence
        """
        if not self._available:
            return {"age": None, "gender": "U", "gender_confidence": 0.0}

        # MiVOLO expects the full image and bboxes
        # It internally handles face+body association
        from PIL import Image
        import numpy as np

        img = Image.open(image_path)
        img_np = np.array(img)

        # Convert bboxes to MiVOLO format (x1, y1, x2, y2)
        faces = []
        bodies = []

        if face_bbox and face_bbox[0] is not None:
            x, y, w, h = face_bbox
            faces.append([x, y, x + w, y + h])

        if body_bbox and body_bbox[0] is not None:
            x, y, w, h = body_bbox
            bodies.append([x, y, x + w, y + h])

        # Run prediction
        result = self.predictor.recognize(
            img_np,
            detected_faces=np.array(faces) if faces else None,
            detected_bodies=np.array(bodies) if bodies else None,
        )

        if result and len(result.ages) > 0:
            age = float(result.ages[0])
            gender = "M" if result.genders[0] == "male" else "F"
            gender_conf = float(result.gender_scores[0]) if hasattr(result, "gender_scores") else 0.9
            return {
                "age": age,
                "gender": gender,
                "gender_confidence": gender_conf,
            }

        return {"age": None, "gender": "U", "gender_confidence": 0.0}


class AgeGenderStage(BaseStage):
    """Stage for estimating age and gender using MiVOLO."""

    def __init__(self, repository, config: dict):
        super().__init__(repository, config)
        self.stage_name = "age_gender"

        force_cpu = os.getenv("MIVOLO_FORCE_CPU", "false").lower() == "true"
        device = "cpu" if force_cpu else ("cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")

        checkpoint_path = config.get(
            "MIVOLO_MODEL_PATH",
            os.getenv("MIVOLO_MODEL_PATH", "models/mivolo_imdb.pth.tar"),
        )
        detector_path = config.get(
            "DETECTION_MODEL_PATH",
            os.getenv("DETECTION_MODEL_PATH", "models/yolov8x_person_face.pt"),
        )

        self.predictor = MiVOLOPredictor(
            checkpoint_path=checkpoint_path,
            detector_path=detector_path,
            device=device,
        )
        logger.debug(f"AgeGenderStage initialized with device: {device}")

    def process_photo(self, photo: Photo, file_path: Path) -> bool:
        """Process age/gender estimation for detections in a photo."""
        try:
            if not photo.normalized_path:
                logger.warning(
                    f"No normalized path for photo {photo.id}, skipping age/gender"
                )
                return False

            normalized_path = Path(self.config["IMG_PATH"]) / photo.normalized_path
            if not normalized_path.exists():
                logger.error(f"Normalized image not found: {normalized_path}")
                return False

            # Get existing detections
            detections = self.repository.get_detections_for_photo(photo.id)
            if not detections:
                logger.debug(f"No detections for photo {photo.id}, skipping age/gender")
                return True

            logger.debug(f"Processing age/gender for {len(detections)} detections")

            updated = 0
            for detection in detections:
                # Build bbox tuples
                face_bbox = None
                if detection.face_bbox_x is not None:
                    face_bbox = (
                        detection.face_bbox_x,
                        detection.face_bbox_y,
                        detection.face_bbox_width,
                        detection.face_bbox_height,
                    )

                body_bbox = None
                if detection.body_bbox_x is not None:
                    body_bbox = (
                        detection.body_bbox_x,
                        detection.body_bbox_y,
                        detection.body_bbox_width,
                        detection.body_bbox_height,
                    )

                # Skip if no face or body
                if face_bbox is None and body_bbox is None:
                    continue

                # Run MiVOLO prediction
                result = self.predictor.predict(
                    image_path=str(normalized_path),
                    face_bbox=face_bbox,
                    body_bbox=body_bbox,
                )

                # Update detection with age/gender
                if result["age"] is not None or result["gender"] != "U":
                    self.repository.update_detection_age_gender(
                        detection_id=detection.id,
                        age_estimate=result["age"],
                        gender=result["gender"],
                        gender_confidence=result["gender_confidence"],
                        mivolo_output=result,
                    )
                    updated += 1

            logger.info(f"Updated age/gender for {updated}/{len(detections)} detections")
            return True

        except Exception as e:
            logger.error(f"Age/gender estimation failed for {file_path}: {e}")
            return False
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_age_gender_stage.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/photodb/stages/age_gender.py tests/test_age_gender_stage.py
git commit -m "feat: add AgeGenderStage for MiVOLO age/gender estimation"
```

---

## Task 8: Update LocalProcessor with New Stages

**Files:**
- Modify: `src/photodb/processors/local_processor.py`
- Modify: `src/photodb/cli_local.py`

**Step 1: Update LocalProcessor imports and stages**

In `local_processor.py`, add imports:

```python
from ..stages.detection import DetectionStage
from ..stages.age_gender import AgeGenderStage
```

Update `__init__` stages dict:

```python
self.stages = {
    "normalize": NormalizeStage(repository, config),
    "metadata": MetadataStage(repository, config),
    "detection": DetectionStage(repository, config),
    "age_gender": AgeGenderStage(repository, config),
    "clustering": ClusteringStage(repository, config),
}
```

Update `_get_stages` method:

```python
def _get_stages(self, stage: str) -> List[str]:
    """Get list of stages to run."""
    if stage == "all":
        return ["normalize", "metadata", "detection", "age_gender", "clustering"]
    elif stage in ["normalize", "metadata", "detection", "age_gender", "clustering"]:
        return [stage]
    # Legacy support
    elif stage == "faces":
        return ["detection"]
    else:
        raise ValueError(f"Invalid stage for LocalProcessor: {stage}")
```

**Step 2: Update CLI options**

In `cli_local.py`, update the stage choice:

```python
@click.option(
    "--stage",
    type=click.Choice(["all", "normalize", "metadata", "detection", "age_gender", "clustering", "faces"]),
    default="all",
    help="Specific stage to run (faces is alias for detection)",
)
```

**Step 3: Commit**

```bash
git add src/photodb/processors/local_processor.py src/photodb/cli_local.py
git commit -m "feat: integrate detection and age_gender stages into pipeline"
```

---

## Task 9: Update Clustering Stage for PersonDetection

**Files:**
- Modify: `src/photodb/stages/clustering.py`

**Step 1: Update imports and references**

Replace `Face` with `PersonDetection` throughout:
- Update imports
- Change method calls from `get_faces_*` to `get_detections_*`
- Update type hints

**Step 2: Update repository method calls**

Replace:
- `repository.get_faces_for_photo` → `repository.get_detections_for_photo`
- `repository.get_face_embedding` → `repository.get_detection_embedding`
- `face.id` → `detection.id`
- `face.bbox_*` → `detection.face_bbox_*`

**Step 3: Commit**

```bash
git add src/photodb/stages/clustering.py
git commit -m "refactor: update clustering stage to use PersonDetection"
```

---

## Task 10: Update Schema File

**Files:**
- Modify: `schema.sql`

**Step 1: Replace face table with person_detection**

Update `schema.sql` to reflect the new schema (for fresh installs):
- Remove `face` table definition
- Add `person_detection` table definition
- Update foreign key references
- Add person table age/gender columns

**Step 2: Commit**

```bash
git add schema.sql
git commit -m "schema: update to use person_detection table"
```

---

## Task 11: Remove Old FacesStage

**Files:**
- Delete: `src/photodb/stages/faces.py`
- Delete: `src/photodb/utils/face_extractor.py`

**Step 1: Remove old files**

```bash
rm src/photodb/stages/faces.py
rm src/photodb/utils/face_extractor.py
```

**Step 2: Update __init__.py if needed**

Remove any imports of removed files.

**Step 3: Commit**

```bash
git add -u
git commit -m "refactor: remove old FacesStage and FaceExtractor (replaced by DetectionStage)"
```

---

## Task 12: Add Person Age/Gender Aggregation

**Files:**
- Create: `src/photodb/utils/age_gender_aggregator.py`
- Create: `tests/test_age_gender_aggregator.py`

**Step 1: Write the test**

Create `tests/test_age_gender_aggregator.py`:

```python
"""Tests for age/gender aggregation."""

import pytest
from datetime import datetime
from statistics import median, stdev

from photodb.utils.age_gender_aggregator import compute_person_age_gender


class TestAgeGenderAggregation:
    def test_compute_birth_year(self):
        """Test birth year computation from photo dates and ages."""
        detections = [
            {"age": 30.0, "photo_year": 2020, "gender": "M", "gender_confidence": 0.9},
            {"age": 32.0, "photo_year": 2022, "gender": "M", "gender_confidence": 0.95},
            {"age": 31.0, "photo_year": 2021, "gender": "M", "gender_confidence": 0.85},
        ]

        result = compute_person_age_gender(detections)

        # All point to birth year ~1990
        assert result["estimated_birth_year"] == 1990
        assert result["gender"] == "M"
        assert result["sample_count"] == 3

    def test_gender_majority(self):
        """Test gender is determined by weighted majority."""
        detections = [
            {"age": 30.0, "photo_year": 2020, "gender": "M", "gender_confidence": 0.6},
            {"age": 30.0, "photo_year": 2020, "gender": "F", "gender_confidence": 0.9},
            {"age": 30.0, "photo_year": 2020, "gender": "F", "gender_confidence": 0.8},
        ]

        result = compute_person_age_gender(detections)

        assert result["gender"] == "F"
```

**Step 2: Write implementation**

Create `src/photodb/utils/age_gender_aggregator.py`:

```python
"""Aggregate age/gender data across detections for a person."""

from typing import Dict, List, Optional, Any
from statistics import median, stdev


def compute_person_age_gender(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregated age/gender statistics from detections.

    Args:
        detections: List of dicts with 'age', 'photo_year', 'gender', 'gender_confidence'

    Returns:
        Dict with estimated_birth_year, birth_year_stddev, gender, gender_confidence, sample_count
    """
    birth_years = []
    gender_weights = {"M": 0.0, "F": 0.0}

    for det in detections:
        age = det.get("age")
        photo_year = det.get("photo_year")
        gender = det.get("gender")
        gender_conf = det.get("gender_confidence", 1.0)

        if age is not None and photo_year is not None:
            birth_years.append(int(photo_year - age))

        if gender in ("M", "F"):
            gender_weights[gender] += gender_conf

    result = {
        "estimated_birth_year": None,
        "birth_year_stddev": None,
        "gender": None,
        "gender_confidence": None,
        "sample_count": len(birth_years),
    }

    if birth_years:
        result["estimated_birth_year"] = int(median(birth_years))
        if len(birth_years) > 1:
            result["birth_year_stddev"] = round(stdev(birth_years), 2)

    total_weight = gender_weights["M"] + gender_weights["F"]
    if total_weight > 0:
        if gender_weights["M"] > gender_weights["F"]:
            result["gender"] = "M"
            result["gender_confidence"] = round(gender_weights["M"] / total_weight, 3)
        else:
            result["gender"] = "F"
            result["gender_confidence"] = round(gender_weights["F"] / total_weight, 3)

    return result
```

**Step 3: Commit**

```bash
git add src/photodb/utils/age_gender_aggregator.py tests/test_age_gender_aggregator.py
git commit -m "feat: add age/gender aggregation utility"
```

---

## Task 13: Download Model Files

**Files:**
- Create: `scripts/download_models.sh`

**Step 1: Create download script**

Create `scripts/download_models.sh`:

```bash
#!/bin/bash
# Download required model files for detection and age/gender estimation

set -e

MODELS_DIR="${1:-models}"
mkdir -p "$MODELS_DIR"

echo "Downloading models to $MODELS_DIR..."

# YOLOv8x person_face model (from MiVOLO repo)
if [ ! -f "$MODELS_DIR/yolov8x_person_face.pt" ]; then
    echo "Downloading yolov8x_person_face.pt..."
    wget -O "$MODELS_DIR/yolov8x_person_face.pt" \
        "https://github.com/WildChlamydia/MiVOLO/releases/download/v1.0/yolov8x_person_face.pt"
fi

# MiVOLO model
if [ ! -f "$MODELS_DIR/mivolo_imdb.pth.tar" ]; then
    echo "Downloading mivolo_imdb.pth.tar..."
    wget -O "$MODELS_DIR/mivolo_imdb.pth.tar" \
        "https://github.com/WildChlamydia/MiVOLO/releases/download/v1.0/mivolo_imdb.pth.tar"
fi

echo "Done! Models downloaded to $MODELS_DIR"
ls -lh "$MODELS_DIR"
```

**Step 2: Make executable and commit**

```bash
chmod +x scripts/download_models.sh
git add scripts/download_models.sh
git commit -m "scripts: add model download script"
```

---

## Task 14: Update CLAUDE.md and Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/plans/2026-02-01-age-gender-body-detection-design.md` (mark as implemented)

**Step 1: Update CLAUDE.md**

Add new configuration options and stage documentation.

**Step 2: Commit**

```bash
git add CLAUDE.md docs/
git commit -m "docs: update documentation for age/gender detection"
```

---

## Task 15: Run Full Test Suite

**Step 1: Run all tests**

```bash
uv run pytest tests/ -v --ignore=tests/test_changes.py
```

Expected: All new tests pass, existing tests pass or have known pre-existing failures.

**Step 2: Run linting**

```bash
uv run ruff check --fix
uv run ruff format
```

**Step 3: Commit any fixes**

```bash
git add -u
git commit -m "fix: linting and formatting"
```

---

## Summary

After completing all tasks, the pipeline will have:

1. **New `person_detection` table** replacing `face` table
2. **New `detection` stage** using YOLO for face+body detection
3. **New `age_gender` stage** using MiVOLO
4. **Person-level age/gender aggregation** in `person` table
5. **Migration path** from existing face data
6. **Model download script** for required weights

Run the migration on your database before testing:

```bash
psql $DATABASE_URL -f migrations/005_add_person_detection.sql
```
