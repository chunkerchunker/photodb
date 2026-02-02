# Age/Gender Detection with MiVOLO and Body Detection

**Date:** 2026-02-01
**Status:** Draft
**Goal:** Enhance the photo processing pipeline to generate age and gender data using MiVOLO, augmenting existing face detection with body detection.

## Overview

This design adds age/gender estimation to the PhotoDB pipeline using MiVOLO, which requires both face and body detections as input. The existing face-only detection (MTCNN) is replaced with MiVOLO's bundled YOLOv8x detector that detects both faces and bodies simultaneously.

### Use Cases

1. **Photo organization** — Filter photos by age ranges ("photos of kids", "photos with grandparents")
2. **Timeline reconstruction** — Track how people age across the photo collection, estimate photo dates based on apparent ages

## Schema Changes

### New Table: `person_detection`

Replaces the existing `face` table. Models a detected person with optional face and/or body evidence.

```sql
CREATE TABLE person_detection (
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

    -- Clustering (carried over from existing face table)
    person_id BIGINT REFERENCES person(id) ON DELETE SET NULL,
    cluster_status TEXT,
    cluster_id BIGINT,
    cluster_confidence REAL,

    -- Detector metadata
    detector_model TEXT,
    detector_version TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX person_detection_photo_id_idx ON person_detection(photo_id);
CREATE INDEX person_detection_person_id_idx ON person_detection(person_id);
CREATE INDEX person_detection_cluster_id_idx ON person_detection(cluster_id);
CREATE INDEX person_detection_age_idx ON person_detection(age_estimate) WHERE age_estimate IS NOT NULL;
CREATE INDEX person_detection_gender_idx ON person_detection(gender) WHERE gender IS NOT NULL;
```

### Person Table Additions

Add aggregated age/gender statistics:

```sql
ALTER TABLE person ADD COLUMN estimated_birth_year INT;
ALTER TABLE person ADD COLUMN birth_year_stddev REAL;
ALTER TABLE person ADD COLUMN gender CHAR(1) CHECK (gender IN ('M', 'F', 'U'));
ALTER TABLE person ADD COLUMN gender_confidence REAL;
ALTER TABLE person ADD COLUMN age_gender_sample_count INT DEFAULT 0;
ALTER TABLE person ADD COLUMN age_gender_updated_at TIMESTAMPTZ;
```

### Face Embedding Table Update

Update foreign key to reference new table:

```sql
ALTER TABLE face_embedding DROP CONSTRAINT face_embedding_face_id_fkey;
ALTER TABLE face_embedding RENAME COLUMN face_id TO person_detection_id;
ALTER TABLE face_embedding ADD CONSTRAINT face_embedding_pd_fkey
    FOREIGN KEY (person_detection_id) REFERENCES person_detection(id) ON DELETE CASCADE;
```

## Migration Strategy

1. Create `person_detection` table
2. Migrate existing face data:
   ```sql
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
   ```
3. Update `face_embedding` foreign key
4. Update `cluster` table column names (`representative_face_id` → `representative_detection_id`, etc.)
5. Update constraint tables (`must_link`, `cannot_link`, `face_match_candidate`)
6. Drop old `face` table
7. Reset sequence: `SELECT setval('person_detection_id_seq', (SELECT MAX(id) FROM person_detection));`

## Pipeline Changes

### New Stage Structure

```
normalize → metadata → detection → age_gender → enrich (LLM) → clustering
```

Two new stages replace the existing `faces` stage:

1. **`detection`** — Runs YOLO + FaceNet
2. **`age_gender`** — Runs MiVOLO

### Detection Stage

```python
class DetectionStage(BaseStage):
    """Stage for detecting faces and bodies, extracting face embeddings."""

    def __init__(self, repository, config):
        super().__init__(repository, config)
        self.detector = YOLO(config.get("DETECTION_MODEL_PATH", "models/yolov8x_person_face.pt"))
        self.embedding_model = InceptionResnetV1(pretrained="vggface2").eval()
        # Device handling similar to existing FacesStage

    def process_photo(self, photo: Photo, file_path: Path) -> bool:
        # Load image
        # Run detector → face + body bboxes
        # For each detection:
        #   - Create PersonDetection record
        #   - If face detected, extract embedding with FaceNet
        #   - Save to database
        pass
```

### Age/Gender Stage

```python
class AgeGenderStage(BaseStage):
    """Stage for estimating age and gender using MiVOLO."""

    def __init__(self, repository, config):
        super().__init__(repository, config)
        self.mivolo = MiVOLOPredictor(
            checkpoint=config.get("MIVOLO_MODEL_PATH", "models/mivolo_imdb.pth.tar")
        )

    def process_photo(self, photo: Photo, file_path: Path) -> bool:
        detections = self.repository.get_detections_for_photo(photo.id)
        for det in detections:
            result = self.mivolo.predict(
                image=file_path,
                face_bbox=(det.face_bbox_x, det.face_bbox_y,
                          det.face_bbox_width, det.face_bbox_height),
                body_bbox=(det.body_bbox_x, det.body_bbox_y,
                          det.body_bbox_width, det.body_bbox_height)
            )
            self.repository.update_detection_age_gender(
                det.id,
                age_estimate=result.age,
                gender=result.gender,  # 'M', 'F', or 'U'
                gender_confidence=result.gender_confidence,
                mivolo_output=result.to_dict()
            )
        return True
```

## Dependencies

### Python Packages

```bash
uv add ultralytics
uv add git+https://github.com/WildChlamydia/MiVOLO.git
# PyTorch 1.13+ (existing dependency)
# facenet-pytorch (existing - still needed for embeddings)
```

### Model Files

Download to `models/` directory:

| File | Size | Purpose |
|------|------|---------|
| `yolov8x_person_face.pt` | ~130MB | Face + body detector |
| `mivolo_imdb.pth.tar` | ~90MB | Age/gender estimation |

### Configuration

```bash
# .env additions
DETECTION_MODEL_PATH=models/yolov8x_person_face.pt
MIVOLO_MODEL_PATH=models/mivolo_imdb.pth.tar
DETECTION_CONFIDENCE=0.5
DETECTION_DEVICE=cuda:0  # or 'mps', 'cpu'
```

## Person-Level Aggregation

When clusters are verified/updated, compute aggregated statistics:

```python
def update_person_age_gender(person_id: int):
    detections = repo.get_detections_for_person(person_id)

    birth_years = []
    genders = {'M': 0, 'F': 0}

    for det in detections:
        if det.age_estimate and det.photo.captured_at:
            photo_year = det.photo.captured_at.year
            birth_years.append(photo_year - det.age_estimate)
        if det.gender in ('M', 'F'):
            genders[det.gender] += det.gender_confidence or 1.0

    if birth_years:
        person.estimated_birth_year = int(median(birth_years))
        person.birth_year_stddev = float(stddev(birth_years)) if len(birth_years) > 1 else None

    if genders['M'] or genders['F']:
        person.gender = 'M' if genders['M'] > genders['F'] else 'F'
        person.gender_confidence = max(genders.values()) / sum(genders.values())

    person.age_gender_sample_count = len(birth_years)
    person.age_gender_updated_at = datetime.now(timezone.utc)
```

## Example Queries

### Find photos of children (ages 5-12)

```sql
SELECT p.* FROM photo p
JOIN person_detection pd ON pd.photo_id = p.id
WHERE pd.age_estimate BETWEEN 5 AND 12;
```

### Find photos of a person when they were young

```sql
SELECT p.* FROM photo p
JOIN person_detection pd ON pd.photo_id = p.id
JOIN person per ON pd.person_id = per.id
WHERE per.id = 42
  AND EXTRACT(YEAR FROM p.captured_at) - per.estimated_birth_year BETWEEN 5 AND 10;
```

### Get age distribution for a person

```sql
SELECT
    pd.age_estimate,
    m.captured_at,
    EXTRACT(YEAR FROM m.captured_at) - pd.age_estimate AS implied_birth_year
FROM person_detection pd
JOIN photo p ON pd.photo_id = p.id
JOIN metadata m ON m.photo_id = p.id
WHERE pd.person_id = 42
  AND pd.age_estimate IS NOT NULL
  AND m.captured_at IS NOT NULL
ORDER BY m.captured_at;
```

## Removed Components

After migration:
- `face` table → replaced by `person_detection`
- MTCNN detector → replaced by YOLOv8x person_face
- `FaceExtractor` class → replaced by new `PersonDetector` class

FaceNet (`InceptionResnetV1`) is retained for embedding extraction (clustering).

## Future Considerations

- **YOLO26 upgrade:** When YOLO26 face+person weights become available, can swap detector without schema changes
- **Re-identification:** Body detections could enable re-id even when faces aren't visible
- **Pose data:** YOLO26-pose could add body pose keypoints for activity detection
