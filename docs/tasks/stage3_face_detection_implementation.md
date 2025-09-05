# Stage 3: Face Detection Implementation Plan

## Overview

Add face detection as Stage 3 to the local processing pipeline using the existing `FaceExtractor` utility from `src/photodb/utils/face_extractor.py`.

## Database Schema

Using existing schema from `schema.sql`:

- **face** table: Stores detected faces with bounding boxes
  - id (text, UUID)
  - photo_id (text, references photos)
  - bbox_x, bbox_y, bbox_width, bbox_height (real)
  - person_id (text, nullable reference to person)
  - confidence (decimal 0.00-1.00)

- **face_embedding** table: Stores face embeddings separately
  - face_id (text, references face)
  - embedding (vector(512) using pgvector extension)

The schema already has proper indexes and foreign key constraints in place.

## Implementation Steps

### 1. Create Face Detection Stage (`src/photodb/stages/faces.py`)

```python
from pathlib import Path
from typing import List
from datetime import datetime

from .base import BaseStage
from ..database.models import Face
from ..utils.face_extractor import FaceExtractor

class FacesStage(BaseStage):
    """Stage for detecting faces and extracting embeddings."""
    
    def __init__(self, repository, config: dict):
        super().__init__("faces", repository, config)
        self.face_extractor = FaceExtractor()  # Auto-detects MPS/CUDA/CPU
    
    def should_process(self, file_path: Path, force: bool = False) -> bool:
        # Get photo record
        photo = self.repository.get_photo_by_filename(str(file_path))
        if not photo:
            return False
            
        # Check if faces have been processed
        existing_faces = self.repository.get_faces_for_photo(photo.id)
        return force or len(existing_faces) == 0
    
    def process(self, file_path: Path) -> None:
        # Get photo record
        photo = self.repository.get_photo_by_filename(str(file_path))
        if not photo:
            raise ValueError(f"Photo not found for {file_path}")
        
        # Clear existing faces if reprocessing
        self.repository.delete_faces_for_photo(photo.id)
        
        # Extract faces using normalized image
        normalized_path = Path(self.config["IMG_PATH"]) / photo.normalized_path
        result = self.face_extractor.extract_from_image(str(normalized_path))
        
        if result["status"] == "no_faces_detected":
            # No faces found, but mark stage as completed
            return
        
        # Store detected faces
        for face_data in result["faces"]:
            bbox = face_data["bbox"]
            
            # Create Face record using existing schema
            face = Face.create(
                photo_id=photo.id,
                bbox_x=bbox["x1"],
                bbox_y=bbox["y1"], 
                bbox_width=bbox["width"],
                bbox_height=bbox["height"],
                confidence=face_data["confidence"]
            )
            
            # Save face record
            self.repository.create_face(face)
            
            # Save embedding separately using pgvector
            self.repository.save_face_embedding(face.id, face_data["embedding"])
```

### 2. Update `src/photodb/cli_local.py`

#### Changes needed:
- Add "faces" to stage choices in CLI options (line 22)
- Update help text to mention face detection

```python
@click.option(
    "--stage",
    type=click.Choice(["all", "normalize", "metadata", "faces"]),  # Add "faces"
    default="all",
    help="Specific stage to run",
)
```

### 3. Update `src/photodb/processors/local_processor.py`

#### Changes needed:

##### A. Import the new stage (around line 11)
```python
from ..stages.faces import FacesStage
```

##### B. Add faces stage to stages dict in `__init__` (line 42-45)
```python
self.stages = {
    "normalize": NormalizeStage(repository, config),
    "metadata": MetadataStage(repository, config),
    "faces": FacesStage(repository, config),  # Add this
}
```

##### C. Update `_get_stages` method (lines 47-54)
```python
def _get_stages(self, stage: str) -> List[str]:
    """Get list of stages to run (including face detection)."""
    if stage == "all":
        return ["normalize", "metadata", "faces"]  # Add "faces"
    elif stage in ["normalize", "metadata", "faces"]:  # Add "faces"
        return [stage]
    else:
        raise ValueError(f"Invalid stage for LocalProcessor: {stage}")
```

##### D. Update pooled stages creation (lines 166-174)
```python
pooled_stages = {
    "normalize": NormalizeStage(pooled_repo, self.config)
    if "normalize" in stages
    else None,
    "metadata": MetadataStage(pooled_repo, self.config)
    if "metadata" in stages
    else None,
    "faces": FacesStage(pooled_repo, self.config)  # Add this
    if "faces" in stages
    else None,
}
```

### 4. Update Database Repository (`src/photodb/database/repository.py`)

Add method for face embeddings (face CRUD methods already exist):

```python
def save_face_embedding(self, face_id: str, embedding: List[float]) -> None:
    """Save face embedding using pgvector."""
    with self.pool.transaction() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """INSERT INTO face_embedding (face_id, embedding)
                   VALUES (%s, %s)
                   ON CONFLICT (face_id) 
                   DO UPDATE SET embedding = EXCLUDED.embedding""",
                (face_id, embedding)
            )

def get_face_embedding(self, face_id: str) -> Optional[List[float]]:
    """Get face embedding by face ID."""
    with self.pool.get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT embedding FROM face_embedding WHERE face_id = %s",
                (face_id,)
            )
            row = cursor.fetchone()
            return list(row[0]) if row else None

def find_similar_faces(self, query_embedding: List[float], threshold: float = 0.6, limit: int = 10) -> List[tuple]:
    """Find similar faces using pgvector cosine similarity."""
    with self.pool.get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """SELECT f.id, f.photo_id, f.confidence, 1 - (fe.embedding <=> %s) as similarity
                   FROM face f
                   JOIN face_embedding fe ON f.id = fe.face_id
                   WHERE 1 - (fe.embedding <=> %s) >= %s
                   ORDER BY similarity DESC
                   LIMIT %s""",
                (query_embedding, query_embedding, threshold, limit)
            )
            return cursor.fetchall()
```

## Usage Examples

After implementation:

```bash
# Process all stages including face detection
uv run process-local /path/to/photos --parallel 100

# Process only face detection stage
uv run process-local /path/to/photos --stage faces --parallel 100

# Force reprocess faces
uv run process-local /path/to/photos --stage faces --force

# Dry run to see what would be processed
uv run process-local /path/to/photos --stage faces --dry-run
```

## Testing Plan

1. **Unit tests** for `FacesStage` class
2. **Integration tests** with sample photos containing:
   - Single face
   - Multiple faces
   - No faces
   - Edge cases (partial faces, profiles)
3. **Performance testing** with parallel processing
4. **Database tests** for face data storage/retrieval

## Performance Considerations

- Face detection is CPU/GPU intensive
- Use connection pooling for parallel processing
- Consider batch processing for GPU operations
- **MPS Issues**: Some PyTorch operations may fail on Apple Silicon MPS due to incomplete adaptive pooling support
- **Automatic fallback**: System automatically falls back to CPU when MPS errors occur
- **CPU performance**: Apple Silicon CPUs are very efficient for face detection tasks
- Default parallel workers should handle face extraction well

### Handling MPS Issues

If you encounter "Adaptive pool MPS" errors:

1. **Set environment variable** (recommended):
   ```bash
   export FACE_DETECTION_FORCE_CPU=true
   uv run process-local /path/to/photos --stage faces
   ```

2. **Automatic fallback**: The system will automatically detect MPS errors and fall back to CPU processing

3. **For consistent performance**:
   ```bash
   FACE_DETECTION_FORCE_CPU=true uv run process-local --stage faces --parallel 50
   ```

## Key Changes from Original Schema Proposal

1. **Using existing face/face_embedding tables** instead of creating new ones
2. **Face coordinates stored as separate columns** (bbox_x, bbox_y, bbox_width, bbox_height) instead of JSONB
3. **Embeddings stored in separate table** with pgvector for efficient similarity search
4. **Person association** supported via person_id column (can be null initially)
5. **Repository methods already exist** for basic face operations

## Dependencies

Already installed:
- `facenet-pytorch` (includes MTCNN and InceptionResnetV1)
- `torch` (with MPS/CUDA support)
- `PIL` (image handling)
- `numpy` (array operations)

The database already has the pgvector extension enabled in the schema.

## Migration Notes

For existing photos in the database:

1. Run `process-local --stage faces` to add face detection to already processed photos
2. Use `--force` flag to reprocess if face detection parameters change
3. Face detection stage runs after metadata extraction, so normalized photos must exist

## Future Enhancements

- Face clustering/grouping across photos
- Face recognition/identification
- Search by similar faces
- Face-based photo organization
- Integration with web UI for face browsing