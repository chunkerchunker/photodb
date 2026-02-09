from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, List
import json
import warnings


@dataclass
class Photo:
    id: Optional[int]
    collection_id: int
    filename: str
    normalized_path: str | None
    width: Optional[int]
    height: Optional[int]
    normalized_width: Optional[int]
    normalized_height: Optional[int]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def create(
        cls, filename: str, collection_id: int, normalized_path: Optional[str] = None
    ) -> "Photo":
        """Create a new photo record."""
        now = datetime.now(timezone.utc)
        return cls(
            id=None,  # Will be assigned by database
            collection_id=collection_id,
            filename=filename,
            normalized_path=normalized_path,
            width=None,
            height=None,
            normalized_width=None,
            normalized_height=None,
            created_at=now,
            updated_at=now,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "collection_id": self.collection_id,
            "filename": self.filename,
            "normalized_path": self.normalized_path,
            "width": self.width,
            "height": self.height,
            "normalized_width": self.normalized_width,
            "normalized_height": self.normalized_height,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class Album:
    id: Optional[int]
    collection_id: int
    name: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def create(cls, collection_id: int, name: str) -> "Album":
        """Create a new album record."""
        now = datetime.now(timezone.utc)
        return cls(
            id=None,  # Will be assigned by database
            collection_id=collection_id,
            name=name,
            created_at=now,
            updated_at=now,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "collection_id": self.collection_id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class Metadata:
    photo_id: int
    collection_id: int
    captured_at: Optional[datetime]
    latitude: Optional[float]
    longitude: Optional[float]
    extra: Dict[str, Any]
    created_at: datetime

    @classmethod
    def create(cls, photo_id: int, collection_id: int, **kwargs) -> "Metadata":
        """Create metadata record from extracted data."""
        return cls(
            photo_id=photo_id,
            collection_id=collection_id,
            captured_at=kwargs.get("captured_at"),
            latitude=kwargs.get("latitude"),
            longitude=kwargs.get("longitude"),
            extra=kwargs.get("extra", {}),
            created_at=datetime.now(timezone.utc),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "photo_id": self.photo_id,
            "collection_id": self.collection_id,
            "captured_at": self.captured_at,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "extra": json.dumps(self.extra),
            "created_at": self.created_at,
        }


class Status(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingStatus:
    photo_id: int
    stage: str
    status: Status
    processed_at: Optional[datetime]
    error_message: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "photo_id": self.photo_id,
            "stage": self.stage,
            "status": self.status,
            "processed_at": self.processed_at,
            "error_message": self.error_message,
        }


@dataclass
class LLMAnalysis:
    id: Optional[int]
    photo_id: int
    model_name: str
    model_version: Optional[str]
    processed_at: datetime
    batch_id: Optional[str]
    analysis: Dict[str, Any]
    description: Optional[str]
    objects: Optional[List[str]]
    people_count: Optional[int]
    location_description: Optional[str]
    emotional_tone: Optional[str]
    confidence_score: Optional[float]
    processing_duration_ms: Optional[int]

    # Token usage tracking (per photo)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_creation_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None

    error_message: Optional[str] = None

    @classmethod
    def create(
        cls, photo_id: int, model_name: str, analysis: Dict[str, Any], **kwargs
    ) -> "LLMAnalysis":
        """Create LLM analysis record from processed data."""
        return cls(
            id=None,  # Will be assigned by database
            photo_id=photo_id,
            model_name=model_name,
            model_version=kwargs.get("model_version"),
            processed_at=datetime.now(timezone.utc),
            batch_id=kwargs.get("batch_id"),
            analysis=analysis,
            description=kwargs.get("description"),
            objects=kwargs.get("objects"),
            people_count=kwargs.get("people_count"),
            location_description=kwargs.get("location_description"),
            emotional_tone=kwargs.get("emotional_tone"),
            confidence_score=kwargs.get("confidence_score"),
            processing_duration_ms=kwargs.get("processing_duration_ms"),
            input_tokens=kwargs.get("input_tokens"),
            output_tokens=kwargs.get("output_tokens"),
            cache_creation_tokens=kwargs.get("cache_creation_tokens"),
            cache_read_tokens=kwargs.get("cache_read_tokens"),
            error_message=kwargs.get("error_message"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "photo_id": self.photo_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "processed_at": self.processed_at,
            "batch_id": self.batch_id,
            "analysis": json.dumps(self.analysis),
            "description": self.description,
            "objects": self.objects,
            "people_count": self.people_count,
            "location_description": self.location_description,
            "emotional_tone": self.emotional_tone,
            "confidence_score": self.confidence_score,
            "processing_duration_ms": self.processing_duration_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "error_message": self.error_message,
        }


@dataclass
class Person:
    id: Optional[int]
    collection_id: int
    first_name: str
    last_name: Optional[str]
    created_at: datetime
    updated_at: datetime
    # Age/gender aggregation fields
    estimated_birth_year: Optional[int] = None
    birth_year_stddev: Optional[float] = None
    gender: Optional[str] = None  # 'M', 'F', 'U'
    gender_confidence: Optional[float] = None
    age_gender_sample_count: int = 0
    age_gender_updated_at: Optional[datetime] = None

    @classmethod
    def create(
        cls,
        collection_id: int,
        first_name: str,
        last_name: Optional[str] = None,
        estimated_birth_year: Optional[int] = None,
        birth_year_stddev: Optional[float] = None,
        gender: Optional[str] = None,
        gender_confidence: Optional[float] = None,
        age_gender_sample_count: int = 0,
        age_gender_updated_at: Optional[datetime] = None,
    ) -> "Person":
        """Create a new person record."""
        now = datetime.now(timezone.utc)
        return cls(
            id=None,  # Will be assigned by database
            collection_id=collection_id,
            first_name=first_name,
            last_name=last_name,
            created_at=now,
            updated_at=now,
            estimated_birth_year=estimated_birth_year,
            birth_year_stddev=birth_year_stddev,
            gender=gender,
            gender_confidence=gender_confidence,
            age_gender_sample_count=age_gender_sample_count,
            age_gender_updated_at=age_gender_updated_at,
        )

    @property
    def full_name(self) -> str:
        """Get the full name (first + last)."""
        if self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.first_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "collection_id": self.collection_id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": self.full_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "estimated_birth_year": self.estimated_birth_year,
            "birth_year_stddev": self.birth_year_stddev,
            "gender": self.gender,
            "gender_confidence": self.gender_confidence,
            "age_gender_sample_count": self.age_gender_sample_count,
            "age_gender_updated_at": self.age_gender_updated_at,
        }


@dataclass
class PersonDetection:
    """Represents a detected person (face and/or body) in a photo with age/gender estimates."""

    id: Optional[int]
    photo_id: int
    collection_id: int
    # Face bounding box (optional - may have body only)
    face_bbox_x: Optional[float] = None
    face_bbox_y: Optional[float] = None
    face_bbox_width: Optional[float] = None
    face_bbox_height: Optional[float] = None
    face_confidence: Optional[float] = None
    # Body bounding box (optional - may have face only)
    body_bbox_x: Optional[float] = None
    body_bbox_y: Optional[float] = None
    body_bbox_width: Optional[float] = None
    body_bbox_height: Optional[float] = None
    body_confidence: Optional[float] = None
    # Age/gender estimates
    age_estimate: Optional[float] = None
    gender: Optional[str] = None  # 'M', 'F', 'U'
    gender_confidence: Optional[float] = None
    # Raw model output for debugging/analysis
    mivolo_output: Optional[Dict[str, Any]] = None
    # Person association (via clustering or manual assignment)
    person_id: Optional[int] = None
    # Clustering fields
    cluster_status: Optional[str] = None  # 'auto', 'pending', 'manual', 'unassigned', 'constrained'
    cluster_id: Optional[int] = None
    cluster_confidence: Optional[float] = None
    unassigned_since: Optional[datetime] = None  # When added to unassigned pool
    # Detector metadata
    detector_model: Optional[str] = None
    detector_version: Optional[str] = None
    created_at: Optional[datetime] = None

    @classmethod
    def create(
        cls,
        photo_id: int,
        collection_id: int,
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
        age_estimate: Optional[float] = None,
        gender: Optional[str] = None,
        gender_confidence: Optional[float] = None,
        mivolo_output: Optional[Dict[str, Any]] = None,
        person_id: Optional[int] = None,
        cluster_status: Optional[str] = None,
        cluster_id: Optional[int] = None,
        cluster_confidence: Optional[float] = None,
        detector_model: Optional[str] = None,
        detector_version: Optional[str] = None,
    ) -> "PersonDetection":
        """Create a new person detection record."""
        return cls(
            id=None,  # Will be assigned by database
            photo_id=photo_id,
            collection_id=collection_id,
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
            age_estimate=age_estimate,
            gender=gender,
            gender_confidence=gender_confidence,
            mivolo_output=mivolo_output,
            person_id=person_id,
            cluster_status=cluster_status,
            cluster_id=cluster_id,
            cluster_confidence=cluster_confidence,
            detector_model=detector_model,
            detector_version=detector_version,
            created_at=datetime.now(timezone.utc),
        )

    def has_face(self) -> bool:
        """Check if this detection includes a face bounding box."""
        return (
            self.face_bbox_x is not None
            and self.face_bbox_y is not None
            and self.face_bbox_width is not None
            and self.face_bbox_height is not None
        )

    def has_body(self) -> bool:
        """Check if this detection includes a body bounding box."""
        return (
            self.body_bbox_x is not None
            and self.body_bbox_y is not None
            and self.body_bbox_width is not None
            and self.body_bbox_height is not None
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "photo_id": self.photo_id,
            "collection_id": self.collection_id,
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
            "mivolo_output": json.dumps(self.mivolo_output) if self.mivolo_output else None,
            "person_id": self.person_id,
            "cluster_status": self.cluster_status,
            "cluster_id": self.cluster_id,
            "cluster_confidence": self.cluster_confidence,
            "detector_model": self.detector_model,
            "detector_version": self.detector_version,
            "created_at": self.created_at,
        }


@dataclass
class Face:
    """
    DEPRECATED: Use PersonDetection instead.

    This class is maintained for backward compatibility during migration.
    It will be removed in a future version.
    """

    id: Optional[int]
    photo_id: int
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    person_id: Optional[int]
    confidence: float
    # Clustering fields
    cluster_status: Optional[str] = None  # 'auto', 'pending', 'manual'
    cluster_id: Optional[int] = None
    cluster_confidence: Optional[float] = None

    def __post_init__(self):
        warnings.warn(
            "Face is deprecated and will be removed in a future version. "
            "Use PersonDetection instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    @classmethod
    def create(
        cls,
        photo_id: int,
        bbox_x: float,
        bbox_y: float,
        bbox_width: float,
        bbox_height: float,
        confidence: float,
        person_id: Optional[int] = None,
        cluster_status: Optional[str] = None,
        cluster_id: Optional[int] = None,
        cluster_confidence: Optional[float] = None,
    ) -> "Face":
        """Create a new face detection record."""
        return cls(
            id=None,  # Will be assigned by database
            photo_id=photo_id,
            bbox_x=bbox_x,
            bbox_y=bbox_y,
            bbox_width=bbox_width,
            bbox_height=bbox_height,
            person_id=person_id,
            confidence=confidence,
            cluster_status=cluster_status,
            cluster_id=cluster_id,
            cluster_confidence=cluster_confidence,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "photo_id": self.photo_id,
            "bbox_x": self.bbox_x,
            "bbox_y": self.bbox_y,
            "bbox_width": self.bbox_width,
            "bbox_height": self.bbox_height,
            "person_id": self.person_id,
            "confidence": self.confidence,
            "cluster_status": self.cluster_status,
            "cluster_id": self.cluster_id,
            "cluster_confidence": self.cluster_confidence,
        }


@dataclass
class Cluster:
    id: Optional[int]
    collection_id: int
    face_count: int
    face_count_at_last_medoid: int
    representative_detection_id: Optional[int]
    centroid: Optional[List[float]]  # 512-dimensional vector
    medoid_detection_id: Optional[int]
    person_id: Optional[int]
    verified: bool
    verified_at: Optional[datetime]
    verified_by: Optional[str]
    hidden: bool
    created_at: datetime
    updated_at: datetime
    # HDBSCAN clustering fields
    epsilon: Optional[float] = None  # Per-cluster distance threshold
    core_count: int = 0  # Number of core points in this cluster

    @classmethod
    def create(
        cls,
        collection_id: int,
        face_count: int = 0,
        representative_detection_id: Optional[int] = None,
        centroid: Optional[List[float]] = None,
        medoid_detection_id: Optional[int] = None,
        person_id: Optional[int] = None,
        epsilon: Optional[float] = None,
        core_count: int = 0,
    ) -> "Cluster":
        """Create a new cluster record."""
        now = datetime.now(timezone.utc)
        return cls(
            id=None,  # Will be assigned by database
            collection_id=collection_id,
            face_count=face_count,
            face_count_at_last_medoid=face_count,
            representative_detection_id=representative_detection_id,
            centroid=centroid,
            medoid_detection_id=medoid_detection_id,
            person_id=person_id,
            verified=False,
            verified_at=None,
            verified_by=None,
            hidden=False,
            created_at=now,
            updated_at=now,
            epsilon=epsilon,
            core_count=core_count,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "collection_id": self.collection_id,
            "face_count": self.face_count,
            "face_count_at_last_medoid": self.face_count_at_last_medoid,
            "representative_detection_id": self.representative_detection_id,
            "centroid": self.centroid,
            "medoid_detection_id": self.medoid_detection_id,
            "person_id": self.person_id,
            "verified": self.verified,
            "verified_at": self.verified_at,
            "verified_by": self.verified_by,
            "hidden": self.hidden,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "epsilon": self.epsilon,
            "core_count": self.core_count,
        }


@dataclass
class FaceMatchCandidate:
    candidate_id: Optional[int]
    face_id: int
    cluster_id: int
    similarity: float
    status: str  # 'pending', 'accepted', 'rejected'
    created_at: datetime

    @classmethod
    def create(
        cls,
        face_id: int,
        cluster_id: int,
        similarity: float,
        status: str = "pending",
    ) -> "FaceMatchCandidate":
        """Create a new face match candidate record."""
        return cls(
            candidate_id=None,  # Will be assigned by database
            face_id=face_id,
            cluster_id=cluster_id,
            similarity=similarity,
            status=status,
            created_at=datetime.now(timezone.utc),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "face_id": self.face_id,
            "cluster_id": self.cluster_id,
            "similarity": self.similarity,
            "status": self.status,
            "created_at": self.created_at,
        }


@dataclass
class BatchJob:
    id: Optional[int]
    provider_batch_id: str
    status: str  # 'submitted', 'processing', 'completed', 'failed'
    submitted_at: datetime
    completed_at: Optional[datetime]
    photo_count: int
    processed_count: int
    failed_count: int
    photo_ids: List[int]

    # Token usage tracking
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0

    # Cost tracking (in USD cents)
    estimated_cost_cents: int = 0
    actual_cost_cents: int = 0

    # Additional metadata
    model_name: Optional[str] = None
    batch_discount_applied: bool = True

    error_message: Optional[str] = None

    @classmethod
    def create(
        cls, provider_batch_id: str, photo_ids: List[int], model_name: Optional[str] = None
    ) -> "BatchJob":
        """Create new batch job record."""
        return cls(
            id=None,  # Will be assigned by database
            provider_batch_id=provider_batch_id,
            status="submitted",
            submitted_at=datetime.now(timezone.utc),
            completed_at=None,
            photo_count=len(photo_ids),
            processed_count=0,
            failed_count=0,
            photo_ids=photo_ids,
            model_name=model_name,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "provider_batch_id": self.provider_batch_id,
            "status": self.status,
            "submitted_at": self.submitted_at,
            "completed_at": self.completed_at,
            "photo_count": self.photo_count,
            "processed_count": self.processed_count,
            "failed_count": self.failed_count,
            "photo_ids": self.photo_ids,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cache_creation_tokens": self.total_cache_creation_tokens,
            "total_cache_read_tokens": self.total_cache_read_tokens,
            "estimated_cost_cents": self.estimated_cost_cents,
            "actual_cost_cents": self.actual_cost_cents,
            "model_name": self.model_name,
            "batch_discount_applied": self.batch_discount_applied,
            "error_message": self.error_message,
        }


@dataclass
class PromptCategory:
    """Category for organizing prompts (face_emotion, scene_setting, etc.)."""

    id: Optional[int]
    name: str
    target: str  # 'face' or 'scene'
    selection_mode: str  # 'single' or 'multi'
    min_confidence: float
    max_results: int
    description: Optional[str]
    display_order: int
    is_active: bool
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    @classmethod
    def create(
        cls,
        name: str,
        target: str,
        selection_mode: str = "single",
        min_confidence: float = 0.1,
        max_results: int = 5,
        description: Optional[str] = None,
        display_order: int = 0,
    ) -> "PromptCategory":
        now = datetime.now(timezone.utc)
        return cls(
            id=None,
            name=name,
            target=target,
            selection_mode=selection_mode,
            min_confidence=min_confidence,
            max_results=max_results,
            description=description,
            display_order=display_order,
            is_active=True,
            created_at=now,
            updated_at=now,
        )


@dataclass
class PromptEmbedding:
    """A prompt with precomputed text embedding for zero-shot classification."""

    id: Optional[int]
    category_id: int
    label: str
    prompt_text: str
    embedding: Optional[List[float]]  # 512-dim vector
    model_name: str
    model_version: Optional[str]
    display_name: Optional[str]
    parent_label: Optional[str]
    confidence_boost: float
    metadata: Optional[Dict[str, Any]]
    is_active: bool
    embedding_computed_at: Optional[datetime]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    @classmethod
    def create(
        cls,
        category_id: int,
        label: str,
        prompt_text: str,
        model_name: str,
        embedding: Optional[List[float]] = None,
        display_name: Optional[str] = None,
        parent_label: Optional[str] = None,
        confidence_boost: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "PromptEmbedding":
        now = datetime.now(timezone.utc)
        return cls(
            id=None,
            category_id=category_id,
            label=label,
            prompt_text=prompt_text,
            embedding=embedding,
            model_name=model_name,
            model_version=None,
            display_name=display_name,
            parent_label=parent_label,
            confidence_boost=confidence_boost,
            metadata=metadata,
            is_active=True,
            embedding_computed_at=now if embedding else None,
            created_at=now,
            updated_at=now,
        )


@dataclass
class PhotoTag:
    """A tag assigned to a photo from prompt-based classification."""

    id: Optional[int]
    photo_id: int
    prompt_id: int
    confidence: float
    rank_in_category: Optional[int]
    analysis_output_id: Optional[int]
    created_at: Optional[datetime]

    @classmethod
    def create(
        cls,
        photo_id: int,
        prompt_id: int,
        confidence: float,
        rank_in_category: Optional[int] = None,
        analysis_output_id: Optional[int] = None,
    ) -> "PhotoTag":
        return cls(
            id=None,
            photo_id=photo_id,
            prompt_id=prompt_id,
            confidence=confidence,
            rank_in_category=rank_in_category,
            analysis_output_id=analysis_output_id,
            created_at=datetime.now(timezone.utc),
        )


@dataclass
class DetectionTag:
    """A tag assigned to a face detection from prompt-based classification."""

    id: Optional[int]
    detection_id: int
    prompt_id: int
    confidence: float
    rank_in_category: Optional[int]
    analysis_output_id: Optional[int]
    created_at: Optional[datetime]

    @classmethod
    def create(
        cls,
        detection_id: int,
        prompt_id: int,
        confidence: float,
        rank_in_category: Optional[int] = None,
        analysis_output_id: Optional[int] = None,
    ) -> "DetectionTag":
        return cls(
            id=None,
            detection_id=detection_id,
            prompt_id=prompt_id,
            confidence=confidence,
            rank_in_category=rank_in_category,
            analysis_output_id=analysis_output_id,
            created_at=datetime.now(timezone.utc),
        )


@dataclass
class AnalysisOutput:
    """Model-agnostic storage for raw analysis outputs."""

    id: Optional[int]
    photo_id: int
    model_type: str
    model_name: str
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


@dataclass
class SceneAnalysis:
    """Photo-level scene analysis results."""

    id: Optional[int]
    photo_id: int
    taxonomy_labels: Optional[List[str]]
    taxonomy_confidences: Optional[List[float]]
    taxonomy_output_id: Optional[int]
    mobileclip_output_id: Optional[int]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    @classmethod
    def create(
        cls,
        photo_id: int,
        taxonomy_labels: Optional[List[str]] = None,
        taxonomy_confidences: Optional[List[float]] = None,
        taxonomy_output_id: Optional[int] = None,
        mobileclip_output_id: Optional[int] = None,
    ) -> "SceneAnalysis":
        now = datetime.now(timezone.utc)
        return cls(
            id=None,
            photo_id=photo_id,
            taxonomy_labels=taxonomy_labels,
            taxonomy_confidences=taxonomy_confidences,
            taxonomy_output_id=taxonomy_output_id,
            mobileclip_output_id=mobileclip_output_id,
            created_at=now,
            updated_at=now,
        )
