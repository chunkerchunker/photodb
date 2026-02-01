from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, List
import json


@dataclass
class Photo:
    id: Optional[int]
    filename: str
    normalized_path: str | None
    width: Optional[int]
    height: Optional[int]
    normalized_width: Optional[int]
    normalized_height: Optional[int]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def create(cls, filename: str, normalized_path: Optional[str] = None) -> "Photo":
        """Create a new photo record."""
        now = datetime.now(timezone.utc)
        return cls(
            id=None,  # Will be assigned by database
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
class Metadata:
    photo_id: int
    captured_at: Optional[datetime]
    latitude: Optional[float]
    longitude: Optional[float]
    extra: Dict[str, Any]
    created_at: datetime

    @classmethod
    def create(cls, photo_id: int, **kwargs) -> "Metadata":
        """Create metadata record from extracted data."""
        return cls(
            photo_id=photo_id,
            captured_at=kwargs.get("captured_at"),
            latitude=kwargs.get("latitude"),
            longitude=kwargs.get("longitude"),
            extra=kwargs.get("extra", {}),
            created_at=datetime.now(timezone.utc),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "photo_id": self.photo_id,
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
    name: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def create(cls, name: str) -> "Person":
        """Create a new person record."""
        now = datetime.now(timezone.utc)
        return cls(
            id=None,  # Will be assigned by database
            name=name,
            created_at=now,
            updated_at=now,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class Face:
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
    face_count: int
    face_count_at_last_medoid: int
    representative_face_id: Optional[int]
    centroid: Optional[List[float]]  # 512-dimensional vector
    medoid_face_id: Optional[int]
    person_id: Optional[int]
    verified: bool
    verified_at: Optional[datetime]
    verified_by: Optional[str]
    hidden: bool
    created_at: datetime
    updated_at: datetime

    @classmethod
    def create(
        cls,
        face_count: int = 0,
        representative_face_id: Optional[int] = None,
        centroid: Optional[List[float]] = None,
        medoid_face_id: Optional[int] = None,
        person_id: Optional[int] = None,
    ) -> "Cluster":
        """Create a new cluster record."""
        now = datetime.now(timezone.utc)
        return cls(
            id=None,  # Will be assigned by database
            face_count=face_count,
            face_count_at_last_medoid=face_count,
            representative_face_id=representative_face_id,
            centroid=centroid,
            medoid_face_id=medoid_face_id,
            person_id=person_id,
            verified=False,
            verified_at=None,
            verified_by=None,
            hidden=False,
            created_at=now,
            updated_at=now,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "face_count": self.face_count,
            "face_count_at_last_medoid": self.face_count_at_last_medoid,
            "representative_face_id": self.representative_face_id,
            "centroid": self.centroid,
            "medoid_face_id": self.medoid_face_id,
            "person_id": self.person_id,
            "verified": self.verified,
            "verified_at": self.verified_at,
            "verified_by": self.verified_by,
            "hidden": self.hidden,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
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
