from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List
import json
import uuid


@dataclass
class Photo:
    id: str
    filename: str
    normalized_path: str
    created_at: datetime
    updated_at: datetime
    
    @classmethod
    def create(cls, filename: str, normalized_path: str) -> 'Photo':
        """Create a new photo record."""
        now = datetime.now()
        return cls(
            id=str(uuid.uuid4()),
            filename=filename,
            normalized_path=normalized_path,
            created_at=now,
            updated_at=now
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'filename': self.filename,
            'normalized_path': self.normalized_path,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class Metadata:
    photo_id: str
    captured_at: Optional[datetime]
    latitude: Optional[float]
    longitude: Optional[float]
    extra: Dict[str, Any]
    created_at: datetime
    
    @classmethod
    def create(cls, photo_id: str, **kwargs) -> 'Metadata':
        """Create metadata record from extracted data."""
        return cls(
            photo_id=photo_id,
            captured_at=kwargs.get('captured_at'),
            latitude=kwargs.get('latitude'),
            longitude=kwargs.get('longitude'),
            extra=kwargs.get('extra', {}),
            created_at=datetime.now()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'photo_id': self.photo_id,
            'captured_at': self.captured_at.isoformat() if self.captured_at else None,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'extra': json.dumps(self.extra),
            'created_at': self.created_at.isoformat()
        }


@dataclass
class ProcessingStatus:
    photo_id: str
    stage: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    processed_at: Optional[datetime]
    error_message: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'photo_id': self.photo_id,
            'stage': self.stage,
            'status': self.status,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'error_message': self.error_message
        }


@dataclass
class LLMAnalysis:
    id: str
    photo_id: str
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
    error_message: Optional[str]
    
    @classmethod
    def create(cls, photo_id: str, model_name: str, analysis: Dict[str, Any], **kwargs) -> 'LLMAnalysis':
        """Create LLM analysis record from processed data."""
        return cls(
            id=str(uuid.uuid4()),
            photo_id=photo_id,
            model_name=model_name,
            model_version=kwargs.get('model_version'),
            processed_at=datetime.now(),
            batch_id=kwargs.get('batch_id'),
            analysis=analysis,
            description=kwargs.get('description'),
            objects=kwargs.get('objects'),
            people_count=kwargs.get('people_count'),
            location_description=kwargs.get('location_description'),
            emotional_tone=kwargs.get('emotional_tone'),
            confidence_score=kwargs.get('confidence_score'),
            processing_duration_ms=kwargs.get('processing_duration_ms'),
            error_message=kwargs.get('error_message')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'photo_id': self.photo_id,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'processed_at': self.processed_at.isoformat(),
            'batch_id': self.batch_id,
            'analysis': json.dumps(self.analysis),
            'description': self.description,
            'objects': self.objects,
            'people_count': self.people_count,
            'location_description': self.location_description,
            'emotional_tone': self.emotional_tone,
            'confidence_score': self.confidence_score,
            'processing_duration_ms': self.processing_duration_ms,
            'error_message': self.error_message
        }


@dataclass
class BatchJob:
    id: str
    provider_batch_id: str
    status: str  # 'submitted', 'processing', 'completed', 'failed'
    submitted_at: datetime
    completed_at: Optional[datetime]
    photo_count: int
    processed_count: int
    failed_count: int
    error_message: Optional[str]
    
    @classmethod
    def create(cls, provider_batch_id: str, photo_count: int) -> 'BatchJob':
        """Create new batch job record."""
        return cls(
            id=str(uuid.uuid4()),
            provider_batch_id=provider_batch_id,
            status='submitted',
            submitted_at=datetime.now(),
            completed_at=None,
            photo_count=photo_count,
            processed_count=0,
            failed_count=0,
            error_message=None
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'provider_batch_id': self.provider_batch_id,
            'status': self.status,
            'submitted_at': self.submitted_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'photo_count': self.photo_count,
            'processed_count': self.processed_count,
            'failed_count': self.failed_count,
            'error_message': self.error_message
        }