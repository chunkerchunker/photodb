from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
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