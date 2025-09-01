"""Pydantic models for structured photo analysis using the schema from analyze_photo.md"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class QualityInfo(BaseModel):
    aesthetic_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    notes: Optional[str] = None


class TimeHypothesis(BaseModel):
    value: Optional[Literal["morning", "afternoon", "evening", "night", "indoors-unclear"]] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    evidence: Optional[str] = None


class DateEstimate(BaseModel):
    value: Optional[str] = None
    lower: Optional[str] = None
    upper: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    evidence: Optional[str] = None


class TimeInfo(BaseModel):
    from_exif: Optional[str] = None
    season: Optional[Literal["winter", "spring", "summer", "autumn", "unknown"]] = "unknown"
    time_of_day_hypotheses: Optional[List[TimeHypothesis]] = Field(default_factory=list)
    date_estimate: Optional[DateEstimate] = None


class LocationHypothesis(BaseModel):
    value: Optional[str] = None
    granularity: Optional[Literal["room", "building", "street", "city", "region", "country", "landmark"]] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    evidence: Optional[str] = None


class LocationInfo(BaseModel):
    environment: Optional[Literal["indoor", "outdoor", "vehicle-interior", "unknown"]] = "unknown"
    hypotheses: Optional[List[LocationHypothesis]] = Field(default_factory=list)


class Expression(BaseModel):
    value: Optional[Literal["smile", "neutral", "serious", "surprised", "eyes-closed", "unknown"]] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class Face(BaseModel):
    bbox: Optional[List[float]] = Field(None, min_length=4, max_length=4)  # [x, y, w, h] normalized
    expression: Optional[Expression] = None
    attributes: Optional[List[str]] = Field(default_factory=list)


class PeopleInfo(BaseModel):
    count: Optional[int] = 0
    faces: Optional[List[Face]] = Field(default_factory=list)


class ActivityVerb(BaseModel):
    value: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    evidence: Optional[str] = None


class EventHypothesis(BaseModel):
    type: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    evidence: Optional[str] = None


class ActivitiesInfo(BaseModel):
    verbs: Optional[List[ActivityVerb]] = Field(default_factory=list)
    event_hypotheses: Optional[List[EventHypothesis]] = Field(default_factory=list)


class ObjectItem(BaseModel):
    label: Optional[str] = None
    bbox: Optional[List[float]] = Field(None, min_length=4, max_length=4)  # [x, y, w, h] normalized
    significance: Optional[Literal["foreground", "background", "decor", "prop", "unknown"]] = "unknown"


class ObjectsInfo(BaseModel):
    items: Optional[List[ObjectItem]] = Field(default_factory=list)


class TextLine(BaseModel):
    text: Optional[str] = None
    bbox: Optional[List[float]] = Field(None, min_length=4, max_length=4)  # [x, y, w, h] normalized
    lang: Optional[str] = "en"


class TextInImage(BaseModel):
    full_text: Optional[str] = None
    lines: Optional[List[TextLine]] = Field(default_factory=list)


class PhotoAnalysisResponse(BaseModel):
    """Complete photo analysis response matching the schema from analyze_photo.md"""

    description: Optional[str] = None
    quality: Optional[QualityInfo] = None
    time: Optional[TimeInfo] = None
    location: Optional[LocationInfo] = None
    people: Optional[PeopleInfo] = None
    activities: Optional[ActivitiesInfo] = None
    objects: Optional[ObjectsInfo] = None
    text_in_image: Optional[TextInImage] = None
    tags: Optional[List[str]] = None
