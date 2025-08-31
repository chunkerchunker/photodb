"""Pydantic models for structured photo analysis using the schema from analyze_photo.md"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class QualityInfo(BaseModel):
    aesthetic_score: float = Field(ge=0.0, le=1.0)
    notes: Optional[str] = None


class TimeHypothesis(BaseModel):
    value: Literal["morning", "afternoon", "evening", "night", "indoors-unclear"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str


class DateEstimate(BaseModel):
    value: Optional[str] = None
    lower: Optional[str] = None
    upper: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: Optional[str] = None


class TimeInfo(BaseModel):
    from_exif: Optional[str] = None
    season: Literal["winter", "spring", "summer", "autumn", "unknown"] = "unknown"
    time_of_day_hypotheses: List[TimeHypothesis] = Field(default_factory=list)
    date_estimate: DateEstimate = Field(default_factory=lambda: DateEstimate(confidence=0))


class LocationHypothesis(BaseModel):
    value: str
    granularity: Literal["room", "building", "street", "city", "region", "country", "landmark"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str


class LocationInfo(BaseModel):
    environment: Literal["indoor", "outdoor", "vehicle-interior", "unknown"] = "unknown"
    hypotheses: List[LocationHypothesis] = Field(default_factory=list)


class Expression(BaseModel):
    value: Literal["smile", "neutral", "serious", "surprised", "eyes-closed", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)


class Face(BaseModel):
    bbox: List[float] = Field(min_length=4, max_length=4)  # [x, y, w, h] normalized
    expression: Expression
    attributes: List[str] = Field(default_factory=list)


class PeopleInfo(BaseModel):
    count: int = 0
    faces: List[Face] = Field(default_factory=list)


class ActivityVerb(BaseModel):
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str


class EventHypothesis(BaseModel):
    type: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str


class ActivitiesInfo(BaseModel):
    verbs: List[ActivityVerb] = Field(default_factory=list)
    event_hypotheses: List[EventHypothesis] = Field(default_factory=list)


class ObjectItem(BaseModel):
    label: str
    bbox: List[float] = Field(min_length=4, max_length=4)  # [x, y, w, h] normalized
    significance: Literal["foreground", "background", "decor", "prop", "unknown"] = "unknown"


class ObjectsInfo(BaseModel):
    items: List[ObjectItem] = Field(default_factory=list)


class TextLine(BaseModel):
    text: str
    bbox: List[float] = Field(min_length=4, max_length=4)  # [x, y, w, h] normalized
    lang: str = "en"


class TextInImage(BaseModel):
    full_text: str = ""
    lines: List[TextLine] = Field(default_factory=list)


class PhotoAnalysisResponse(BaseModel):
    """Complete photo analysis response matching the schema from analyze_photo.md"""

    description: str
    quality: QualityInfo
    time: TimeInfo
    location: LocationInfo
    people: PeopleInfo
    activities: ActivitiesInfo
    objects: ObjectsInfo
    text_in_image: TextInImage
    tags: List[str] = Field(default_factory=list)
