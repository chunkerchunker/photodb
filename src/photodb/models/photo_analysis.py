"""Pydantic models for structured photo analysis using the schema from analyze_photo.md"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class ExifData(BaseModel):
    datetime_original: Optional[str] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    lens: Optional[str] = None
    focal_length_mm: int = 0
    exposure_time_s: float = 0.0
    iso: int = 0
    flash: Optional[Literal["on", "off"]] = None


class ScanInfo(BaseModel):
    is_scan: bool = False
    notes: Optional[str] = None


class QualityInfo(BaseModel):
    aesthetic_score: float = Field(ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class TechnicalInfo(BaseModel):
    width_px: int = 0
    height_px: int = 0
    orientation: Literal["landscape", "portrait", "square", "unknown"] = "unknown"
    scan: ScanInfo = Field(default_factory=ScanInfo)
    quality: QualityInfo


class ImageInfo(BaseModel):
    id: str
    filename: Optional[str] = None
    exif: ExifData = Field(default_factory=ExifData)
    technical: TechnicalInfo


class SceneInfo(BaseModel):
    type: List[str] = Field(default_factory=list)
    primary_subject: Literal["person", "people", "object", "place", "document", "unknown"] = "unknown"
    short_caption: str
    long_description: str


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
    date_estimate: DateEstimate = Field(default_factory=DateEstimate)


class LocationHypothesis(BaseModel):
    value: str
    granularity: Literal["room", "building", "street", "city", "region", "country", "landmark"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str


class LocationInfo(BaseModel):
    environment: Literal["indoor", "outdoor", "vehicle-interior", "unknown"] = "unknown"
    hypotheses: List[LocationHypothesis] = Field(default_factory=list)


class AgeRange(BaseModel):
    min: int
    max: int
    confidence: float = Field(ge=0.0, le=1.0)


class GenderPresentation(BaseModel):
    value: Literal["masculine", "feminine", "androgynous", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)


class Expression(BaseModel):
    value: Literal["smile", "neutral", "serious", "surprised", "eyes-closed", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)


class RoleHypothesis(BaseModel):
    value: Literal["self", "sibling", "parent", "friend", "teammate", "teacher", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str


class Face(BaseModel):
    bbox: List[float] = Field(min_items=4, max_items=4)  # [x, y, w, h] normalized
    age_range_years: AgeRange
    gender_presentation: GenderPresentation
    expression: Expression
    attributes: List[str] = Field(default_factory=list)
    role_hypotheses: List[RoleHypothesis] = Field(default_factory=list)


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


class BrandHypothesis(BaseModel):
    value: str
    confidence: float = Field(ge=0.0, le=1.0)


class ObjectItem(BaseModel):
    label: str
    bbox: List[float] = Field(min_items=4, max_items=4)  # [x, y, w, h] normalized
    brand_hypotheses: List[BrandHypothesis] = Field(default_factory=list)
    significance: Literal["foreground", "background", "decor", "prop", "unknown"] = "unknown"


class ObjectsInfo(BaseModel):
    items: List[ObjectItem] = Field(default_factory=list)


class TextLine(BaseModel):
    text: str
    bbox: List[float] = Field(min_items=4, max_items=4)  # [x, y, w, h] normalized
    lang: str = "en"


class TextInImage(BaseModel):
    full_text: str = ""
    lines: List[TextLine] = Field(default_factory=list)


class ColorsInfo(BaseModel):
    dominant_hex: List[str] = Field(default_factory=list)
    palette_hex: List[str] = Field(default_factory=list, max_items=5)


class CompositionInfo(BaseModel):
    subject_focus: Literal["single-subject", "multi-subject", "environmental-portrait", "wide-scene", "macro", "unknown"] = "unknown"
    framing: List[str] = Field(default_factory=list)
    camera_view: Literal["eye-level", "high-angle", "low-angle", "overhead", "unknown"] = "unknown"


class AccessibilityInfo(BaseModel):
    alt_text: str
    audio_description: str


class RegionCaption(BaseModel):
    bbox: List[float] = Field(min_items=4, max_items=4)  # [x, y, w, h] normalized
    caption: str


class EmbeddingsInfo(BaseModel):
    caption_for_text_embedding: str
    region_captions: List[RegionCaption] = Field(default_factory=list)


class DedupInfo(BaseModel):
    scene_fingerprint: str
    near_duplicate_hints: List[str] = Field(default_factory=list)


class PhotoAnalysisResponse(BaseModel):
    """Complete photo analysis response matching the schema from analyze_photo.md"""
    
    image: ImageInfo
    scene: SceneInfo
    time: TimeInfo
    location: LocationInfo
    people: PeopleInfo
    activities: ActivitiesInfo
    objects: ObjectsInfo
    text_in_image: TextInImage
    colors: ColorsInfo
    composition: CompositionInfo
    accessibility: AccessibilityInfo
    tags: List[str] = Field(default_factory=list)
    embeddings: EmbeddingsInfo
    dedup: DedupInfo