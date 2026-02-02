"""Age/gender aggregation utility for person-level statistics.

This module provides functions to aggregate age and gender data from
individual detections into person-level statistics.
"""

import statistics
from typing import Any, Dict, List, Optional


def compute_person_age_gender(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregated age/gender statistics from detections.

    Args:
        detections: List of dicts with 'age', 'photo_year', 'gender', 'gender_confidence'

    Returns:
        Dict with:
            - estimated_birth_year: Median of (photo_year - age) values, or None if no valid data
            - birth_year_stddev: Standard deviation of birth year estimates if >1 valid sample
            - gender: Gender determined by weighted majority of confidence scores
            - gender_confidence: Proportion of total confidence for winning gender
            - sample_count: Total number of detections processed
    """
    if not detections:
        return {
            "estimated_birth_year": None,
            "birth_year_stddev": None,
            "gender": None,
            "gender_confidence": None,
            "sample_count": 0,
        }

    # Collect valid birth year estimates
    birth_years: List[int] = []
    for detection in detections:
        age = detection.get("age")
        photo_year = detection.get("photo_year")

        # Skip if age or photo_year is missing or invalid
        if age is None or photo_year is None:
            continue
        if not isinstance(age, (int, float)) or age < 0:
            continue

        birth_year = int(photo_year - age)
        birth_years.append(birth_year)

    # Compute birth year statistics
    estimated_birth_year: Optional[int] = None
    birth_year_stddev: Optional[float] = None

    if birth_years:
        estimated_birth_year = int(statistics.median(birth_years))
        if len(birth_years) > 1:
            birth_year_stddev = statistics.stdev(birth_years)

    # Compute gender by weighted majority
    gender_weights: Dict[str, float] = {}
    for detection in detections:
        gender = detection.get("gender")
        confidence = detection.get("gender_confidence")

        # Treat None/missing as unknown
        if gender is None:
            gender = "unknown"
        if confidence is None:
            confidence = 0.0

        if gender not in gender_weights:
            gender_weights[gender] = 0.0
        gender_weights[gender] += confidence

    # Determine winning gender
    gender: Optional[str] = None
    gender_confidence: Optional[float] = None

    if gender_weights:
        total_weight = sum(gender_weights.values())

        # Find gender with highest weight
        winning_gender = max(gender_weights.keys(), key=lambda g: gender_weights[g])
        gender = winning_gender

        if total_weight > 0:
            gender_confidence = gender_weights[winning_gender] / total_weight
        else:
            gender_confidence = 0.0

    return {
        "estimated_birth_year": estimated_birth_year,
        "birth_year_stddev": birth_year_stddev,
        "gender": gender,
        "gender_confidence": gender_confidence,
        "sample_count": len(detections),
    }
