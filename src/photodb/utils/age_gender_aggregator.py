"""Age/gender aggregation utility for person-level and cluster-level statistics.

This module provides functions to aggregate age and gender data from
individual detections into person-level and cluster-level statistics.
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


def compute_cluster_age_gender(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate age/gender from detections within a single cluster.

    Since all detections in a cluster are the same person at roughly
    the same age, we take the median age_estimate directly (no birth
    year conversion).

    Args:
        detections: List of dicts with 'age_estimate', 'gender', 'gender_confidence'

    Returns:
        Dict with:
            - age_estimate: Median of age_estimate values, or None if no valid data
            - age_estimate_stddev: Standard deviation if >1 valid sample, else None
            - gender: Weighted majority gender ('M', 'F', or 'U')
            - gender_confidence: Proportion of total confidence for winning gender
            - sample_count: Number of detections with valid age_estimate
    """
    if not detections:
        return {
            "age_estimate": None,
            "age_estimate_stddev": None,
            "gender": None,
            "gender_confidence": None,
            "sample_count": 0,
        }

    # Collect valid age estimates
    ages: List[float] = []
    for detection in detections:
        age = detection.get("age_estimate")
        if age is not None and isinstance(age, (int, float)) and age >= 0:
            ages.append(float(age))

    # Compute age statistics
    age_estimate: Optional[float] = None
    age_estimate_stddev: Optional[float] = None

    if ages:
        age_estimate = statistics.median(ages)
        if len(ages) > 1:
            age_estimate_stddev = statistics.stdev(ages)

    # Compute gender by weighted majority (same logic as person-level)
    gender_weights: Dict[str, float] = {}
    for detection in detections:
        g = detection.get("gender")
        confidence = detection.get("gender_confidence")

        if g is None:
            g = "U"
        if confidence is None:
            confidence = 0.0

        if g not in gender_weights:
            gender_weights[g] = 0.0
        gender_weights[g] += confidence

    # Determine winning gender
    gender: Optional[str] = None
    gender_confidence: Optional[float] = None

    if gender_weights:
        total_weight = sum(gender_weights.values())
        winning_gender = max(gender_weights.keys(), key=lambda k: gender_weights[k])
        gender = winning_gender

        if total_weight > 0:
            gender_confidence = gender_weights[winning_gender] / total_weight
        else:
            gender_confidence = 0.0

    return {
        "age_estimate": age_estimate,
        "age_estimate_stddev": age_estimate_stddev,
        "gender": gender,
        "gender_confidence": gender_confidence,
        "sample_count": len(ages),
    }
