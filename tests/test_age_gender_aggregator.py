"""Tests for age/gender aggregation utility."""

import pytest
from src.photodb.utils.age_gender_aggregator import compute_person_age_gender


class TestComputePersonAgeGender:
    """Tests for compute_person_age_gender function."""

    def test_single_detection(self):
        """Test with a single detection."""
        detections = [{"age": 30, "photo_year": 2020, "gender": "male", "gender_confidence": 0.95}]

        result = compute_person_age_gender(detections)

        assert result["estimated_birth_year"] == 1990  # 2020 - 30
        assert result["birth_year_stddev"] is None  # No stddev with single sample
        assert result["gender"] == "male"
        # Confidence is proportion of total weight: 0.95/0.95 = 1.0
        assert result["gender_confidence"] == 1.0
        assert result["sample_count"] == 1

    def test_multiple_detections_consistent_birth_year(self):
        """Test with multiple detections that agree on birth year."""
        detections = [
            {"age": 30, "photo_year": 2020, "gender": "female", "gender_confidence": 0.90},
            {"age": 31, "photo_year": 2021, "gender": "female", "gender_confidence": 0.85},
            {"age": 32, "photo_year": 2022, "gender": "female", "gender_confidence": 0.92},
        ]

        result = compute_person_age_gender(detections)

        # All yield birth year 1990
        assert result["estimated_birth_year"] == 1990
        assert result["birth_year_stddev"] == 0.0  # All agree
        assert result["gender"] == "female"
        assert result["sample_count"] == 3

    def test_multiple_detections_varying_birth_year(self):
        """Test with detections that yield varying birth years."""
        detections = [
            {"age": 28, "photo_year": 2020, "gender": "male", "gender_confidence": 0.90},  # 1992
            {"age": 30, "photo_year": 2020, "gender": "male", "gender_confidence": 0.85},  # 1990
            {"age": 32, "photo_year": 2020, "gender": "male", "gender_confidence": 0.88},  # 1988
        ]

        result = compute_person_age_gender(detections)

        # Median of [1988, 1990, 1992] = 1990
        assert result["estimated_birth_year"] == 1990
        assert result["birth_year_stddev"] is not None
        assert result["birth_year_stddev"] > 0  # Should have some variation
        assert result["sample_count"] == 3

    def test_gender_weighted_majority_clear_winner(self):
        """Test gender determined by weighted majority with clear winner."""
        detections = [
            {"age": 30, "photo_year": 2020, "gender": "male", "gender_confidence": 0.95},
            {"age": 31, "photo_year": 2021, "gender": "male", "gender_confidence": 0.90},
            {"age": 32, "photo_year": 2022, "gender": "female", "gender_confidence": 0.60},
        ]

        result = compute_person_age_gender(detections)

        # male: 0.95 + 0.90 = 1.85
        # female: 0.60
        assert result["gender"] == "male"
        # Confidence should reflect the total weight for winning gender
        assert result["gender_confidence"] == pytest.approx(1.85 / (1.85 + 0.60), rel=0.01)

    def test_gender_weighted_majority_close_race(self):
        """Test gender when confidence totals are close."""
        detections = [
            {"age": 30, "photo_year": 2020, "gender": "male", "gender_confidence": 0.51},
            {"age": 31, "photo_year": 2021, "gender": "female", "gender_confidence": 0.49},
        ]

        result = compute_person_age_gender(detections)

        # male: 0.51, female: 0.49
        assert result["gender"] == "male"
        assert result["gender_confidence"] == pytest.approx(0.51, rel=0.01)

    def test_empty_detections_list(self):
        """Test with empty detections list."""
        result = compute_person_age_gender([])

        assert result["estimated_birth_year"] is None
        assert result["birth_year_stddev"] is None
        assert result["gender"] is None
        assert result["gender_confidence"] is None
        assert result["sample_count"] == 0

    def test_all_unknown_gender(self):
        """Test when all detections have unknown gender."""
        detections = [
            {"age": 30, "photo_year": 2020, "gender": "unknown", "gender_confidence": 0.0},
            {"age": 31, "photo_year": 2021, "gender": "unknown", "gender_confidence": 0.0},
        ]

        result = compute_person_age_gender(detections)

        assert result["estimated_birth_year"] == 1990
        assert result["gender"] == "unknown"
        assert result["gender_confidence"] == 0.0
        assert result["sample_count"] == 2

    def test_mixed_genders_with_unknown(self):
        """Test with mix of known and unknown genders."""
        detections = [
            {"age": 30, "photo_year": 2020, "gender": "male", "gender_confidence": 0.80},
            {"age": 31, "photo_year": 2021, "gender": "unknown", "gender_confidence": 0.0},
            {"age": 32, "photo_year": 2022, "gender": "female", "gender_confidence": 0.70},
        ]

        result = compute_person_age_gender(detections)

        # male: 0.80, female: 0.70, unknown: 0.0
        assert result["gender"] == "male"
        assert result["sample_count"] == 3

    def test_none_gender_treated_as_unknown(self):
        """Test that None gender is treated as unknown."""
        detections = [
            {"age": 30, "photo_year": 2020, "gender": None, "gender_confidence": None},
            {"age": 31, "photo_year": 2021, "gender": "male", "gender_confidence": 0.85},
        ]

        result = compute_person_age_gender(detections)

        assert result["gender"] == "male"
        assert result["sample_count"] == 2

    def test_median_birth_year_even_count(self):
        """Test median calculation with even number of samples."""
        detections = [
            {"age": 28, "photo_year": 2020, "gender": "male", "gender_confidence": 0.90},  # 1992
            {"age": 30, "photo_year": 2020, "gender": "male", "gender_confidence": 0.85},  # 1990
            {"age": 32, "photo_year": 2020, "gender": "male", "gender_confidence": 0.88},  # 1988
            {"age": 34, "photo_year": 2020, "gender": "male", "gender_confidence": 0.82},  # 1986
        ]

        result = compute_person_age_gender(detections)

        # Median of [1986, 1988, 1990, 1992] = (1988 + 1990) / 2 = 1989
        assert result["estimated_birth_year"] == 1989
        assert result["sample_count"] == 4

    def test_stddev_with_two_samples(self):
        """Test stddev calculation with exactly two samples."""
        detections = [
            {"age": 30, "photo_year": 2020, "gender": "male", "gender_confidence": 0.90},  # 1990
            {"age": 32, "photo_year": 2020, "gender": "male", "gender_confidence": 0.85},  # 1988
        ]

        result = compute_person_age_gender(detections)

        # Stddev of [1988, 1990] should be 1.0 (population stddev)
        assert result["birth_year_stddev"] is not None
        assert result["birth_year_stddev"] > 0
        assert result["sample_count"] == 2

    def test_missing_age_field(self):
        """Test handling of detection missing age field."""
        detections = [
            {"photo_year": 2020, "gender": "male", "gender_confidence": 0.90},  # Missing age
            {"age": 30, "photo_year": 2020, "gender": "male", "gender_confidence": 0.85},
        ]

        result = compute_person_age_gender(detections)

        # Should only use the valid detection
        assert result["estimated_birth_year"] == 1990
        assert result["sample_count"] == 2  # Counts all detections
        assert result["birth_year_stddev"] is None  # Only one valid birth year sample

    def test_missing_photo_year_field(self):
        """Test handling of detection missing photo_year field."""
        detections = [
            {"age": 30, "gender": "male", "gender_confidence": 0.90},  # Missing photo_year
            {"age": 30, "photo_year": 2020, "gender": "male", "gender_confidence": 0.85},
        ]

        result = compute_person_age_gender(detections)

        # Should only use the valid detection for birth year
        assert result["estimated_birth_year"] == 1990
        assert result["sample_count"] == 2

    def test_zero_age(self):
        """Test with age of 0 (newborn)."""
        detections = [
            {"age": 0, "photo_year": 2020, "gender": "female", "gender_confidence": 0.70},
        ]

        result = compute_person_age_gender(detections)

        assert result["estimated_birth_year"] == 2020
        assert result["sample_count"] == 1

    def test_high_age(self):
        """Test with high age value."""
        detections = [
            {"age": 90, "photo_year": 2020, "gender": "male", "gender_confidence": 0.80},
        ]

        result = compute_person_age_gender(detections)

        assert result["estimated_birth_year"] == 1930
        assert result["sample_count"] == 1

    def test_gender_confidence_normalized(self):
        """Test that gender confidence reflects proportion of total weight."""
        detections = [
            {"age": 30, "photo_year": 2020, "gender": "male", "gender_confidence": 0.90},
            {"age": 31, "photo_year": 2021, "gender": "female", "gender_confidence": 0.10},
        ]

        result = compute_person_age_gender(detections)

        # male: 0.90, female: 0.10, total: 1.0
        # male wins with confidence = 0.90 / 1.0 = 0.90
        assert result["gender"] == "male"
        assert result["gender_confidence"] == pytest.approx(0.90, rel=0.01)

    def test_large_sample_count(self):
        """Test with many detections."""
        detections = [
            {"age": 30 + i % 5, "photo_year": 2020, "gender": "female", "gender_confidence": 0.85}
            for i in range(100)
        ]

        result = compute_person_age_gender(detections)

        assert result["sample_count"] == 100
        assert result["gender"] == "female"
        assert result["estimated_birth_year"] is not None

    def test_negative_age_ignored(self):
        """Test that negative ages are ignored for birth year calculation."""
        detections = [
            {"age": -5, "photo_year": 2020, "gender": "male", "gender_confidence": 0.90},  # Invalid
            {"age": 30, "photo_year": 2020, "gender": "male", "gender_confidence": 0.85},
        ]

        result = compute_person_age_gender(detections)

        # Should only use the valid detection
        assert result["estimated_birth_year"] == 1990
        assert result["sample_count"] == 2

    def test_all_invalid_ages(self):
        """Test when all detections have invalid ages."""
        detections = [
            {"age": -5, "photo_year": 2020, "gender": "male", "gender_confidence": 0.90},
            {"age": None, "photo_year": 2020, "gender": "female", "gender_confidence": 0.85},
        ]

        result = compute_person_age_gender(detections)

        # No valid birth year data
        assert result["estimated_birth_year"] is None
        assert result["birth_year_stddev"] is None
        # But gender can still be computed
        assert result["gender"] == "male"  # Higher confidence
        assert result["sample_count"] == 2
