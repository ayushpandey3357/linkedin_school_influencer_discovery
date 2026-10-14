"""
Tests for CandidateProfile and ReferenceProfile schema validation.
"""

import pytest
from pydantic import ValidationError
from app.models import CandidateProfile, ReferenceProfile


def test_valid_candidate_profile():
    profile = CandidateProfile(
        profile_id="TEST-001",
        name="Dr. Anita Sharma",
        linkedin_url="https://example.com/profiles/test-001",
        current_role="Former Principal & Education Consultant",
        organization="Educational Excellence Trust",
        location="New Delhi, India",
        professional_summary="Over 20 years leading CBSE schools.",
        experience_years=22,
        school_ecosystem_indicators=["Former Principal", "Advises 10 schools"],
        skills=["CBSE Affiliation", "Teacher Training"],
        source_type="SYNTHETIC TEST DATA",
    )
    assert profile.profile_id == "TEST-001"
    assert profile.name == "Dr. Anita Sharma"
    assert profile.experience_years == 22.0
    assert len(profile.school_ecosystem_indicators) == 2
    assert profile.source_type == "SYNTHETIC TEST DATA"


def test_missing_required_fields_raises_validation_error():
    # Missing current_role
    with pytest.raises(ValidationError):
        CandidateProfile(
            profile_id="TEST-002",
            name="Test User",
            linkedin_url="https://example.com/profiles/test-002",
            current_role="",  # empty role should fail
        )

    # Missing name
    with pytest.raises(ValidationError):
        CandidateProfile(
            profile_id="TEST-003",
            name="",  # empty name should fail
            linkedin_url="https://example.com/profiles/test-003",
            current_role="Education Consultant",
        )


def test_missing_optional_fields_normalization():
    profile = CandidateProfile(
        profile_id="TEST-004",
        name="Test Professional",
        linkedin_url="https://example.com/profiles/test-004",
        current_role="School Advisor",
        organization=None,
        location="",
    )
    assert profile.organization == "Not specified"
    assert profile.location == "Not specified"
    assert profile.experience_years == 0.0
    assert profile.school_ecosystem_indicators == []
    assert profile.skills == []


def test_delimiter_parsing_for_indicators_and_skills():
    profile = CandidateProfile(
        profile_id="TEST-005",
        name="Multi Delimiter User",
        linkedin_url="https://example.com/profiles/test-005",
        current_role="School Director",
        school_ecosystem_indicators="Former Principal; Oversees 5 schools; NEP Consultant",
        skills="Governance, Budgeting, Leadership",
    )
    assert len(profile.school_ecosystem_indicators) == 3
    assert "Former Principal" in profile.school_ecosystem_indicators
    assert len(profile.skills) == 3
    assert "Governance" in profile.skills


def test_experience_years_parsing():
    p1 = CandidateProfile(
        profile_id="EXP-1",
        name="Exp 1",
        linkedin_url="https://example.com/exp-1",
        current_role="Teacher",
        experience_years="14.5",
    )
    assert p1.experience_years == 14.5

    p2 = CandidateProfile(
        profile_id="EXP-2",
        name="Exp 2",
        linkedin_url="https://example.com/exp-2",
        current_role="Teacher",
        experience_years="invalid_string",
    )
    assert p2.experience_years == 0.0
