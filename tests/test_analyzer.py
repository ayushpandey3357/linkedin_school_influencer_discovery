"""
Tests for ProfileAnalyzer module.
"""

from app.models import CandidateProfile
from app.profile_analyzer import ProfileAnalyzer


def test_analyzer_derives_leadership_and_multi_school():
    profile = CandidateProfile(
        profile_id="TEST-ANA-1",
        name="Sunita Sharma",
        linkedin_url="https://example.com/profiles/ana-1",
        current_role="Former Principal & Education Advisor",
        organization="National School Trust",
        location="Delhi",
        experience_years=24,
        professional_summary="Overseeing 15+ campuses and advising school promoters on CBSE affiliation.",
        school_ecosystem_indicators=["Former Principal", "Oversees 15 campuses"],
        skills=["CBSE Affiliation", "Teacher Training"],
    )

    report = ProfileAnalyzer.analyze(profile)
    assert report.is_school_leader is True
    assert report.is_consultant_or_advisor is True
    assert report.has_multi_school_experience is True
    assert report.education_sector == "K-12 School Education"
    assert "Executive" in report.seniority or "Principal" in report.seniority
    assert len(report.explicit_evidence) > 0


def test_analyzer_flags_missing_fields():
    profile = CandidateProfile(
        profile_id="TEST-ANA-2",
        name="Sparse Profile",
        linkedin_url="https://example.com/profiles/ana-2",
        current_role="Advisor",
        organization="",  # Missing
        location="Not specified",  # Missing
        experience_years=0,
    )

    report = ProfileAnalyzer.analyze(profile)
    assert any("Organization not specified" in u for u in report.missing_fields_or_uncertainties)
    assert any("Geographic location not specified" in u for u in report.missing_fields_or_uncertainties)
    assert any("Experience years" in u for u in report.missing_fields_or_uncertainties)


def test_analyzer_control_profile_non_education():
    profile = CandidateProfile(
        profile_id="TEST-ANA-3",
        name="Devika Pillai",
        linkedin_url="https://example.com/profiles/ana-3",
        current_role="Senior Cloud Architect",
        organization="Fintech Global",
        location="Bengaluru",
        experience_years=11,
        professional_summary="Backend Kubernetes microservices and distributed databases.",
        skills=["Kubernetes", "Java", "AWS"],
    )

    report = ProfileAnalyzer.analyze(profile)
    assert report.education_sector == "General / Non-School Domain"
    assert report.is_school_leader is False
    assert report.is_consultant_or_advisor is False
    assert report.has_multi_school_experience is False
