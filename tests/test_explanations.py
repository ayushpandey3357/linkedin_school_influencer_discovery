"""
Tests for ExplanationEngine module.
"""

from app.explanations import ExplanationEngine
from app.models import CandidateProfile, ComponentScores, RankedCandidate


def test_explanation_generation_high_match():
    candidate = RankedCandidate(
        rank=1,
        profile=CandidateProfile(
            profile_id="EXP-TEST-1",
            name="Dr. Sunita Sharma",
            linkedin_url="https://example.com/profiles/exp-1",
            current_role="Former Principal & Senior Education Consultant",
            organization="Independent Educational Advisory",
            location="New Delhi",
            experience_years=24,
            school_ecosystem_indicators=["Former Principal of 2 CBSE schools"],
            source_type="SYNTHETIC TEST DATA",
        ),
        relevance_score=84.5,
        component_scores=ComponentScores(
            semantic_similarity=85.0,
            school_ecosystem=88.0,
            leadership=85.0,
            multi_school=80.0,
            skills_experience=82.0,
        ),
        explanation="",
        key_matching_factors=["School leadership track record", "Strong semantic profile alignment"],
    )

    explanation = ExplanationEngine.generate_explanation(candidate)
    assert "[SYNTHETIC TEST DATA]" in explanation
    assert "84.5/100" in explanation
    assert "Former Principal" in explanation
    assert "Independent Educational Advisory" in explanation
    assert "Primary algorithmic scoring drivers" in explanation


def test_explanation_generation_low_match():
    candidate = RankedCandidate(
        rank=25,
        profile=CandidateProfile(
            profile_id="EXP-TEST-2",
            name="Devika Pillai",
            linkedin_url="https://example.com/profiles/exp-2",
            current_role="Senior Cloud Architect",
            organization="Fintech Global",
            location="Bengaluru",
            experience_years=11,
            source_type="SYNTHETIC TEST DATA",
        ),
        relevance_score=18.2,
        component_scores=ComponentScores(
            semantic_similarity=15.0,
            school_ecosystem=10.0,
            leadership=20.0,
            multi_school=10.0,
            skills_experience=12.0,
        ),
        explanation="",
        key_matching_factors=["Baseline professional relevance in education domain"],
    )

    explanation = ExplanationEngine.generate_explanation(candidate)
    assert "Low relevance match" in explanation
    assert "minimal K-12 school ecosystem experience" in explanation


def test_explanation_cites_factual_indicators_for_ecosystem_and_consulting():
    candidate = RankedCandidate(
        rank=2,
        profile=CandidateProfile(
            profile_id="EXP-TEST-3",
            name="Rajesh Patel",
            linkedin_url="https://example.com/profiles/exp-3",
            current_role="Managing Director & Trustee",
            organization="Vanguard Schools",
            location="Ahmedabad",
            experience_years=20,
            school_ecosystem_indicators=[
                "Managing Trustee overseeing 8 K-12 schools",
                "Advisory board member for 4 school trusts",
            ],
            skills=["School Network Management", "Trustee Governance"],
            source_type="SYNTHETIC TEST DATA",
        ),
        relevance_score=79.2,
        component_scores=ComponentScores(
            semantic_similarity=72.0,
            school_ecosystem=85.0,
            leadership=85.0,
            multi_school=80.0,
            skills_experience=70.0,
        ),
        explanation="",
    )

    explanation = ExplanationEngine.generate_explanation(candidate)
    # Must cite actual dataset evidence without fabricating external connections
    assert "[SYNTHETIC TEST DATA]" in explanation
    assert "Managing Director & Trustee" in explanation
    assert "Vanguard Schools" in explanation
    assert "8 K-12 schools" in explanation
    # Ensure no fabricated claims like mutual connections
    assert "mutual connection" not in explanation.lower()
    assert "network friend" not in explanation.lower()
