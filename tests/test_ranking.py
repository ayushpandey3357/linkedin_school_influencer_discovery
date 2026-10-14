"""
Tests for ML similarity, cosine similarity, TF-IDF fallback, and multi-component ranking.
"""

from app.config import ScoringWeights
from app.models import ReferenceProfile
from app.providers.synthetic_provider import SyntheticDatasetProvider
from app.ranking import RankingEngine, SimilarityEngine


def test_similarity_engine_tfidf_fallback():
    # Force TF-IDF engine
    engine = SimilarityEngine(engine_preference="tfidf")
    assert "TF-IDF" in engine.active_engine

    ref_text = "Senior K-12 education consultant, former CBSE principal advising school trusts."
    candidates = [
        "Former principal of CBSE school with 20 years in K-12 education advisory.",
        "Cloud backend software engineer with Kubernetes and AWS experience.",
        "School trustee and managing director of progressive school network.",
    ]

    scores = engine.compute_similarity(ref_text, candidates)
    assert len(scores) == 3
    # Check normalization to [0, 100]
    for s in scores:
        assert 0.0 <= s <= 100.0

    # The education consultant / principal must have higher similarity than the software engineer
    assert scores[0] > scores[1], f"Expected {scores[0]} > {scores[1]}"


def test_weights_normalization():
    # Non-normalized weights
    weights = ScoringWeights(
        semantic_similarity=0.8,
        school_ecosystem=0.4,
        leadership=0.2,
        multi_school=0.2,
        skills_experience=0.4,
    )
    assert not weights.validate_weights()

    norm_weights = weights.normalize()
    assert norm_weights.validate_weights()
    total = (
        norm_weights.semantic_similarity
        + norm_weights.school_ecosystem
        + norm_weights.leadership
        + norm_weights.multi_school
        + norm_weights.skills_experience
    )
    assert abs(total - 1.0) < 1e-3


def test_reference_profile_exclusion():
    provider = SyntheticDatasetProvider()
    candidates = provider.get_profiles()
    first_candidate = candidates[0]

    # Use first candidate as reference profile
    ref_profile = ReferenceProfile(
        name=first_candidate.name,
        linkedin_url=first_candidate.linkedin_url,
        current_role=first_candidate.current_role,
        organization=first_candidate.organization,
        professional_summary=first_candidate.professional_summary,
        experience_years=first_candidate.experience_years,
        school_ecosystem_indicators=first_candidate.school_ecosystem_indicators,
        skills=first_candidate.skills,
    )

    ranker = RankingEngine()
    ranked = ranker.rank_candidates(reference_profile=ref_profile, candidates=candidates)

    # Verify that the reference candidate was excluded
    ranked_urls = [r.profile.linkedin_url for r in ranked]
    assert first_candidate.linkedin_url not in ranked_urls
    assert all(r.profile.name != first_candidate.name for r in ranked)


def test_minimum_20_results_for_test_reference():
    provider = SyntheticDatasetProvider()
    candidates = provider.get_profiles()

    ref_profile = ReferenceProfile(
        name="Dr. Sunita Sharma (Reference)",
        linkedin_url="https://example.com/profiles/reference-eval",
        current_role="Senior Education Consultant & K-12 School Advisor",
        organization="Independent Educational Advisory",
        location="New Delhi, India",
        experience_years=24.0,
        professional_summary="Former CBSE Principal advising school trusts, directors, and boards on school turnaround.",
        school_ecosystem_indicators=["Former Principal", "Consultant to 15+ K-12 schools"],
        skills=["School Turnaround", "CBSE Affiliation", "Teacher Training"],
    )

    ranker = RankingEngine()
    ranked = ranker.rank_candidates(
        reference_profile=ref_profile,
        candidates=candidates,
        min_threshold=0.0,
        top_k=25,
    )

    # Minimum 20 results required by the assignment specification
    assert len(ranked) >= 20, f"Expected at least 20 ranked candidates, got {len(ranked)}"
    assert ranked[0].rank == 1
    assert ranked[19].rank == 20


def test_score_normalization_and_ranges():
    provider = SyntheticDatasetProvider()
    candidates = provider.get_profiles()

    ref_profile = ReferenceProfile(
        current_role="Managing Director & Trustee",
        organization="Vanguard Group of Schools",
        professional_summary="Trustee overseeing 8 premier K-12 campuses.",
    )

    ranker = RankingEngine()
    ranked = ranker.rank_candidates(reference_profile=ref_profile, candidates=candidates)

    for r in ranked:
        assert 0.0 <= r.relevance_score <= 100.0
        assert 0.0 <= r.component_scores.semantic_similarity <= 100.0
        assert 0.0 <= r.component_scores.school_ecosystem <= 100.0
        assert 0.0 <= r.component_scores.leadership <= 100.0
        assert 0.0 <= r.component_scores.multi_school <= 100.0
        assert 0.0 <= r.component_scores.skills_experience <= 100.0


def test_deterministic_ranking():
    provider = SyntheticDatasetProvider()
    candidates = provider.get_profiles()

    ref_profile = ReferenceProfile(
        current_role="Senior Education Consultant",
        professional_summary="Advisor on school turnaround and CBSE affiliation.",
    )

    ranker = RankingEngine()
    run_1 = ranker.rank_candidates(reference_profile=ref_profile, candidates=candidates)
    run_2 = ranker.rank_candidates(reference_profile=ref_profile, candidates=candidates)

    assert len(run_1) == len(run_2)
    for c1, c2 in zip(run_1, run_2):
        assert c1.profile.profile_id == c2.profile.profile_id
        assert c1.relevance_score == c2.relevance_score
        assert c1.rank == c2.rank


def test_ranking_discrimination_against_non_education_controls():
    provider = SyntheticDatasetProvider()
    candidates = provider.get_profiles()

    ref_profile = ReferenceProfile(
        current_role="Senior Education Consultant & K-12 School Advisor",
        professional_summary="Former Principal advising school trusts on CBSE accreditation and governance.",
        school_ecosystem_indicators=["Former Principal", "Advisor to 15 schools"],
    )

    ranker = RankingEngine()
    ranked = ranker.rank_candidates(reference_profile=ref_profile, candidates=candidates)

    # Control profiles in our dataset: SYN-027 (Cloud Software Architect), SYN-028 (Retail Store Manager)
    # Target education profiles: e.g. SYN-001, SYN-002, SYN-003, SYN-005 (Principals, Directors, Trustees)
    scores_by_id = {r.profile.profile_id: r.relevance_score for r in ranked}

    top_education_score = max(
        scores_by_id.get("SYN-002", 0),
        scores_by_id.get("SYN-003", 0),
        scores_by_id.get("SYN-005", 0),
    )
    control_software_score = scores_by_id.get("SYN-027", 0)
    control_retail_score = scores_by_id.get("SYN-028", 0)

    assert top_education_score > 55.0, f"Expected top education score > 55, got {top_education_score}"
    assert control_software_score < 40.0, f"Expected control score < 40, got {control_software_score}"
    assert control_retail_score < 40.0, f"Expected control score < 40, got {control_retail_score}"
    assert top_education_score > control_software_score + 25.0
