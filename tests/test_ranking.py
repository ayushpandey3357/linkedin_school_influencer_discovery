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


def test_reference_profile_exclusion_by_stable_id():
    provider = SyntheticDatasetProvider()
    candidates = provider.get_profiles()

    # Reference profile with explicit stable ID matching SYN-001
    ref_profile = ReferenceProfile(
        profile_id="SYN-001",
        name="Any Name Variant",
        linkedin_url="https://example.com/custom-url",
        current_role="Senior Education Consultant",
    )

    ranker = RankingEngine()
    ranked = ranker.rank_candidates(reference_profile=ref_profile, candidates=candidates)

    ranked_ids = [r.profile.profile_id for r in ranked]
    assert "SYN-001" not in ranked_ids, "SYN-001 must be excluded by stable profile ID"
    assert len(ranked) == len(candidates) - 1


def test_reference_profile_exclusion_by_normalized_url():
    provider = SyntheticDatasetProvider()
    candidates = provider.get_profiles()

    # Reference with differing protocol, www prefix, trailing slash, query parameter
    ref_profile = ReferenceProfile(
        name="Different Name",
        linkedin_url="http://www.example.com/profiles/syn-002/?trk=public_profile#details",
        current_role="Education Leader",
    )

    ranker = RankingEngine()
    ranked = ranker.rank_candidates(reference_profile=ref_profile, candidates=candidates)

    ranked_ids = [r.profile.profile_id for r in ranked]
    assert "SYN-002" not in ranked_ids, "SYN-002 must be excluded by normalized URL"


def test_reference_profile_exclusion_by_conservative_multi_field():
    provider = SyntheticDatasetProvider()
    candidates = provider.get_profiles()

    # Reference profile with generic placeholder URL and no profile_id,
    # but base name, role, organization, and experience match Dr. Sunita Sharma
    ref_profile = ReferenceProfile(
        name="Dr. Sunita Sharma (Reference Profile)",
        linkedin_url="https://example.com/profiles/reference-profile",
        current_role="Senior Education Consultant & K-12 School Advisor",
        organization="Independent Educational Advisory",
        location="New Delhi, India",
        experience_years=24.0,
        professional_summary="Former CBSE Senior Principal advising school trusts on school turnaround.",
        school_ecosystem_indicators=["Former Principal of 2 flagship CBSE schools"],
    )

    ranker = RankingEngine()
    ranked = ranker.rank_candidates(reference_profile=ref_profile, candidates=candidates)

    ranked_ids = [r.profile.profile_id for r in ranked]
    assert "SYN-001" not in ranked_ids, "SYN-001 must be excluded by conservative multi-field check"


def test_unrelated_person_with_same_name_is_not_excluded():
    provider = SyntheticDatasetProvider()
    candidates = provider.get_profiles()

    # Unrelated person who happens to share the name 'Dr. Sunita Sharma',
    # but holds a completely different role and organization
    ref_profile = ReferenceProfile(
        name="Dr. Sunita Sharma",
        linkedin_url="https://example.com/profiles/unrelated-sunita",
        current_role="Senior Secondary Physics Teacher",
        organization="St. Xavier's High School",
        location="Kolkata, India",
        experience_years=8.0,
        professional_summary="Teaching senior secondary physics and lab coordinator.",
    )

    ranker = RankingEngine()
    ranked = ranker.rank_candidates(reference_profile=ref_profile, candidates=candidates)

    ranked_ids = [r.profile.profile_id for r in ranked]
    assert "SYN-001" in ranked_ids, "Unrelated person with same name must NOT be excluded"


def test_dr_sunita_sharma_reference_search_returns_at_least_20_results_excluding_reference():
    provider = SyntheticDatasetProvider()
    candidates = provider.get_profiles()

    # The exact test search from the assignment
    ref_profile = ReferenceProfile(
        profile_id="SYN-001",
        name="Dr. Sunita Sharma (Reference Profile)",
        linkedin_url="https://example.com/profiles/syn-001",
        current_role="Senior Education Consultant & K-12 School Advisor",
        organization="Independent Educational Advisory",
        location="New Delhi, India",
        experience_years=24.0,
        professional_summary="Former CBSE Senior Principal with 24 years in K-12 school education. Now advising school trusts, founding boards, and directors on school turnaround, teacher development, CBSE/NEP curriculum alignment, and institutional governance across 15+ private schools.",
        school_ecosystem_indicators=[
            "Former Principal of 2 flagship CBSE schools (1,800+ students)",
            "Advisory board member for 4 school trusts",
            "Consultant to 15+ K-12 schools across Northern India",
        ],
        skills=["School Turnaround", "Curriculum Strategy", "CBSE Affiliation"],
    )

    ranker = RankingEngine()
    ranked = ranker.rank_candidates(
        reference_profile=ref_profile,
        candidates=candidates,
        min_threshold=0.0,
        top_k=25,
    )

    # 1. At least 20 results returned
    assert len(ranked) >= 20, f"Expected at least 20 ranked candidates, got {len(ranked)}"

    # 2. Reference profile itself is strictly NOT returned
    returned_ids = [r.profile.profile_id for r in ranked]
    assert "SYN-001" not in returned_ids, "Dr. Sunita Sharma (SYN-001) must NOT be returned in results"
    for r in ranked:
        assert r.profile.profile_id != "SYN-001"
        assert r.profile.linkedin_url != "https://example.com/profiles/syn-001"

    # 3. Ranks are contiguous from 1
    assert ranked[0].rank == 1
    assert ranked[19].rank == 20


def test_total_relevance_score_formula_calculation():
    provider = SyntheticDatasetProvider()
    candidates = provider.get_profiles()

    custom_weights = ScoringWeights(
        semantic_similarity=0.35,
        school_ecosystem=0.30,
        leadership=0.15,
        multi_school=0.15,
        skills_experience=0.05,
    ).normalize()

    ref_profile = ReferenceProfile(
        current_role="Education Director & Trustee",
        organization="Global School Network",
        professional_summary="Director of multi-campus education network.",
    )

    ranker = RankingEngine(weights=custom_weights)
    ranked = ranker.rank_candidates(reference_profile=ref_profile, candidates=candidates)

    for r in ranked:
        cs = r.component_scores
        expected_score = (
            (custom_weights.semantic_similarity * cs.semantic_similarity)
            + (custom_weights.school_ecosystem * cs.school_ecosystem)
            + (custom_weights.leadership * cs.leadership)
            + (custom_weights.multi_school * cs.multi_school)
            + (custom_weights.skills_experience * cs.skills_experience)
        )
        expected_score = round(max(0.0, min(100.0, expected_score)), 2)
        assert abs(r.relevance_score - expected_score) <= 0.05, (
            f"Score mismatch for {r.profile.profile_id}: {r.relevance_score} != {expected_score}"
        )


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
        profile_id="SYN-001",
        current_role="Senior Education Consultant & K-12 School Advisor",
        professional_summary="Former Principal advising school trusts on CBSE accreditation and governance.",
        school_ecosystem_indicators=["Former Principal", "Advisor to 15 schools"],
    )

    ranker = RankingEngine()
    ranked = ranker.rank_candidates(reference_profile=ref_profile, candidates=candidates)

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
    assert top_education_score > control_software_score + 15.0
