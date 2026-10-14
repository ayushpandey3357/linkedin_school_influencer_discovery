"""
Tests for SQLite Database persistence layer.
"""

from app.config import ScoringWeights
from app.database import Database
from app.models import (
    CandidateProfile,
    ComponentScores,
    RankedCandidate,
    ReferenceProfile,
)


def test_database_init_and_search_session_persistence(tmp_path):
    test_db_path = tmp_path / "test_search.db"
    db = Database(db_path=test_db_path)

    ref = ReferenceProfile(
        name="Test Ref Leader",
        current_role="Senior Education Advisor",
        organization="Apex School Society",
        professional_summary="Advisor to 10 schools.",
    )
    weights = ScoringWeights()

    candidate = RankedCandidate(
        rank=1,
        profile=CandidateProfile(
            profile_id="DB-TEST-001",
            name="Candidate One",
            linkedin_url="https://example.com/db-01",
            current_role="Former Principal",
            organization="Delhi Public School",
            location="Delhi",
            experience_years=20,
            school_ecosystem_indicators=["Former Principal"],
            source_type="SYNTHETIC TEST DATA",
        ),
        relevance_score=88.5,
        component_scores=ComponentScores(
            semantic_similarity=90.0,
            school_ecosystem=88.0,
            leadership=85.0,
            multi_school=86.0,
            skills_experience=80.0,
        ),
        explanation="Strong match based on school leadership.",
    )

    session_id = db.save_search_session(
        dataset_mode="TEST DATASET MODE",
        reference_profile=ref,
        weights=weights,
        ranking_engine="TF-IDF + Cosine",
        total_evaluated=30,
        ranked_candidates=[candidate],
    )
    assert session_id > 0

    sessions = db.list_search_sessions()
    assert len(sessions) == 1
    assert sessions[0]["id"] == session_id
    assert sessions[0]["reference_role"] == "Senior Education Advisor"

    results = db.get_session_results(session_id)
    assert len(results) == 1
    assert results[0]["profile_id"] == "DB-TEST-001"
    assert results[0]["relevance_score"] == 88.5


def test_database_imported_profiles_persistence(tmp_path):
    test_db_path = tmp_path / "test_imported.db"
    db = Database(db_path=test_db_path)

    profile = CandidateProfile(
        profile_id="IMP-SQL-1",
        name="Imported Leader",
        linkedin_url="https://example.com/imported/sql-1",
        current_role="Trustee & Director",
        organization="Vanguard Schools",
        location="Pune",
        experience_years=18,
        school_ecosystem_indicators=["Trustee of 4 schools"],
        skills=["Trustee Governance"],
        source_type="IMPORTED DATA",
    )

    inserted = db.save_imported_profiles([profile])
    assert inserted == 1

    loaded = db.get_imported_profiles()
    assert len(loaded) == 1
    assert loaded[0].profile_id == "IMP-SQL-1"
    assert loaded[0].source_type == "IMPORTED DATA"


def test_two_independent_search_sessions_persistence(tmp_path):
    test_db_path = tmp_path / "test_independent_sessions.db"
    db = Database(db_path=test_db_path)

    # Search 1: Dr. Sunita Sharma
    ref_1 = ReferenceProfile(
        profile_id="SYN-001",
        name="Dr. Sunita Sharma",
        current_role="Senior Education Consultant & K-12 School Advisor",
        organization="Independent Educational Advisory",
    )
    cand_1 = RankedCandidate(
        rank=1,
        profile=CandidateProfile(
            profile_id="CAND-01",
            name="Candidate A",
            linkedin_url="https://example.com/c1",
            current_role="Former Principal",
        ),
        relevance_score=85.0,
        component_scores=ComponentScores(
            semantic_similarity=80.0,
            school_ecosystem=85.0,
            leadership=90.0,
            multi_school=80.0,
            skills_experience=85.0,
        ),
        explanation="Exp 1",
    )
    s1_id = db.save_search_session(
        dataset_mode="TEST DATASET MODE",
        reference_profile=ref_1,
        weights=ScoringWeights(),
        ranking_engine="TF-IDF + Cosine",
        total_evaluated=30,
        ranked_candidates=[cand_1],
    )

    # Search 2: Rajesh V. Patel
    ref_2 = ReferenceProfile(
        profile_id="SYN-002",
        name="Rajesh V. Patel",
        current_role="Managing Director & Trustee",
        organization="Vanguard Group of Schools",
    )
    cand_2 = RankedCandidate(
        rank=1,
        profile=CandidateProfile(
            profile_id="CAND-02",
            name="Candidate B",
            linkedin_url="https://example.com/c2",
            current_role="Managing Trustee",
        ),
        relevance_score=78.0,
        component_scores=ComponentScores(
            semantic_similarity=75.0,
            school_ecosystem=80.0,
            leadership=85.0,
            multi_school=70.0,
            skills_experience=80.0,
        ),
        explanation="Exp 2",
    )
    s2_id = db.save_search_session(
        dataset_mode="TEST DATASET MODE",
        reference_profile=ref_2,
        weights=ScoringWeights(),
        ranking_engine="TF-IDF + Cosine",
        total_evaluated=30,
        ranked_candidates=[cand_2],
    )

    assert s1_id != s2_id
    sessions = db.list_search_sessions()
    assert len(sessions) == 2

    # Verify Session 1 is intact and distinct
    s1_results = db.get_session_results(s1_id)
    assert len(s1_results) == 1
    assert s1_results[0]["profile_id"] == "CAND-01"
    assert s1_results[0]["relevance_score"] == 85.0

    # Verify Session 2 is intact and distinct
    s2_results = db.get_session_results(s2_id)
    assert len(s2_results) == 1
    assert s2_results[0]["profile_id"] == "CAND-02"
    assert s2_results[0]["relevance_score"] == 78.0


def test_failed_search_does_not_corrupt_history(tmp_path):
    import pytest
    test_db_path = tmp_path / "test_transaction_rollback.db"
    db = Database(db_path=test_db_path)

    # 1. Establish valid history
    ref_initial = ReferenceProfile(
        name="Initial Leader",
        current_role="Director",
    )
    valid_cand = RankedCandidate(
        rank=1,
        profile=CandidateProfile(
            profile_id="CAND-INIT",
            name="Candidate Init",
            linkedin_url="https://example.com/c-init",
            current_role="Director",
        ),
        relevance_score=80.0,
        component_scores=ComponentScores(
            semantic_similarity=80.0,
            school_ecosystem=80.0,
            leadership=80.0,
            multi_school=80.0,
            skills_experience=80.0,
        ),
        explanation="Init",
    )
    init_session_id = db.save_search_session(
        dataset_mode="TEST DATASET MODE",
        reference_profile=ref_initial,
        weights=ScoringWeights(),
        ranking_engine="TF-IDF + Cosine",
        total_evaluated=10,
        ranked_candidates=[valid_cand],
    )
    assert len(db.list_search_sessions()) == 1

    # 2. Attempt a broken save (e.g. invalid candidate causing SQLite type / constraint failure)
    broken_ref = ReferenceProfile(name="Broken", current_role="Crash Test")
    
    # Class that causes an exception during iteration in save_search_session
    class CorruptList:
        def __iter__(self):
            yield valid_cand
            raise RuntimeError("Simulated mid-transaction failure")
        def __len__(self):
            return 2

    with pytest.raises(RuntimeError):
        db.save_search_session(
            dataset_mode="TEST DATASET MODE",
            reference_profile=broken_ref,
            weights=ScoringWeights(),
            ranking_engine="TF-IDF + Cosine",
            total_evaluated=10,
            ranked_candidates=CorruptList(),  # type: ignore
        )

    # 3. Confirm existing history remains 100% intact and uncorrupted
    sessions_after = db.list_search_sessions()
    assert len(sessions_after) == 1
    assert sessions_after[0]["id"] == init_session_id
    results_after = db.get_session_results(init_session_id)
    assert len(results_after) == 1
    assert results_after[0]["profile_id"] == "CAND-INIT"
