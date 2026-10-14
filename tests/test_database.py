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
