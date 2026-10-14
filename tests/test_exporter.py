"""
Tests for CSVExporter module.
"""

import io
import pandas as pd
from app.exporter import CSVExporter
from app.models import CandidateProfile, ComponentScores, RankedCandidate


def test_exporter_to_dataframe_and_csv_string():
    cand = RankedCandidate(
        rank=1,
        profile=CandidateProfile(
            profile_id="EXP-1",
            name="Pooja Chowdhury",
            linkedin_url="https://example.com/pooja",
            current_role="Founding Principal",
            organization="Apex Academy",
            location="Kolkata",
            experience_years=19,
            school_ecosystem_indicators=["Founding Principal of CBSE school"],
            skills=["Academic Audit", "CBSE Affiliation"],
            source_type="SYNTHETIC TEST DATA",
        ),
        relevance_score=82.4,
        component_scores=ComponentScores(
            semantic_similarity=85.0,
            school_ecosystem=84.0,
            leadership=80.0,
            multi_school=78.0,
            skills_experience=75.0,
        ),
        explanation="Strong match based on school founding and leadership.",
        ranking_method="TF-IDF + Cosine Similarity",
    )

    df = CSVExporter.to_dataframe([cand])
    assert len(df) == 1
    assert "rank" in df.columns
    assert "relevance_score" in df.columns
    assert "semantic_score" in df.columns
    assert "source_type" in df.columns
    assert df.iloc[0]["name"] == "Pooja Chowdhury"
    assert df.iloc[0]["relevance_score"] == 82.4

    csv_text = CSVExporter.to_csv_string([cand])
    assert isinstance(csv_text, str)
    assert "Pooja Chowdhury" in csv_text

    # Verify re-parsing with pandas
    re_parsed = pd.read_csv(io.StringIO(csv_text))
    assert len(re_parsed) == 1
    assert re_parsed.iloc[0]["profile_id"] == "EXP-1"


def test_full_search_export_consistency():
    from app.explanations import ExplanationEngine
    from app.models import ReferenceProfile
    from app.providers.synthetic_provider import SyntheticDatasetProvider
    from app.ranking import RankingEngine

    provider = SyntheticDatasetProvider()
    candidates = provider.get_profiles()

    ref = ReferenceProfile(
        profile_id="SYN-001",
        name="Dr. Sunita Sharma (Reference Profile)",
        linkedin_url="https://example.com/profiles/syn-001",
        current_role="Senior Education Consultant & K-12 School Advisor",
        organization="Independent Educational Advisory",
        location="New Delhi, India",
        experience_years=24.0,
        professional_summary="Advising school trusts and boards.",
    )

    ranker = RankingEngine()
    ranked = ranker.rank_candidates(ref, candidates, top_k=25)
    ranked = ExplanationEngine.attach_explanations(ranked)

    assert len(ranked) >= 20

    csv_text = CSVExporter.to_csv_string(ranked)
    df = pd.read_csv(io.StringIO(csv_text))

    assert len(df) == len(ranked)
    for idx, cand in enumerate(ranked):
        row = df.iloc[idx]
        assert row["rank"] == cand.rank
        assert row["profile_id"] == cand.profile.profile_id
        assert row["name"] == cand.profile.name
        assert round(float(row["relevance_score"]), 2) == round(cand.relevance_score, 2)
        assert round(float(row["semantic_score"]), 2) == round(cand.component_scores.semantic_similarity, 2)
        assert round(float(row["ecosystem_score"]), 2) == round(cand.component_scores.school_ecosystem, 2)
        assert round(float(row["leadership_score"]), 2) == round(cand.component_scores.leadership, 2)
        assert round(float(row["multi_school_score"]), 2) == round(cand.component_scores.multi_school, 2)
        assert round(float(row["skills_score"]), 2) == round(cand.component_scores.skills_experience, 2)
        assert row["explanation"] == cand.explanation
        assert row["source_type"] == cand.profile.source_type
        assert row["linkedin_url"] == cand.profile.linkedin_url
