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
