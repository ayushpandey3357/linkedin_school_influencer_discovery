"""
CSV Export Module.
Exports ranked candidate profiles with complete score breakdowns, explanations,
and provenance metadata into structured CSV format.
"""

import io
from typing import List
import pandas as pd
from app.models import RankedCandidate


class CSVExporter:
    """Handles serialization and CSV export of discovery results."""

    @classmethod
    def to_dataframe(cls, candidates: List[RankedCandidate]) -> pd.DataFrame:
        """Converts a list of RankedCandidate objects into a structured pandas DataFrame."""
        rows = []
        for c in candidates:
            p = c.profile
            scores = c.component_scores
            rows.append(
                {
                    "profile_id": p.profile_id,
                    "rank": c.rank,
                    "relevance_score": c.relevance_score,
                    "name": p.name,
                    "current_role": p.current_role,
                    "organization": p.organization,
                    "location": p.location,
                    "experience_years": p.experience_years,
                    "semantic_score": scores.semantic_similarity,
                    "ecosystem_score": scores.school_ecosystem,
                    "leadership_score": scores.leadership,
                    "multi_school_score": scores.multi_school,
                    "skills_score": scores.skills_experience,
                    "explanation": c.explanation,
                    "school_ecosystem_indicators": "; ".join(p.school_ecosystem_indicators),
                    "skills": "; ".join(p.skills),
                    "linkedin_url": p.linkedin_url,
                    "source_type": p.source_type,
                    "source_description": p.source_description,
                    "source_url": p.source_url,
                    "ranking_method": c.ranking_method,
                    "last_verified_at": p.last_verified_at,
                }
            )
        return pd.DataFrame(rows)

    @classmethod
    def to_csv_string(cls, candidates: List[RankedCandidate]) -> str:
        """Generates a CSV string suitable for download streams."""
        df = cls.to_dataframe(candidates)
        output = io.StringIO()
        df.to_csv(output, index=False)
        return output.getvalue()
