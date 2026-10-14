"""
Pydantic models for profile schema, reference profile, scoring, and search parameters.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class CandidateProfile(BaseModel):
    """Normalized candidate profile schema adhering to compliance and data provenance rules."""

    profile_id: str = Field(..., description="Unique immutable profile identifier")
    name: str = Field(..., description="Professional's full name")
    linkedin_url: str = Field(
        ...,
        description="Profile URL or synthetic placeholder (e.g. https://example.com/profiles/syn-001)",
    )
    current_role: str = Field(..., description="Current primary professional title")
    organization: str = Field(
        default="Not specified",
        description="Current employing organization or school network",
    )
    location: str = Field(
        default="Not specified", description="City, state, or region"
    )
    professional_summary: str = Field(
        default="", description="Summary of career and education focus"
    )
    experience_years: float = Field(
        default=0.0, ge=0.0, description="Total years of professional experience"
    )
    school_ecosystem_indicators: List[str] = Field(
        default_factory=list,
        description="List of factual achievements, school leadership, advisory, or trustee roles",
    )
    skills: List[str] = Field(
        default_factory=list,
        description="List of relevant skills and domain areas",
    )
    source_type: str = Field(
        default="SYNTHETIC TEST DATA",
        description="Mode / provenance: SYNTHETIC TEST DATA, IMPORTED DATA, or AUTHORIZED CONNECTOR",
    )
    source_description: str = Field(
        default="", description="Origin or audit details of the data"
    )
    source_url: str = Field(
        default="", description="Reference or origin URL for data provenance"
    )
    last_verified_at: str = Field(
        default="", description="ISO timestamp or date of verification"
    )

    @field_validator("name", "current_role")
    @classmethod
    def validate_non_empty(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(f"{info.field_name} cannot be empty")
        return v.strip()

    @field_validator("school_ecosystem_indicators", "skills", mode="before")
    @classmethod
    def parse_string_to_list(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        if isinstance(v, str):
            if not v.strip() or v.strip().lower() == "not specified":
                return []
            # Support semicolon, pipe, or comma separated formats
            delimiters = [";", "|", ","]
            chosen_delimiter = ";"
            for d in delimiters:
                if d in v:
                    chosen_delimiter = d
                    break
            return [item.strip() for item in v.split(chosen_delimiter) if item.strip()]
        return []

    @field_validator("experience_years", mode="before")
    @classmethod
    def parse_experience(cls, v: Union[int, float, str]) -> float:
        if v is None or v == "":
            return 0.0
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0

    @field_validator("organization", "location", mode="before")
    @classmethod
    def clean_text(cls, v: Optional[str]) -> str:
        if v is None or not str(v).strip():
            return "Not specified"
        return str(v).strip()

    @field_validator("professional_summary", mode="before")
    @classmethod
    def clean_summary(cls, v: Optional[str]) -> str:
        if v is None:
            return ""
        return str(v).strip()

    def get_full_text_representation(self) -> str:
        """Constructs an enriched text representation for ML vectorization."""
        indicators_str = " ".join(self.school_ecosystem_indicators)
        skills_str = " ".join(self.skills)
        parts = [
            f"Role: {self.current_role}.",
            f"Organization: {self.organization}.",
            f"Location: {self.location}.",
            f"Summary: {self.professional_summary}.",
            f"School Ecosystem Experience: {indicators_str}.",
            f"Skills: {skills_str}.",
            f"Experience: {self.experience_years} years.",
        ]
        return " ".join([p for p in parts if p.strip()])


class ReferenceProfile(BaseModel):
    """Reference profile provided by the user to discover similar school ecosystem influencers."""

    profile_id: Optional[str] = Field(
        default=None, description="Optional stable profile ID (e.g. SYN-001 or IMP-001)"
    )
    name: str = Field(default="Reference Professional")
    linkedin_url: Optional[str] = Field(
        default=None, description="Optional LinkedIn URL (manual entry fallback provided)"
    )
    current_role: str = Field(..., description="Target role (e.g. Senior K-12 School Consultant)")
    organization: str = Field(default="Not specified")
    location: str = Field(default="Not specified")
    professional_summary: str = Field(
        default="", description="Summary of background, leadership, and school network experience"
    )
    experience_years: float = Field(default=15.0, ge=0.0)
    school_ecosystem_indicators: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)

    @field_validator("school_ecosystem_indicators", "skills", mode="before")
    @classmethod
    def parse_string_to_list(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        if isinstance(v, str):
            if not v.strip() or v.strip().lower() == "not specified":
                return []
            delimiters = [";", "|", ","]
            chosen = ";"
            for d in delimiters:
                if d in v:
                    chosen = d
                    break
            return [item.strip() for item in v.split(chosen) if item.strip()]
        return []

    def get_full_text_representation(self) -> str:
        indicators_str = " ".join(self.school_ecosystem_indicators)
        skills_str = " ".join(self.skills)
        parts = [
            f"Role: {self.current_role}.",
            f"Organization: {self.organization}.",
            f"Location: {self.location}.",
            f"Summary: {self.professional_summary}.",
            f"School Ecosystem Experience: {indicators_str}.",
            f"Skills: {skills_str}.",
            f"Experience: {self.experience_years} years.",
        ]
        return " ".join([p for p in parts if p.strip()])


class ComponentScores(BaseModel):
    """Component scores normalized strictly between 0 and 100."""

    semantic_similarity: float = Field(ge=0.0, le=100.0)
    school_ecosystem: float = Field(ge=0.0, le=100.0)
    leadership: float = Field(ge=0.0, le=100.0)
    multi_school: float = Field(ge=0.0, le=100.0)
    skills_experience: float = Field(ge=0.0, le=100.0)


class RankedCandidate(BaseModel):
    """Candidate with calculated relevance score, component breakdown, and explanation."""

    rank: int
    profile: CandidateProfile
    relevance_score: float = Field(
        ge=0.0, le=100.0, description="Overall weighted relevance score (0-100)"
    )
    component_scores: ComponentScores
    explanation: str
    key_matching_factors: List[str] = Field(default_factory=list)
    uncertainties_or_missing: List[str] = Field(default_factory=list)
    ranking_method: str = Field(
        default="TF-IDF Cosine Similarity + Multi-Component Weighted Model"
    )
