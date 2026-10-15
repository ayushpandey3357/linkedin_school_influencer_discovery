"""
Configuration settings for EDXSO School Influencer Discovery Tool.
"""

from enum import Enum
import os
from pathlib import Path
from pydantic import BaseModel, Field

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DB_PATH = DATA_DIR / "search_history.db"
SYNTHETIC_DATA_PATH = DATA_DIR / "synthetic_profiles.json"
SAMPLE_CSV_PATH = DATA_DIR / "sample_import.csv"


class OperatingMode(str, Enum):
    TEST_DATASET = "TEST_DATASET"
    IMPORTED_DATA = "IMPORTED_DATA"
    AUTHORIZED_CONNECTOR = "AUTHORIZED_CONNECTOR"


class ScoringWeights(BaseModel):
    semantic_similarity: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Weight for semantic embedding / TF-IDF text similarity (0-1)",
    )
    school_ecosystem: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for school education sector involvement & keywords (0-1)",
    )
    leadership: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for school leadership (principal, trustee, director, head) (0-1)",
    )
    multi_school: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for multi-school advisory, consulting, or network experience (0-1)",
    )
    skills_experience: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Weight for domain skills and experience depth (0-1)",
    )

    @classmethod
    def default(cls) -> "ScoringWeights":
        """Returns a default ScoringWeights instance."""
        return cls()

    def validate_weights(self) -> bool:
        total = (
            self.semantic_similarity
            + self.school_ecosystem
            + self.leadership
            + self.multi_school
            + self.skills_experience
        )
        return abs(total - 1.0) < 1e-4

    def normalize(self) -> "ScoringWeights":
        total = (
            self.semantic_similarity
            + self.school_ecosystem
            + self.leadership
            + self.multi_school
            + self.skills_experience
        )
        if total <= 0:
            return ScoringWeights()
        sem = round(self.semantic_similarity / total, 4)
        eco = round(self.school_ecosystem / total, 4)
        lead = round(self.leadership / total, 4)
        multi = round(self.multi_school / total, 4)
        skill = round(1.0 - (sem + eco + lead + multi), 4)
        return ScoringWeights(
            semantic_similarity=sem,
            school_ecosystem=eco,
            leadership=lead,
            multi_school=multi,
            skills_experience=skill,
        )


class AppConfig:
    """Application configuration parameters loaded from environment or defaults."""

    DEFAULT_MODE: OperatingMode = OperatingMode(
        os.getenv("DEFAULT_OPERATING_MODE", OperatingMode.TEST_DATASET.value)
    )
    DATABASE_PATH: Path = Path(os.getenv("DATABASE_PATH", str(DEFAULT_DB_PATH)))
    SYNTHETIC_DATA_FILE: Path = SYNTHETIC_DATA_PATH
    SAMPLE_CSV_FILE: Path = SAMPLE_CSV_PATH
    DEFAULT_WEIGHTS: ScoringWeights = ScoringWeights()
    EMBEDDING_ENGINE: str = os.getenv("EMBEDDING_ENGINE", "auto")
