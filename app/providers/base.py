"""
Base abstract provider interface for candidate profile retrieval.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from app.models import CandidateProfile


class ProfileProvider(ABC):
    """Abstract interface for all profile data sources."""

    @abstractmethod
    def get_profiles(self, filters: Optional[Dict[str, Any]] = None) -> List[CandidateProfile]:
        """Retrieves and returns normalized candidate profiles matching optional filters."""
        raise NotImplementedError

    @abstractmethod
    def get_total_count(self) -> int:
        """Returns the total number of candidate records in this provider's repository."""
        raise NotImplementedError

    @abstractmethod
    def get_mode_name(self) -> str:
        """Returns the human-readable operating mode name."""
        raise NotImplementedError

    @abstractmethod
    def get_provenance_summary(self) -> Dict[str, str]:
        """Returns provenance, verification details, and legal compliance disclosures."""
        raise NotImplementedError
