"""
Synthetic Dataset Provider.
Provides an offline, benchmark dataset of education professionals, former principals,
trustees, and consultants.
Strictly labeled as SYNTHETIC TEST DATA to comply with ethical AI and data integrity standards.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.config import AppConfig
from app.models import CandidateProfile
from app.providers.base import ProfileProvider


class SyntheticDatasetProvider(ProfileProvider):
    """Loads and serves synthetic benchmark profiles for offline evaluation and testing."""

    def __init__(self, data_file: Optional[Path] = None):
        self.data_file = data_file or AppConfig.SYNTHETIC_DATA_FILE
        self._profiles: List[CandidateProfile] = []
        self._load_and_validate()

    def _load_and_validate(self) -> None:
        """Loads JSON file, validates schema with Pydantic, and deduplicates safely."""
        if not self.data_file.exists():
            raise FileNotFoundError(f"Synthetic dataset file not found at: {self.data_file}")

        with open(self.data_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise ValueError("Synthetic dataset must be a JSON array of profile objects.")

        seen_ids = set()
        seen_urls = set()
        validated: List[CandidateProfile] = []

        for item in raw_data:
            # Force provenance adherence
            item["source_type"] = "SYNTHETIC TEST DATA"
            profile = CandidateProfile(**item)

            norm_id = profile.profile_id.strip().upper()
            norm_url = profile.linkedin_url.strip().lower()

            # Deduplication by stable profile ID or normalized URL (never by name)
            if norm_id in seen_ids or norm_url in seen_urls:
                continue

            seen_ids.add(norm_id)
            seen_urls.add(norm_url)
            validated.append(profile)

        self._profiles = validated

    def get_profiles(self, filters: Optional[Dict[str, Any]] = None) -> List[CandidateProfile]:
        """Returns candidate profiles, optionally filtered."""
        if not filters:
            return list(self._profiles)

        results = []
        role_filter = filters.get("role_keyword", "").strip().lower()
        loc_filter = filters.get("location_keyword", "").strip().lower()
        org_filter = filters.get("org_keyword", "").strip().lower()
        min_exp = float(filters.get("min_experience", 0.0))

        for p in self._profiles:
            if role_filter and role_filter not in p.current_role.lower():
                continue
            if loc_filter and loc_filter not in p.location.lower():
                continue
            if org_filter and org_filter not in p.organization.lower():
                continue
            if p.experience_years < min_exp:
                continue
            results.append(p)

        return results

    def get_total_count(self) -> int:
        return len(self._profiles)

    def get_mode_name(self) -> str:
        return "TEST DATASET MODE (Synthetic Benchmark)"

    def get_provenance_summary(self) -> Dict[str, str]:
        return {
            "source_type": "SYNTHETIC TEST DATA",
            "compliance_status": "Fully compliant (Synthetic/Non-Personal)",
            "verification_method": "Synthetic benchmark created for algorithm evaluation",
            "live_scraping": "DISABLED",
            "profile_count": str(len(self._profiles)),
            "disclaimer": (
                "Profiles in this mode are synthetic constructs created for testing, "
                "ranking demonstration, and algorithmic evaluation. They do not represent real persons "
                "and must not be treated as actual LinkedIn profiles."
            ),
        }
