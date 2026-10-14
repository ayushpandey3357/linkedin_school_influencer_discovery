"""
CSV Profile Provider.
Loads, validates, and manages user-imported profile records.
Preserves audit provenance and enforces strict schema compliance.
"""

from datetime import datetime, timezone
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from app.models import CandidateProfile
from app.providers.base import ProfileProvider


class CSVProfileProvider(ProfileProvider):
    """Loads and validates candidate profiles from a lawfully obtained CSV file or stream."""

    REQUIRED_COLUMNS = ["profile_id", "name", "linkedin_url", "current_role"]

    def __init__(
        self,
        source: Optional[Union[str, Path, io.StringIO, io.BytesIO, pd.DataFrame]] = None,
        source_description: str = "User-imported CSV dataset",
        source_url: str = "local://imported_dataset.csv",
    ):
        self.source_description = source_description
        self.source_url = source_url
        self.import_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self._profiles: List[CandidateProfile] = []
        self._validation_errors: List[str] = []

        if source is not None:
            self.load_from_source(source)

    def load_from_source(
        self, source: Union[str, Path, io.StringIO, io.BytesIO, pd.DataFrame]
    ) -> int:
        """Parses CSV source, validates required columns, deduplicates, and normalizes records."""
        self._validation_errors.clear()

        if isinstance(source, pd.DataFrame):
            df = source.copy()
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"CSV file not found: {path}")
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(source)

        # Normalize column headers (strip spaces, lowercase)
        column_mapping = {col: col.strip().lower() for col in df.columns}
        df.rename(columns=column_mapping, inplace=True)

        # Validate required columns
        missing_required = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_required:
            err = f"CSV is missing mandatory columns: {', '.join(missing_required)}"
            self._validation_errors.append(err)
            raise ValueError(err)

        seen_ids = set()
        seen_urls = set()
        valid_profiles: List[CandidateProfile] = []

        for idx, row in df.iterrows():
            row_dict = row.to_dict()

            pid = str(row_dict.get("profile_id", "")).strip()
            name = str(row_dict.get("name", "")).strip()
            url = str(row_dict.get("linkedin_url", "")).strip()
            role = str(row_dict.get("current_role", "")).strip()

            if not pid or not name or not url or not role:
                self._validation_errors.append(
                    f"Row {idx + 1} skipped: one or more required fields (profile_id, name, linkedin_url, current_role) are empty."
                )
                continue

            norm_id = pid.upper()
            norm_url = url.lower()

            # Deduplication: check ID and URL (Never merge purely by name)
            if norm_id in seen_ids:
                self._validation_errors.append(f"Row {idx + 1}: duplicate profile_id '{pid}' skipped.")
                continue
            if norm_url in seen_urls:
                self._validation_errors.append(f"Row {idx + 1}: duplicate URL '{url}' skipped.")
                continue

            seen_ids.add(norm_id)
            seen_urls.add(norm_url)

            # Build CandidateProfile with strict provenance
            profile = CandidateProfile(
                profile_id=pid,
                name=name,
                linkedin_url=url,
                current_role=role,
                organization=str(row_dict.get("organization", "Not specified")),
                location=str(row_dict.get("location", "Not specified")),
                professional_summary=str(row_dict.get("professional_summary", "")),
                experience_years=row_dict.get("experience_years", 0.0),
                school_ecosystem_indicators=row_dict.get("school_ecosystem_indicators", []),
                skills=row_dict.get("skills", []),
                source_type="IMPORTED DATA",
                source_description=str(row_dict.get("source_description", self.source_description)),
                source_url=str(row_dict.get("source_url", self.source_url)),
                last_verified_at=str(row_dict.get("last_verified_at", self.import_timestamp)),
            )
            valid_profiles.append(profile)

        self._profiles = valid_profiles
        return len(self._profiles)

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
        return "IMPORTED DATA MODE (User CSV)"

    def get_validation_errors(self) -> List[str]:
        return list(self._validation_errors)

    def get_provenance_summary(self) -> Dict[str, str]:
        return {
            "source_type": "IMPORTED DATA",
            "compliance_status": "User-supplied lawful data",
            "source_description": self.source_description,
            "source_url": self.source_url,
            "import_timestamp": self.import_timestamp,
            "live_scraping": "DISABLED",
            "profile_count": str(len(self._profiles)),
            "disclaimer": (
                "Data provenance reflects user-supplied CSV input. "
                "The application does not independently verify employment or profile accuracy."
            ),
        }
