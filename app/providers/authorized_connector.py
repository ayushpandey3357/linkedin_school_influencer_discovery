"""
Authorized Connector Provider (Interface & Stub).
Designed for enterprise OAuth/official partner API integrations.
Complies with security policies: unauthorized web scraping is strictly disallowed.
"""

import os
from typing import Any, Dict, List, Optional
from app.models import CandidateProfile
from app.providers.base import ProfileProvider


class AuthorizedConnectorProvider(ProfileProvider):
    """Provider interface for lawful, authorized partner APIs (e.g. LinkedIn Partner API, CRM connector)."""

    def __init__(self, api_key: Optional[str] = None, client_id: Optional[str] = None):
        self.api_key = api_key or os.getenv("LINKEDIN_API_KEY", "")
        self.client_id = client_id or os.getenv("LINKEDIN_CLIENT_ID", "")
        self.is_configured = bool(self.api_key and self.client_id)

    def is_available(self) -> bool:
        """Indicates whether valid enterprise credentials have been supplied."""
        return self.is_configured

    def get_profiles(self, filters: Optional[Dict[str, Any]] = None) -> List[CandidateProfile]:
        """Retrieves profiles from an authorized API connector."""
        if not self.is_configured:
            # Raise an explicit, informative error directing the user to compliant modes
            raise RuntimeError(
                "Authorized Connector Mode is currently not configured with enterprise OAuth credentials. "
                "Per compliance policy, unauthorized scraping is disabled. "
                "Please switch to 'TEST DATASET MODE' or 'IMPORTED DATA MODE' to discover profiles."
            )
        # Stub for future official enterprise API call
        return []

    def get_total_count(self) -> int:
        return 0

    def get_mode_name(self) -> str:
        return "AUTHORIZED CONNECTOR MODE (Enterprise Partner API)"

    def get_provenance_summary(self) -> Dict[str, str]:
        status_text = "Configured" if self.is_configured else "Unconfigured (Enterprise API Key Required)"
        return {
            "source_type": "AUTHORIZED CONNECTOR",
            "compliance_status": "Strict (No unauthorized scraping allowed)",
            "connector_status": status_text,
            "live_scraping": "DISABLED",
            "profile_count": "0",
            "guidance": (
                "To connect real LinkedIn profiles lawfully, an authorized enterprise API "
                "integration (e.g. LinkedIn Learning or Partner API) is required. "
                "In the absence of enterprise credentials, use Test Dataset Mode or CSV Import."
            ),
        }
