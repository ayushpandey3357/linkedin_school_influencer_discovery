"""
Profile Analyzer module.
Extracts and derives structured professional attributes, distinguishes between
explicit evidence and inferred attributes, and flags missing information.
Never fabricates relationships or sensitive personal characteristics.
"""

import re
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from app.models import CandidateProfile, ReferenceProfile


class ProfileAnalysisReport(BaseModel):
    """Structured analysis report of a professional profile."""

    role: str
    organization: str
    location: str
    experience_years: float
    seniority: str
    education_sector: str
    ecosystem_involvement: str
    is_school_leader: bool
    is_consultant_or_advisor: bool
    has_multi_school_experience: bool
    is_entrepreneur: bool
    is_owner_or_trustee: bool
    explicit_evidence: List[str] = Field(default_factory=list)
    inferred_attributes: Dict[str, Any] = Field(default_factory=dict)
    missing_fields_or_uncertainties: List[str] = Field(default_factory=list)
    analysis_disclaimer: str = (
        "Analysis derived strictly from supplied profile fields. "
        "No private connections or external relationships are inferred."
    )


class ProfileAnalyzer:
    """Modular rule- and heuristic-based analyzer for education ecosystem professionals."""

    # Keywords for domain derivation
    LEADERSHIP_KEYWORDS = [
        "principal", "headmistress", "headmaster", "director", "trustee", "founder",
        "co-founder", "president", "dean", "vice principal", "provincial", "secretary general",
        "chief academic officer", "executive director", "managing director", "promoter", "owner"
    ]

    CONSULTING_KEYWORDS = [
        "consultant", "advisor", "advisory", "turnaround", "mentor", "mentoring",
        "inspection", "assessor", "auditor", "alliances", "partnerships", "accreditation"
    ]

    MULTI_SCHOOL_KEYWORDS = [
        "multi-school", "multi-campus", "cluster", "network", "schools", "campuses",
        "consortium", "chain", "society", "trust", "group of schools", "association"
    ]

    OWNER_TRUSTEE_KEYWORDS = [
        "trustee", "owner", "promoter", "managing trustee", "founder & trustee",
        "founding board", "patron", "correspondent"
    ]

    ENTREPRENEUR_KEYWORDS = [
        "founder", "co-founder", "entrepreneur", "founded", "launched", "bootstrapped"
    ]

    K12_SECTOR_KEYWORDS = [
        "k-12", "k12", "cbse", "icse", "cisce", "cambridge", "igcse", "ib dp", "ib pyp",
        "pre-primary", "preschool", "secondary school", "senior secondary", "high school",
        "boarding school", "residential school", "convent school", "vidya mandir"
    ]

    @classmethod
    def analyze(cls, profile: Union[CandidateProfile, ReferenceProfile]) -> ProfileAnalysisReport:
        """Analyzes a profile and returns an evidence-backed structured analysis report."""
        text_corpus = (
            f"{profile.current_role} {profile.organization} {profile.professional_summary} "
            f"{' '.join(profile.school_ecosystem_indicators)} {' '.join(profile.skills)}"
        ).lower()

        explicit_evidence: List[str] = []
        uncertainties: List[str] = []

        # 1. Field Completeness Check
        if not profile.organization or profile.organization.lower() in ["not specified", "none", ""]:
            uncertainties.append("Organization not specified")
        else:
            explicit_evidence.append(f"Associated with organization: {profile.organization}")

        if not profile.location or profile.location.lower() in ["not specified", "none", ""]:
            uncertainties.append("Geographic location not specified")
        else:
            explicit_evidence.append(f"Located in: {profile.location}")

        if profile.experience_years <= 0:
            uncertainties.append("Experience years not recorded or zero")
        else:
            explicit_evidence.append(f"Recorded professional tenure: {profile.experience_years} years")

        if not profile.school_ecosystem_indicators:
            uncertainties.append("No explicit school-ecosystem indicators supplied")
        else:
            explicit_evidence.append(
                f"Supplied {len(profile.school_ecosystem_indicators)} explicit school ecosystem indicators"
            )

        # 2. Derive Attributes
        role_lower = profile.current_role.lower()

        # Seniority
        seniority = cls._derive_seniority(role_lower, profile.experience_years, text_corpus)

        # Education Sector
        sector = cls._derive_sector(text_corpus)

        # School Leadership
        is_leader = any(k in text_corpus for k in cls.LEADERSHIP_KEYWORDS)
        if is_leader:
            matched = [k for k in cls.LEADERSHIP_KEYWORDS if k in text_corpus]
            explicit_evidence.append(f"Leadership indicators detected: {', '.join(matched[:3])}")

        # Consulting / Advisory
        is_consultant = any(k in text_corpus for k in cls.CONSULTING_KEYWORDS)
        if is_consultant:
            matched = [k for k in cls.CONSULTING_KEYWORDS if k in text_corpus]
            explicit_evidence.append(f"Advisory/Consulting indicators detected: {', '.join(matched[:3])}")

        # Multi-school experience
        has_multi_school = False
        multi_matches = [k for k in cls.MULTI_SCHOOL_KEYWORDS if k in text_corpus]
        # Check for numeric school indicators like "10 schools", "15+ campuses", etc.
        numeric_multi = bool(re.search(r"\b(\d+)\+?\s*(schools|campuses|institutions|branches)", text_corpus))
        if multi_matches or numeric_multi:
            has_multi_school = True
            explicit_evidence.append("Multi-school / network-scale experience detected")

        # Owner / Trustee
        is_owner_trustee = any(k in text_corpus for k in cls.OWNER_TRUSTEE_KEYWORDS)
        if is_owner_trustee:
            explicit_evidence.append("School ownership or board trusteeship indicator present")

        # Entrepreneurship
        is_entrepreneur = any(k in text_corpus for k in cls.ENTREPRENEUR_KEYWORDS)
        if is_entrepreneur:
            explicit_evidence.append("Entrepreneurial/founding role identified")

        # Ecosystem Involvement Level
        ecosystem_involvement = cls._derive_ecosystem_involvement(
            is_leader, is_consultant, has_multi_school, is_owner_trustee, sector
        )

        inferred = {
            "seniority_level": seniority,
            "target_sector": sector,
            "ecosystem_depth": ecosystem_involvement,
            "governance_presence": "Board/Trustee Level" if is_owner_trustee else "Executive/Operational",
            "cross_institutional_reach": "High (Multi-institution)" if has_multi_school else "Single Institution / Not Specified",
        }

        return ProfileAnalysisReport(
            role=profile.current_role,
            organization=profile.organization,
            location=profile.location,
            experience_years=profile.experience_years,
            seniority=seniority,
            education_sector=sector,
            ecosystem_involvement=ecosystem_involvement,
            is_school_leader=is_leader,
            is_consultant_or_advisor=is_consultant,
            has_multi_school_experience=has_multi_school,
            is_entrepreneur=is_entrepreneur,
            is_owner_or_trustee=is_owner_trustee,
            explicit_evidence=explicit_evidence,
            inferred_attributes=inferred,
            missing_fields_or_uncertainties=uncertainties,
        )

    @classmethod
    def _derive_seniority(cls, role_lower: str, exp_years: float, corpus: str) -> str:
        if any(k in role_lower for k in ["trustee", "owner", "managing director", "president", "founder", "director"]):
            return "Executive / Board Level"
        if any(k in role_lower for k in ["principal", "headmistress", "headmaster", "dean", "provincial"]):
            return "School Head / Principal"
        if any(k in role_lower for k in ["senior", "lead", "head of", "vp", "vice principal", "cluster"]):
            return "Senior School Executive"
        if exp_years >= 15:
            return "Veteran Educator (15+ yrs)"
        if exp_years >= 7:
            return "Mid-Senior Level"
        return "Mid-Level Professional"

    @classmethod
    def _derive_sector(cls, corpus: str) -> str:
        k12_count = sum(1 for k in cls.K12_SECTOR_KEYWORDS if k in corpus)
        if k12_count >= 2 or "k-12" in corpus or "k12" in corpus or "school" in corpus:
            return "K-12 School Education"
        if "higher ed" in corpus or "university" in corpus or "professor" in corpus:
            return "Higher Education / Tertiary"
        if "edtech" in corpus:
            return "EdTech / Learning Solutions"
        return "General / Non-School Domain"

    @classmethod
    def _derive_ecosystem_involvement(
        cls,
        is_leader: bool,
        is_consultant: bool,
        has_multi: bool,
        is_owner: bool,
        sector: str,
    ) -> str:
        if sector != "K-12 School Education" and not is_leader and not is_consultant:
            return "Low / Non-School"
        score = 0
        if is_owner:
            score += 3
        if is_leader:
            score += 2
        if is_consultant:
            score += 2
        if has_multi:
            score += 2
        if sector == "K-12 School Education":
            score += 1

        if score >= 6:
            return "Very High (Institutional Key Influencer)"
        if score >= 4:
            return "High (Ecosystem Decision-Maker)"
        if score >= 2:
            return "Moderate (School Practitioner)"
        return "Limited School Involvement"
