"""
Relevance Explanation Engine.
Produces evidence-backed, factual explanations explaining why each candidate ranked
at their respective position, citing concrete profile fields, component scores, and provenance.
Does not invent connections or qualifications. Works 100% offline without an external LLM.
"""

from typing import List
from app.models import RankedCandidate


class ExplanationEngine:
    """Generates transparent, evidence-based matching rationales from candidate data and scores."""

    @classmethod
    def generate_explanation(cls, candidate: RankedCandidate) -> str:
        """Constructs an evidence-based explanation for a ranked candidate."""
        p = candidate.profile
        scores = candidate.component_scores
        score = candidate.relevance_score

        # Identify top scoring components
        component_map = [
            ("Semantic Similarity", scores.semantic_similarity),
            ("School Ecosystem Relevance", scores.school_ecosystem),
            ("Leadership Relevance", scores.leadership),
            ("Multi-School / Advisory Reach", scores.multi_school),
            ("Skills & Experience Depth", scores.skills_experience),
        ]
        # Sort components descending
        component_map.sort(key=lambda x: x[1], reverse=True)
        top_drivers = [f"{name} ({val:.1f}/100)" for name, val in component_map[:2] if val > 25.0]

        sentences: List[str] = []

        # 1. Overall Match Assessment & Provenance Tag
        provenance_tag = f"[{p.source_type}]"
        if score >= 75.0:
            qualifier = "Exceptional match"
        elif score >= 60.0:
            qualifier = "Strong match"
        elif score >= 45.0:
            qualifier = "Moderate match"
        elif score >= 30.0:
            qualifier = "Fair match"
        else:
            qualifier = "Low relevance match"

        sentences.append(f"{provenance_tag} {qualifier} with an overall relevance score of {score:.1f}/100.")

        # 2. Concrete role and verified employment evidence
        evidence_points: List[str] = []
        role_lower = p.current_role.lower()
        if any(k in role_lower for k in ["principal", "headmistress", "headmaster", "dean"]):
            evidence_points.append(f"serves or served as {p.current_role}")
        elif any(k in role_lower for k in ["trustee", "owner", "managing director", "promoter"]):
            evidence_points.append(f"holds institutional governance authority as {p.current_role}")
        elif any(k in role_lower for k in ["consultant", "advisor", "advisory"]):
            evidence_points.append(f"operates as an education consultant / advisor ({p.current_role})")
        elif "not specified" not in role_lower:
            evidence_points.append(f"current role is {p.current_role}")

        if p.organization and p.organization.lower() != "not specified":
            evidence_points.append(f"at {p.organization}")

        if p.experience_years > 0:
            evidence_points.append(f"with {p.experience_years:g} years documented experience")

        if evidence_points:
            joined_evidence = ", ".join(evidence_points)
            sentences.append(f"Profile evidence highlights that candidate {joined_evidence}.")

        # 3. Component-specific factual evidence from actual dataset fields
        # A. School ecosystem evidence
        if scores.school_ecosystem >= 40.0 and p.school_ecosystem_indicators:
            eco_kw = [
                "school", "k-12", "cbse", "icse", "cambridge", "ib", "principal",
                "board", "accreditation", "turnaround", "nep", "curriculum"
            ]
            eco_indicators = [
                ind for ind in p.school_ecosystem_indicators
                if any(k in ind.lower() for k in eco_kw) and ind.lower() != "not specified"
            ]
            if eco_indicators:
                sentences.append(f"Ecosystem depth supported by: '{eco_indicators[0]}'.")

        # B. Multi-school / consulting reach evidence
        if scores.multi_school >= 40.0 and p.school_ecosystem_indicators:
            multi_kw = [
                "schools", "campuses", "network", "consortium", "trust", "advis",
                "consult", "turnaround", "cluster", "chain", "multi-"
            ]
            multi_indicators = [
                ind for ind in p.school_ecosystem_indicators
                if any(k in ind.lower() for k in multi_kw) and ind.lower() != "not specified"
            ]
            if multi_indicators:
                chosen = multi_indicators[0]
                if len(multi_indicators) > 1 and len(sentences) > 2 and chosen in sentences[-1]:
                    chosen = multi_indicators[1]
                if len(sentences) <= 2 or chosen not in sentences[-1]:
                    sentences.append(f"Multi-school / advisory reach supported by: '{chosen}'.")

        # 4. Component score drivers
        if top_drivers:
            drivers_str = " and ".join(top_drivers)
            sentences.append(f"Primary algorithmic scoring drivers: {drivers_str}.")
        else:
            sentences.append("Relevance is tempered by lower domain overlap in school education.")

        # 5. Low score context (e.g. for non-education control profiles)
        if score < 35.0:
            sentences.append(
                "Profile exhibits minimal K-12 school ecosystem experience or educational leadership indicators."
            )

        # 6. Missing data notes
        if candidate.uncertainties_or_missing:
            noted = "; ".join(candidate.uncertainties_or_missing[:2])
            sentences.append(f"Note on data completeness: {noted}.")

        return " ".join(sentences)

    @classmethod
    def attach_explanations(cls, candidates: List[RankedCandidate]) -> List[RankedCandidate]:
        """Iterates over a list of candidates and enriches each with its generated explanation."""
        for c in candidates:
            c.explanation = cls.generate_explanation(c)
        return candidates
