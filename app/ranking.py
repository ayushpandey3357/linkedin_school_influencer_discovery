"""
AI/ML Similarity and Ranking Engine.
Computes multi-component transparent relevance scores using semantic text similarity
(Sentence Transformers or Scikit-Learn TF-IDF + Cosine Similarity) along with
domain-specific school ecosystem, leadership, multi-school reach, and skills weights.
"""

import logging
import re
from typing import List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import AppConfig, ScoringWeights
from app.models import (
    CandidateProfile,
    ComponentScores,
    RankedCandidate,
    ReferenceProfile,
)
from app.profile_analyzer import ProfileAnalyzer

logger = logging.getLogger(__name__)

# Attempt to import sentence_transformers if available
_SENTENCE_TRANSFORMER_AVAILABLE = False
_EMBEDDING_MODEL = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMER_AVAILABLE = False


class SimilarityEngine:
    """Manages text vectorization and cosine similarity calculations."""

    def __init__(self, engine_preference: str = "auto"):
        self.engine_preference = engine_preference
        self.active_engine = "TF-IDF (Scikit-Learn) with Cosine Similarity"
        self._init_engine()

    def _init_engine(self) -> None:
        global _EMBEDDING_MODEL
        if (
            self.engine_preference in ["auto", "sentence_transformers"]
            and _SENTENCE_TRANSFORMER_AVAILABLE
        ):
            try:
                if _EMBEDDING_MODEL is None:
                    _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
                self.active_engine = "Sentence-Transformers (all-MiniLM-L6-v2) + Cosine Similarity"
                return
            except Exception as e:
                logger.warning(
                    "Sentence Transformers model load failed, falling back to TF-IDF: %s",
                    e,
                )

        self.active_engine = "TF-IDF (Scikit-Learn) with Cosine Similarity"

    def compute_similarity(
        self, reference_text: str, candidate_texts: List[str]
    ) -> List[float]:
        """Calculates cosine similarity between reference text and each candidate text.
        Returns a list of similarity scores normalized to 0 - 100.
        """
        if not candidate_texts:
            return []

        # Sentence Transformers path
        if "Sentence-Transformers" in self.active_engine and _EMBEDDING_MODEL is not None:
            try:
                corpus = [reference_text] + candidate_texts
                embeddings = _EMBEDDING_MODEL.encode(corpus, show_progress_bar=False)
                ref_emb = embeddings[0].reshape(1, -1)
                cand_embs = embeddings[1:]
                cos_sim = cosine_similarity(ref_emb, cand_embs)[0]
                # Scale from [-1, 1] to [0, 100]
                return [float(np.clip(val * 100.0, 0.0, 100.0)) for val in cos_sim]
            except Exception as e:
                logger.warning("Error computing embedding similarity, falling back to TF-IDF: %s", e)

        # Scikit-Learn TF-IDF Fallback (Works 100% offline, guaranteed)
        corpus = [reference_text] + candidate_texts
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1,
            sublinear_tf=True,
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
            ref_vec = tfidf_matrix[0:1]
            cand_vecs = tfidf_matrix[1:]
            cos_sim = cosine_similarity(ref_vec, cand_vecs)[0]
            return [float(np.clip(val * 100.0, 0.0, 100.0)) for val in cos_sim]
        except Exception as e:
            logger.error("TF-IDF calculation error: %s", e)
            return [0.0] * len(candidate_texts)


def normalize_profile_url(url: Optional[str]) -> str:
    """Normalizes a profile URL by stripping protocol, www, trailing slashes,
    query parameters, and fragments. Returns empty string if missing.
    """
    if not url:
        return ""
    u = url.strip().lower()
    # Remove protocol
    u = re.sub(r"^https?://", "", u)
    # Remove www.
    u = re.sub(r"^www\.", "", u)
    # Remove query params and fragments
    u = u.split("?")[0].split("#")[0]
    # Remove trailing slashes
    u = u.rstrip("/")
    return u


def is_generic_placeholder_url(norm_url: str) -> bool:
    """Detects if a normalized URL is merely a generic root or placeholder
    rather than an individual profile path.
    """
    if not norm_url:
        return True
    generic_roots = {
        "example.com",
        "example.com/profiles",
        "example.com/profiles/reference-profile",
        "linkedin.com",
        "linkedin.com/in",
        "linkedin.com/pub",
    }
    return norm_url in generic_roots


def clean_profile_name(name: str) -> str:
    """Cleans a profile name by stripping parenthetical/bracketed suffixes
    (e.g., '(Reference Profile)', '(Synthetic Profile)', '(Reference Test)', '[Synthetic]')
    and common professional prefixes/honorifics.
    """
    if not name:
        return ""
    # Remove text in parentheses, brackets, or braces
    n = re.sub(r"\s*[\(\[\{].*?[\)\]\}]", "", name).strip().lower()
    # Remove common professional prefixes / honorifics
    n = re.sub(r"^(dr\.|mr\.|mrs\.|ms\.|prof\.|col\.|sister|rev\.)\s+", "", n)
    # Normalize multiple whitespace
    n = re.sub(r"\s+", " ", n).strip()
    return n


def is_reference_profile_match(
    reference: ReferenceProfile, candidate: CandidateProfile
) -> Tuple[bool, str]:
    """Determines whether a candidate profile matches the reference profile,
    and should therefore be strictly excluded from recommendation results.

    Evaluation hierarchy:
    1. Stable Profile ID:
       If reference.profile_id is populated, matches candidate.profile_id (case-insensitive).
    2. Normalized Profile URL:
       If both have valid, non-generic profile URLs that normalize to the same path.
    3. Conservative Multi-Field Duplicate Check:
       Triggered when no stable ID or URL match exists.
       Requires clean base names to match AND multiple strong concordant signals:
       - Role concordance (normalized roles match or have high lexical overlap)
       - Organization concordance (both specify org and they match)
       - Experience concordance (tenures within 1.0 year)
       - Location concordance (both specify location and match)
       - Summary / Indicators concordance (high textual overlap)
       NEVER excludes candidates based on name similarity alone.

    Limitations:
    - Relies on available text fields; if an individual has entirely different role
      and org descriptions with no shared ID or URL, they may not be detected.
    - Conversely, if two distinct professionals coincidentally share the exact same
      name, organization, and role/experience, the conservative check will treat them
      as duplicates.
    """
    # 1. Stable profile ID check
    ref_pid = (reference.profile_id or "").strip().upper()
    cand_pid = candidate.profile_id.strip().upper()
    if ref_pid and cand_pid and ref_pid == cand_pid:
        return True, f"Excluded by stable profile ID: {cand_pid}"

    # 2. Normalized profile URL check
    ref_norm_url = normalize_profile_url(reference.linkedin_url)
    cand_norm_url = normalize_profile_url(candidate.linkedin_url)
    if ref_norm_url and cand_norm_url:
        if not is_generic_placeholder_url(ref_norm_url) and not is_generic_placeholder_url(cand_norm_url):
            if ref_norm_url == cand_norm_url:
                return True, f"Excluded by normalized URL: {cand_norm_url}"

    # 3. Conservative multi-field check
    ref_base_name = clean_profile_name(reference.name)
    cand_base_name = clean_profile_name(candidate.name)

    # If reference name is generic placeholder or empty, or names don't match, do not exclude
    if not ref_base_name or ref_base_name in ["reference professional", "reference leader", "reference test"]:
        return False, "Generic or unassigned reference name"

    if ref_base_name != cand_base_name:
        return False, "Different name"

    # At this point, names match. Now evaluate multi-field concordances:
    matching_signals: List[str] = []

    # A. Role concordance
    ref_role = reference.current_role.strip().lower()
    cand_role = candidate.current_role.strip().lower()
    if ref_role and cand_role:
        if ref_role == cand_role or (len(ref_role) > 10 and (ref_role in cand_role or cand_role in ref_role)):
            matching_signals.append("role")
        else:
            ref_words = set(re.findall(r"\w+", ref_role))
            cand_words = set(re.findall(r"\w+", cand_role))
            if ref_words and cand_words:
                jaccard = len(ref_words & cand_words) / len(ref_words | cand_words)
                if jaccard >= 0.5:
                    matching_signals.append("role_overlap")

    # B. Organization concordance
    ref_org = reference.organization.strip().lower()
    cand_org = candidate.organization.strip().lower()
    invalid_orgs = {"not specified", "none", "", "n/a"}
    if ref_org not in invalid_orgs and cand_org not in invalid_orgs:
        if ref_org == cand_org or ref_org in cand_org or cand_org in ref_org:
            matching_signals.append("organization")

    # C. Location concordance
    ref_loc = reference.location.strip().lower()
    cand_loc = candidate.location.strip().lower()
    invalid_locs = {"not specified", "none", "", "n/a"}
    if ref_loc not in invalid_locs and cand_loc not in invalid_locs:
        if ref_loc == cand_loc or ref_loc in cand_loc or cand_loc in ref_loc:
            matching_signals.append("location")

    # D. Experience years concordance
    if reference.experience_years > 0 and candidate.experience_years > 0:
        if abs(reference.experience_years - candidate.experience_years) <= 1.0:
            matching_signals.append("experience")

    # E. Summary / Indicators text overlap
    ref_summary = reference.professional_summary.strip().lower()
    cand_summary = candidate.professional_summary.strip().lower()
    if ref_summary and cand_summary:
        ref_s_words = set(re.findall(r"\w{4,}", ref_summary))
        cand_s_words = set(re.findall(r"\w{4,}", cand_summary))
        if ref_s_words and cand_s_words:
            overlap = len(ref_s_words & cand_s_words) / min(len(ref_s_words), len(cand_s_words))
            if overlap >= 0.5:
                matching_signals.append("summary_overlap")

    # Conservative rule: NEVER exclude on name alone.
    # Exclude if:
    # - Role and Organization both match, OR
    # - At least 3 independent fields match among {role, organization, location, experience, summary_overlap}
    has_role = any("role" in s for s in matching_signals)
    has_org = "organization" in matching_signals
    if (has_role and has_org) or len(matching_signals) >= 3:
        return True, f"Excluded by conservative multi-field match (name + {', '.join(matching_signals)})"

    return False, "Insufficient field concordance to exclude"


class RankingEngine:
    """Combines semantic similarity and domain heuristic scores into an explainable ranked list."""

    # Curated school ecosystem tokens
    ECOSYSTEM_TOKENS = [
        "school", "k-12", "k12", "cbse", "icse", "cisce", "cambridge", "igcse", "ib",
        "principal", "headmistress", "headmaster", "trustee", "promoter", "board",
        "curriculum", "pedagogy", "accreditation", "turnaround", "nep 2020",
        "admission", "teacher training", "campus", "academic director"
    ]

    LEADERSHIP_TOKENS = [
        "principal", "headmistress", "headmaster", "director", "trustee",
        "managing director", "founder", "co-founder", "dean", "provincial",
        "vice principal", "president", "secretary general", "owner"
    ]

    MULTI_SCHOOL_TOKENS = [
        "multi-school", "multi-campus", "cluster", "network", "consortium",
        "chain", "group of schools", "trust", "advisory", "consultant",
        "alliances", "auditor", "inspection", "inspections"
    ]

    def __init__(
        self,
        similarity_engine: Optional[SimilarityEngine] = None,
        weights: Optional[ScoringWeights] = None,
    ):
        self.similarity_engine = similarity_engine or SimilarityEngine(AppConfig.EMBEDDING_ENGINE)
        self.weights = (weights or AppConfig.DEFAULT_WEIGHTS).normalize()
        assert self.weights.validate_weights(), "Configured scoring weights must sum to 100% (1.0)"

    def rank_candidates(
        self,
        reference_profile: ReferenceProfile,
        candidates: List[CandidateProfile],
        min_threshold: float = 0.0,
        top_k: Optional[int] = None,
    ) -> List[RankedCandidate]:
        """Ranks a list of candidate profiles against a reference profile.
        Strictly excludes the reference profile itself using stable profile IDs,
        normalized URLs, and conservative multi-field matching.
        """
        if not candidates:
            return []

        # 1. Filter out candidate matching reference profile
        eligible_candidates: List[CandidateProfile] = []
        for c in candidates:
            is_match, reason = is_reference_profile_match(reference_profile, c)
            if is_match:
                logger.info("Excluding reference profile candidate %s (%s): %s", c.profile_id, c.name, reason)
                continue
            eligible_candidates.append(c)

        if not eligible_candidates:
            return []

        # 2. Compute Semantic Similarity via ML / TF-IDF
        ref_text = reference_profile.get_full_text_representation()
        cand_texts = [c.get_full_text_representation() for c in eligible_candidates]
        semantic_scores = self.similarity_engine.compute_similarity(ref_text, cand_texts)

        # 3. Analyze reference profile for context
        ref_analysis = ProfileAnalyzer.analyze(reference_profile)

        # 4. Compute Component Scores & Composite Relevance
        ranked_list: List[Tuple[float, RankedCandidate]] = []

        for idx, cand in enumerate(eligible_candidates):
            sem_score = round(semantic_scores[idx], 2)
            cand_analysis = ProfileAnalyzer.analyze(cand)

            eco_score = round(self._compute_ecosystem_relevance(cand, cand_analysis), 2)
            lead_score = round(self._compute_leadership_relevance(cand, cand_analysis, ref_analysis), 2)
            multi_score = round(self._compute_multi_school_relevance(cand, cand_analysis), 2)
            skills_score = round(self._compute_skills_relevance(cand, reference_profile), 2)

            comp_scores = ComponentScores(
                semantic_similarity=sem_score,
                school_ecosystem=eco_score,
                leadership=lead_score,
                multi_school=multi_score,
                skills_experience=skills_score,
            )

            # Composite Score calculated strictly from configured weights and component scores
            overall_score = (
                (self.weights.semantic_similarity * comp_scores.semantic_similarity)
                + (self.weights.school_ecosystem * comp_scores.school_ecosystem)
                + (self.weights.leadership * comp_scores.leadership)
                + (self.weights.multi_school * comp_scores.multi_school)
                + (self.weights.skills_experience * comp_scores.skills_experience)
            )
            overall_score = round(float(np.clip(overall_score, 0.0, 100.0)), 2)

            if overall_score < min_threshold:
                continue

            # Matching factors and uncertainties
            key_factors, uncertainties = self._extract_matching_factors(
                cand, comp_scores, cand_analysis
            )

            # Construct temporary candidate record
            cand_item = RankedCandidate(
                rank=0,  # Assigned after sorting
                profile=cand,
                relevance_score=overall_score,
                component_scores=comp_scores,
                explanation="",  # Generated by explanation engine
                key_matching_factors=key_factors,
                uncertainties_or_missing=uncertainties,
                ranking_method=self.similarity_engine.active_engine,
            )
            ranked_list.append((overall_score, cand_item))

        # 5. Deterministic sorting: sort primarily by score descending, then profile_id ascending
        ranked_list.sort(key=lambda x: (-x[0], x[1].profile.profile_id))

        # 6. Apply top_k and assign ranks
        if top_k is not None and top_k > 0:
            ranked_list = ranked_list[:top_k]

        final_ranked: List[RankedCandidate] = []
        for rank_num, (_, candidate_obj) in enumerate(ranked_list, start=1):
            candidate_obj.rank = rank_num
            final_ranked.append(candidate_obj)

        return final_ranked

    def _compute_ecosystem_relevance(
        self, candidate: CandidateProfile, analysis
    ) -> float:
        """Calculates ecosystem presence (0 - 100)."""
        corpus = (
            f"{candidate.current_role} {candidate.organization} {candidate.professional_summary} "
            f"{' '.join(candidate.school_ecosystem_indicators)}"
        ).lower()

        # Count occurrences of key school ecosystem tokens
        matched_tokens = sum(1 for t in self.ECOSYSTEM_TOKENS if t in corpus)
        token_score = min(matched_tokens * 8.0, 50.0)

        # Bonus for explicit school ecosystem indicators
        indicators_bonus = min(len(candidate.school_ecosystem_indicators) * 10.0, 30.0)

        # Sector alignment
        sector_score = 20.0 if analysis.education_sector == "K-12 School Education" else 0.0

        return float(min(token_score + indicators_bonus + sector_score, 100.0))

    def _compute_leadership_relevance(
        self, candidate: CandidateProfile, cand_analysis, ref_analysis
    ) -> float:
        """Calculates leadership & decision-maker alignment (0 - 100)."""
        corpus = f"{candidate.current_role} {candidate.professional_summary}".lower()
        score = 0.0

        # Exact leadership role
        if cand_analysis.is_owner_or_trustee:
            score += 45.0
        elif cand_analysis.is_school_leader:
            score += 35.0

        # Board / Principal keywords
        matched_leadership = [t for t in self.LEADERSHIP_TOKENS if t in corpus]
        score += min(len(matched_leadership) * 12.0, 30.0)

        # Experience depth
        if candidate.experience_years >= 15:
            score += 25.0
        elif candidate.experience_years >= 8:
            score += 15.0

        return float(min(score, 100.0))

    def _compute_multi_school_relevance(
        self, candidate: CandidateProfile, cand_analysis
    ) -> float:
        """Calculates multi-school reach, advisory, or network experience (0 - 100)."""
        corpus = (
            f"{candidate.current_role} {candidate.professional_summary} "
            f"{' '.join(candidate.school_ecosystem_indicators)}"
        ).lower()

        score = 0.0
        if cand_analysis.has_multi_school_experience:
            score += 40.0
        if cand_analysis.is_consultant_or_advisor:
            score += 30.0

        # Detect numeric school scale (e.g. "15+ schools", "8 campuses")
        num_match = re.search(r"(\d+)\+?\s*(schools|campuses|institutions)", corpus)
        if num_match:
            try:
                count = int(num_match.group(1))
                if count >= 20:
                    score += 30.0
                elif count >= 5:
                    score += 20.0
                else:
                    score += 10.0
            except ValueError:
                score += 15.0

        matched_tokens = sum(1 for t in self.MULTI_SCHOOL_TOKENS if t in corpus)
        score += min(matched_tokens * 8.0, 25.0)

        return float(min(score, 100.0))

    def _compute_skills_relevance(
        self, candidate: CandidateProfile, reference: ReferenceProfile
    ) -> float:
        """Calculates skill and domain overlap (0 - 100)."""
        cand_skills = {s.strip().lower() for s in candidate.skills if s.strip()}
        ref_skills = {s.strip().lower() for s in reference.skills if s.strip()}

        if not ref_skills:
            # Fallback based on experience depth if reference has no explicit skills
            return float(min(candidate.experience_years * 4.0, 100.0))

        if not cand_skills:
            return 20.0  # Baseline

        # Jaccard / Overlap coefficient
        intersection = cand_skills.intersection(ref_skills)
        overlap_ratio = len(intersection) / max(len(ref_skills), 1)
        skill_score = overlap_ratio * 70.0

        # Experience tenure compatibility
        exp_diff = abs(candidate.experience_years - reference.experience_years)
        exp_score = max(30.0 - (exp_diff * 1.5), 0.0)

        return float(min(skill_score + exp_score, 100.0))

    def _extract_matching_factors(
        self,
        candidate: CandidateProfile,
        scores: ComponentScores,
        analysis,
    ) -> Tuple[List[str], List[str]]:
        """Extracts concrete factual matching factors and flags missing or uncertain data."""
        factors: List[str] = []
        uncertainties = list(analysis.missing_fields_or_uncertainties)

        if scores.semantic_similarity >= 40.0:
            factors.append(f"Strong semantic profile alignment ({scores.semantic_similarity:.1f}%)")
        if analysis.is_school_leader:
            factors.append(f"School leadership track record ({analysis.seniority})")
        if analysis.is_owner_or_trustee:
            factors.append("Institutional governance: Trustee / School Owner")
        if analysis.has_multi_school_experience:
            factors.append("Proven multi-school / network-scale reach")
        if analysis.is_consultant_or_advisor:
            factors.append("Active advisory or consulting presence in K-12")
        if scores.school_ecosystem >= 50.0:
            factors.append(f"Deep K-12 ecosystem presence ({scores.school_ecosystem:.1f}%)")

        if not factors:
            factors.append("Baseline professional relevance in education domain")

        return factors, uncertainties
