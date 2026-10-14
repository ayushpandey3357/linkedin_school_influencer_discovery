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

    def rank_candidates(
        self,
        reference_profile: ReferenceProfile,
        candidates: List[CandidateProfile],
        min_threshold: float = 0.0,
        top_k: Optional[int] = None,
    ) -> List[RankedCandidate]:
        """Ranks a list of candidate profiles against a reference profile.
        Strictly excludes the reference profile itself.
        """
        if not candidates:
            return []

        # 1. Filter out candidate that matches the reference profile (by URL or exact name)
        eligible_candidates: List[CandidateProfile] = []
        ref_url = (reference_profile.linkedin_url or "").strip().lower()
        ref_name = reference_profile.name.strip().lower()

        for c in candidates:
            cand_url = c.linkedin_url.strip().lower()
            cand_name = c.name.strip().lower()

            # Exclusion logic: never recommend the reference profile to itself
            if ref_url and cand_url and ref_url == cand_url:
                continue
            if ref_name and ref_name != "reference professional" and ref_name == cand_name:
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
            sem_score = semantic_scores[idx]
            cand_analysis = ProfileAnalyzer.analyze(cand)

            eco_score = self._compute_ecosystem_relevance(cand, cand_analysis)
            lead_score = self._compute_leadership_relevance(cand, cand_analysis, ref_analysis)
            multi_score = self._compute_multi_school_relevance(cand, cand_analysis)
            skills_score = self._compute_skills_relevance(cand, reference_profile)

            # Composite Score = Sum(weight_i * score_i)
            overall_score = (
                (self.weights.semantic_similarity * sem_score)
                + (self.weights.school_ecosystem * eco_score)
                + (self.weights.leadership * lead_score)
                + (self.weights.multi_school * multi_score)
                + (self.weights.skills_experience * skills_score)
            )
            overall_score = float(np.clip(overall_score, 0.0, 100.0))

            if overall_score < min_threshold:
                continue

            comp_scores = ComponentScores(
                semantic_similarity=round(sem_score, 2),
                school_ecosystem=round(eco_score, 2),
                leadership=round(lead_score, 2),
                multi_school=round(multi_score, 2),
                skills_experience=round(skills_score, 2),
            )

            # Matching factors and uncertainties
            key_factors, uncertainties = self._extract_matching_factors(
                cand, comp_scores, cand_analysis
            )

            # Construct temporary candidate record
            cand_item = RankedCandidate(
                rank=0,  # Assigned after sorting
                profile=cand,
                relevance_score=round(overall_score, 2),
                component_scores=comp_scores,
                explanation="",  # Generated by explanation engine
                key_matching_factors=key_factors,
                uncertainties_or_missing=uncertainties,
                ranking_method=self.similarity_engine.active_engine,
            )
            ranked_list.append((overall_score, cand_item))

        # 5. Deterministic sorting: sort primarily by score descending, then profile_id
        ranked_list.sort(key=lambda x: (x[0], x[1].profile.profile_id), reverse=True)

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
