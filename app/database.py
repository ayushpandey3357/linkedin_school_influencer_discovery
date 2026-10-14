"""
SQLite database persistence layer for search sessions, discovery results, and imported profiles.
Employs parameterized queries and transactions to guarantee safety.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.config import AppConfig, ScoringWeights
from app.models import CandidateProfile, RankedCandidate, ReferenceProfile


class Database:
    """Manages SQLite storage for profiles, search sessions, and ranked results."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or AppConfig.DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_schema(self) -> None:
        """Initializes tables, indexes, and triggers if they do not exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Search sessions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS search_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    dataset_mode TEXT NOT NULL,
                    reference_role TEXT NOT NULL,
                    reference_organization TEXT,
                    reference_summary TEXT,
                    reference_payload TEXT NOT NULL,
                    weights_payload TEXT NOT NULL,
                    ranking_engine TEXT NOT NULL,
                    total_evaluated INTEGER NOT NULL,
                    total_returned INTEGER NOT NULL
                );
                """
            )

            # Search results table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS search_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    profile_id TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    current_role TEXT NOT NULL,
                    organization TEXT,
                    location TEXT,
                    linkedin_url TEXT,
                    relevance_score REAL NOT NULL,
                    semantic_score REAL NOT NULL,
                    ecosystem_score REAL NOT NULL,
                    leadership_score REAL NOT NULL,
                    multi_school_score REAL NOT NULL,
                    skills_score REAL NOT NULL,
                    explanation TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES search_sessions(id) ON DELETE CASCADE
                );
                """
            )

            # Imported profiles table (for persistent imported CSV datasets)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS imported_profiles (
                    profile_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    linkedin_url TEXT NOT NULL,
                    current_role TEXT NOT NULL,
                    organization TEXT,
                    location TEXT,
                    professional_summary TEXT,
                    experience_years REAL DEFAULT 0.0,
                    school_ecosystem_indicators TEXT,
                    skills TEXT,
                    source_type TEXT NOT NULL,
                    source_description TEXT,
                    source_url TEXT,
                    last_verified_at TEXT,
                    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            # Indexes for performance
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_search_results_session ON search_results(session_id);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_search_results_profile ON search_results(profile_id);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_search_sessions_date ON search_sessions(created_at DESC);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_imported_profiles_url ON imported_profiles(linkedin_url);"
            )

            conn.commit()

    def save_search_session(
        self,
        dataset_mode: str,
        reference_profile: ReferenceProfile,
        weights: ScoringWeights,
        ranking_engine: str,
        total_evaluated: int,
        ranked_candidates: List[RankedCandidate],
    ) -> int:
        """Saves a search session along with its ranked candidate items in a single transaction."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO search_sessions (
                    dataset_mode, reference_role, reference_organization,
                    reference_summary, reference_payload, weights_payload,
                    ranking_engine, total_evaluated, total_returned
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    dataset_mode,
                    reference_profile.current_role,
                    reference_profile.organization,
                    reference_profile.professional_summary,
                    json.dumps(reference_profile.model_dump()),
                    json.dumps(weights.model_dump()),
                    ranking_engine,
                    total_evaluated,
                    len(ranked_candidates),
                ),
            )
            session_id = cursor.lastrowid

            for cand in ranked_candidates:
                cursor.execute(
                    """
                    INSERT INTO search_results (
                        session_id, profile_id, rank, name, current_role,
                        organization, location, linkedin_url, relevance_score,
                        semantic_score, ecosystem_score, leadership_score,
                        multi_school_score, skills_score, explanation, source_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        session_id,
                        cand.profile.profile_id,
                        cand.rank,
                        cand.profile.name,
                        cand.profile.current_role,
                        cand.profile.organization,
                        cand.profile.location,
                        cand.profile.linkedin_url,
                        round(cand.relevance_score, 2),
                        round(cand.component_scores.semantic_similarity, 2),
                        round(cand.component_scores.school_ecosystem, 2),
                        round(cand.component_scores.leadership, 2),
                        round(cand.component_scores.multi_school, 2),
                        round(cand.component_scores.skills_experience, 2),
                        cand.explanation,
                        cand.profile.source_type,
                    ),
                )

            conn.commit()
            return session_id

    def list_search_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves past search sessions in reverse chronological order."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, created_at, dataset_mode, reference_role,
                       reference_organization, ranking_engine,
                       total_evaluated, total_returned
                FROM search_sessions
                ORDER BY created_at DESC
                LIMIT ?;
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_session_results(self, session_id: int) -> List[Dict[str, Any]]:
        """Retrieves all candidate results for a given search session."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM search_results
                WHERE session_id = ?
                ORDER BY rank ASC;
                """,
                (session_id,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def save_imported_profiles(self, profiles: List[CandidateProfile]) -> int:
        """Inserts or updates imported profiles in the SQLite store."""
        inserted_count = 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for p in profiles:
                cursor.execute(
                    """
                    INSERT INTO imported_profiles (
                        profile_id, name, linkedin_url, current_role,
                        organization, location, professional_summary,
                        experience_years, school_ecosystem_indicators,
                        skills, source_type, source_description,
                        source_url, last_verified_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(profile_id) DO UPDATE SET
                        name=excluded.name,
                        linkedin_url=excluded.linkedin_url,
                        current_role=excluded.current_role,
                        organization=excluded.organization,
                        location=excluded.location,
                        professional_summary=excluded.professional_summary,
                        experience_years=excluded.experience_years,
                        school_ecosystem_indicators=excluded.school_ecosystem_indicators,
                        skills=excluded.skills,
                        source_type=excluded.source_type,
                        source_description=excluded.source_description,
                        source_url=excluded.source_url,
                        last_verified_at=excluded.last_verified_at;
                    """,
                    (
                        p.profile_id,
                        p.name,
                        p.linkedin_url,
                        p.current_role,
                        p.organization,
                        p.location,
                        p.professional_summary,
                        p.experience_years,
                        "; ".join(p.school_ecosystem_indicators),
                        "; ".join(p.skills),
                        p.source_type,
                        p.source_description,
                        p.source_url,
                        p.last_verified_at,
                    ),
                )
                inserted_count += 1
            conn.commit()
        return inserted_count

    def get_imported_profiles(self) -> List[CandidateProfile]:
        """Loads all stored imported profiles."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM imported_profiles ORDER BY profile_id ASC;")
            rows = cursor.fetchall()
            profiles = []
            for r in rows:
                p_dict = dict(r)
                profiles.append(
                    CandidateProfile(
                        profile_id=p_dict["profile_id"],
                        name=p_dict["name"],
                        linkedin_url=p_dict["linkedin_url"],
                        current_role=p_dict["current_role"],
                        organization=p_dict["organization"],
                        location=p_dict["location"],
                        professional_summary=p_dict["professional_summary"],
                        experience_years=p_dict["experience_years"],
                        school_ecosystem_indicators=p_dict["school_ecosystem_indicators"],
                        skills=p_dict["skills"],
                        source_type=p_dict["source_type"],
                        source_description=p_dict.get("source_description", ""),
                        source_url=p_dict.get("source_url", ""),
                        last_verified_at=p_dict.get("last_verified_at", ""),
                    )
                )
            return profiles
