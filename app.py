"""
EDXSO School Influencer Discovery - Streamlit Dashboard.
A compliant, AI-powered system for discovering similar education leaders,
former principals, school owners, trustees, and K-12 advisors.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.config import AppConfig, OperatingMode, ScoringWeights
from app.database import Database
from app.explanations import ExplanationEngine
from app.exporter import CSVExporter
from app.models import CandidateProfile, RankedCandidate, ReferenceProfile
from app.profile_analyzer import ProfileAnalyzer
from app.providers import (
    AuthorizedConnectorProvider,
    CSVProfileProvider,
    SyntheticDatasetProvider,
)
from app.ranking import RankingEngine, SimilarityEngine

# Set Page Config
st.set_page_config(
    page_title="EDXSO School Influencer Discovery",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-synthetic {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid #FCD34D;
    }
    .badge-imported {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid #C7D2FE;
    }
    .candidate-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .candidate-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-color: #CBD5E1;
    }
    .score-pill {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F766E;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_database() -> Database:
    return Database()


@st.cache_resource
def get_synthetic_provider() -> SyntheticDatasetProvider:
    return SyntheticDatasetProvider()


@st.cache_resource
def get_ranking_engine() -> RankingEngine:
    return RankingEngine()


# Initialize Session State
if "operating_mode" not in st.session_state:
    st.session_state.operating_mode = OperatingMode.TEST_DATASET.value

if "custom_csv_provider" not in st.session_state:
    st.session_state.custom_csv_provider = None

if "last_ranked_results" not in st.session_state:
    st.session_state.last_ranked_results = []

if "last_reference_profile" not in st.session_state:
    st.session_state.last_reference_profile = None

if "selected_candidate_id" not in st.session_state:
    st.session_state.selected_candidate_id = None

db = get_database()
ranking_engine = get_ranking_engine()
synthetic_provider = get_synthetic_provider()

# Sidebar Navigation
st.sidebar.image(
    "https://img.icons8.com/fluency/96/graduation-cap.png", width=64
)
st.sidebar.title("EDXSO Discovery")
st.sidebar.caption("AI-Powered School Ecosystem Influencer Matching")

nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "1. Overview",
        "2. Discover Profiles",
        "3. Ranked Results",
        "4. Profile Details",
        "5. Import Dataset",
        "6. Search History",
        "7. Methodology & Limitations",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Operating Mode")
mode_selection = st.sidebar.selectbox(
    "Active Data Provider",
    [
        OperatingMode.TEST_DATASET.value,
        OperatingMode.IMPORTED_DATA.value,
        OperatingMode.AUTHORIZED_CONNECTOR.value,
    ],
    index=0 if st.session_state.operating_mode == OperatingMode.TEST_DATASET.value else (1 if st.session_state.operating_mode == OperatingMode.IMPORTED_DATA.value else 2),
    help="Default is Test Dataset Mode. All synthetic records are explicitly labeled.",
)
st.session_state.operating_mode = mode_selection

# Active Provider Resolution
current_provider = None
if mode_selection == OperatingMode.TEST_DATASET.value:
    current_provider = synthetic_provider
elif mode_selection == OperatingMode.IMPORTED_DATA.value:
    if st.session_state.custom_csv_provider is not None:
        current_provider = st.session_state.custom_csv_provider
    else:
        # Load sample imported CSV as initial fallback
        current_provider = CSVProfileProvider(
            AppConfig.SAMPLE_CSV_FILE,
            source_description="Bundled sample lawful CSV import",
        )
else:
    current_provider = AuthorizedConnectorProvider()

st.sidebar.info(
    f"**Engine**: `{ranking_engine.similarity_engine.active_engine.split('+')[0].strip()}`\n\n"
    f"**Candidates in Pool**: `{current_provider.get_total_count()}`"
)


# ==============================================================================
# TAB 1: OVERVIEW
# ==============================================================================
if nav_choice == "1. Overview":
    st.markdown('<div class="main-header">EDXSO School Influencer Discovery</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-powered discovery tool to find school education consultants, former principals, directors, and trustees with high professional ecosystem relevance.</div>',
        unsafe_allow_html=True,
    )

    # Compliance Banner
    st.warning(
        "🛡️ **Compliance & Provenance Guardrail**: Unauthorized scraping of LinkedIn is strictly disabled. "
        "The default mode operates on an offline, peer-reviewed **SYNTHETIC TEST BENCHMARK** where records are explicitly marked. "
        "For real-world analysis, import your lawfully collected profile CSV in 'Import Dataset'."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric-card"><h4>Active Mode</h4><h3>{current_provider.get_mode_name().split("(")[0]}</h3><small>Configurable in sidebar</small></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><h4>Candidate Pool</h4><h3>{current_provider.get_total_count()} Profiles</h3><small>Available for matching</small></div>',
            unsafe_allow_html=True,
        )
    with col3:
        last_count = len(st.session_state.last_ranked_results)
        st.markdown(
            f'<div class="metric-card"><h4>Last Discovered</h4><h3>{last_count} Results</h3><small>From recent search</small></div>',
            unsafe_allow_html=True,
        )
    with col4:
        engine_label = "TF-IDF + Cosine" if "TF-IDF" in ranking_engine.similarity_engine.active_engine else "Sentence Transformers"
        st.markdown(
            f'<div class="metric-card"><h4>ML Similarity</h4><h3>{engine_label}</h3><small>Transparent scoring</small></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Target Professional Ecosystem")
    colA, colB = st.columns(2)
    with colA:
        st.markdown(
            """
            - 🏫 **Former Principals & Headmistresses**: Veteran educators with 15-25+ years in CBSE, ICSE, Cambridge, or IB leadership.
            - 🏛️ **School Owners & Managing Trustees**: Board-level leaders governing single-campus and multi-campus school trusts.
            - 🧭 **K-12 Education Consultants & Turnaround Advisors**: Specialists advising school promoters on affiliation, pedagogical audits, and growth.
            """
        )
    with colB:
        st.markdown(
            """
            - 🌐 **School Network & Cluster Directors**: Multi-school leaders overseeing curriculum standardisation and teacher performance across 10-30+ campuses.
            - 💡 **Education Entrepreneurs & Alliance Leads**: Founders and community leaders connecting hundreds of school correspondents.
            - ⚖️ **Control Profiles**: Software engineers, retail managers, and corporate analysts included to verify ranking discrimination.
            """
        )

    st.markdown("---")
    st.markdown("### Provenance & Data Origin Details")
    prov = current_provider.get_provenance_summary()
    st.json(prov)


# ==============================================================================
# TAB 2: DISCOVER PROFILES
# ==============================================================================
elif nav_choice == "2. Discover Profiles":
    st.markdown('<div class="main-header">Discover Similar School Influencers</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Input a reference professional profile to discover similar leaders, consultants, and trustees in the school ecosystem.</div>',
        unsafe_allow_html=True,
    )

    if mode_selection == OperatingMode.AUTHORIZED_CONNECTOR.value and not current_provider.is_available():
        st.error(
            "⚠️ **Authorized Connector Mode requires enterprise OAuth API keys.**\n\n"
            "Per compliance guidelines, unauthorized scraping of LinkedIn is disabled. "
            "Please switch the sidebar to **TEST DATASET MODE** (Default) or **IMPORTED DATA MODE** to proceed."
        )
        st.stop()

    # Instant Preset Buttons
    st.markdown("#### Quick Test Presets (1-Click Evaluation)")
    preset_cols = st.columns(3)

    preset_data = None
    if preset_cols[0].button("📋 Preset 1: Senior K-12 Consultant & Former Principal", use_container_width=True):
        preset_data = {
            "profile_id": "SYN-001",
            "name": "Dr. Sunita Sharma (Reference Profile)",
            "url": "https://example.com/profiles/syn-001",
            "role": "Senior Education Consultant & K-12 School Advisor",
            "org": "Independent Educational Advisory",
            "loc": "New Delhi, India",
            "exp": 24.0,
            "summary": "Former CBSE Senior Principal with 24 years in K-12 school education. Now advising school trusts, founding boards, and directors on school turnaround, teacher development, CBSE/NEP curriculum alignment, and institutional governance across 15+ private schools.",
            "indicators": "Former Principal of 2 flagship CBSE schools; Advisory board member for 4 school trusts; Consultant to 15+ K-12 schools; NEP 2020 pedagogical integration specialist",
            "skills": "School Turnaround; Curriculum Strategy; CBSE Affiliation; Teacher Training; School Governance; K-12 Leadership",
        }
    elif preset_cols[1].button("🏛️ Preset 2: Managing Director & Multi-Campus Trustee", use_container_width=True):
        preset_data = {
            "profile_id": "SYN-002",
            "name": "Rajesh V. Patel (Reference Profile)",
            "url": "https://example.com/profiles/syn-002",
            "role": "Managing Director & Trustee",
            "org": "Vanguard Group of Schools",
            "loc": "Ahmedabad, Gujarat, India",
            "exp": 20.0,
            "summary": "Trustee and Managing Director overseeing 8 premier K-12 campuses with over 12,000 students. Active networker with state education boards, school owner associations, and educational philanthropists. Focus on technology-enabled classrooms and multi-campus standardisation.",
            "indicators": "Managing Trustee overseeing 8 K-12 schools; Executive Committee Member, State Independent Schools Association; Direct relationships with 50+ school owners; Multi-campus infrastructure planner",
            "skills": "School Network Management; Trustee Governance; Multi-school Operations; K-12 Expansion; Budgeting; School Promoters Network",
        }
    elif preset_cols[2].button("💡 Preset 3: Education Entrepreneur & Network Lead", use_container_width=True):
        preset_data = {
            "profile_id": "SYN-003",
            "name": "Meera Krishnan (Reference Profile)",
            "url": "https://example.com/profiles/syn-003",
            "role": "Founder & Chief Academic Officer",
            "org": "Auroville School Mentors Network",
            "loc": "Bengaluru, Karnataka, India",
            "exp": 18.0,
            "summary": "Education entrepreneur and former ICSE School Headmistress. Founded an educational consultancy that trains academic coordinators and designs STEM/experiential learning curricula for 30+ affiliated schools across South India.",
            "indicators": "Former Headmistress at Bishop Cotton Girls School; Founded K-12 academic consultancy servicing 30+ schools; Lead curriculum architect for ICSE and Cambridge primary programs",
            "skills": "Pedagogical Leadership; Multi-school Curriculum; ICSE & Cambridge Alignment; Teacher Mentorship; Education Entrepreneurship",
        }

    st.markdown("---")
    st.markdown("#### Reference Profile Input")

    with st.form("discovery_form"):
        st.caption(
            "Note: Direct URL fetching requires configured enterprise APIs. Manual profile input is provided to maintain full user control without violating platform terms."
        )
        u_col1, u_col2 = st.columns([2, 5])
        with u_col1:
            ref_pid = st.text_input(
                "Reference Profile ID (Optional)",
                value=preset_data.get("profile_id", "SYN-001") if preset_data else "SYN-001",
                help="Stable profile ID (e.g. SYN-001) for deterministic self-exclusion.",
            )
        with u_col2:
            url_input = st.text_input(
                "Reference LinkedIn Profile URL (Optional / Placeholder)",
                value=preset_data["url"] if preset_data else "https://example.com/profiles/syn-001",
                help="Synthetic placeholder or lawful public profile URL.",
            )

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            ref_name = st.text_input(
                "Reference Professional Name",
                value=preset_data["name"] if preset_data else "Dr. Sunita Sharma (Reference Profile)",
            )
            ref_role = st.text_input(
                "Current / Primary Role *",
                value=preset_data["role"] if preset_data else "Senior Education Consultant & K-12 School Advisor",
            )
            ref_org = st.text_input(
                "Organization / Trust",
                value=preset_data["org"] if preset_data else "Independent Educational Advisory",
            )
            ref_loc = st.text_input(
                "Geographic Location",
                value=preset_data["loc"] if preset_data else "New Delhi, India",
            )

        with f_col2:
            ref_exp = st.number_input(
                "Experience in Education (Years)",
                min_value=0.0,
                max_value=60.0,
                value=float(preset_data["exp"]) if preset_data else 24.0,
                step=1.0,
            )
            ref_indicators = st.text_area(
                "School Ecosystem Experience Indicators (semicolon-separated)",
                value=preset_data["indicators"] if preset_data else "Former Principal of 2 flagship CBSE schools; Advisory board member for 4 school trusts; Consultant to 15+ K-12 schools; NEP 2020 specialist",
                height=90,
            )
            ref_skills = st.text_input(
                "Relevant Skills / Competencies (semicolon-separated)",
                value=preset_data["skills"] if preset_data else "School Turnaround; Curriculum Strategy; CBSE Affiliation; Teacher Training; School Governance",
            )

        ref_summary = st.text_area(
            "Professional Summary",
            value=preset_data["summary"] if preset_data else "Former CBSE Senior Principal with 24 years in K-12 school education. Now advising school trusts, founding boards, and directors on school turnaround, teacher development, CBSE/NEP curriculum alignment, and institutional governance across 15+ private schools.",
            height=100,
        )

        st.markdown("#### Candidate Filters & Search Configuration")
        cfg1, cfg2, cfg3, cfg4 = st.columns(4)
        with cfg1:
            filter_role = st.text_input("Filter Role Keyword", placeholder="e.g. Principal, Director")
        with cfg2:
            filter_loc = st.text_input("Filter Location Keyword", placeholder="e.g. Delhi, Mumbai")
        with cfg3:
            min_score = st.slider("Min Relevance Threshold", 0.0, 100.0, 20.0, 5.0)
        with cfg4:
            top_k_select = st.selectbox("Max Results", [10, 20, 25, 30, 50], index=2)

        # Advanced Tunable Weights Expander
        with st.expander("⚙️ Customize ML & Domain Scoring Weights (Must sum to 100%)"):
            w_col1, w_col2, w_col3, w_col4, w_col5 = st.columns(5)
            with w_col1:
                w_sem = st.number_input("Semantic Sim (%)", 0, 100, 40, step=5)
            with w_col2:
                w_eco = st.number_input("School Ecosystem (%)", 0, 100, 25, step=5)
            with w_col3:
                w_lead = st.number_input("Leadership (%)", 0, 100, 15, step=5)
            with w_col4:
                w_multi = st.number_input("Multi-School (%)", 0, 100, 15, step=5)
            with w_col5:
                w_skill = st.number_input("Skills & Exp (%)", 0, 100, 5, step=5)

            total_weight = w_sem + w_eco + w_lead + w_multi + w_skill
            if total_weight != 100:
                st.warning(f"⚠️ Current weights sum to {total_weight}%. They will be automatically normalized to 100%.")

        submitted = st.form_submit_button("🔍 Discover & Rank Candidate Influencers", type="primary", use_container_width=True)

    if submitted:
        if not ref_role.strip():
            st.error("Please provide at least the current/target role.")
        else:
            with st.spinner("Analyzing reference profile and computing ML similarity ranking..."):
                # Construct ReferenceProfile
                ref_profile = ReferenceProfile(
                    profile_id=ref_pid.strip() if ref_pid.strip() else None,
                    name=ref_name,
                    linkedin_url=url_input,
                    current_role=ref_role,
                    organization=ref_org,
                    location=ref_loc,
                    professional_summary=ref_summary,
                    experience_years=ref_exp,
                    school_ecosystem_indicators=ref_indicators,
                    skills=ref_skills,
                )
                st.session_state.last_reference_profile = ref_profile

                # Setup weights
                weights = ScoringWeights(
                    semantic_similarity=w_sem / 100.0,
                    school_ecosystem=w_eco / 100.0,
                    leadership=w_lead / 100.0,
                    multi_school=w_multi / 100.0,
                    skills_experience=w_skill / 100.0,
                ).normalize()

                # Get candidate pool
                filters = {
                    "role_keyword": filter_role,
                    "location_keyword": filter_loc,
                }
                candidates_pool = current_provider.get_profiles(filters=filters)

                # Initialize custom ranking engine with chosen weights
                active_ranker = RankingEngine(similarity_engine=ranking_engine.similarity_engine, weights=weights)
                ranked_results = active_ranker.rank_candidates(
                    reference_profile=ref_profile,
                    candidates=candidates_pool,
                    min_threshold=min_score,
                    top_k=top_k_select,
                )

                # Generate evidence-backed explanations
                ranked_results = ExplanationEngine.attach_explanations(ranked_results)
                st.session_state.last_ranked_results = ranked_results

                # Save session to SQLite
                session_id = db.save_search_session(
                    dataset_mode=current_provider.get_mode_name(),
                    reference_profile=ref_profile,
                    weights=weights,
                    ranking_engine=active_ranker.similarity_engine.active_engine,
                    total_evaluated=len(candidates_pool),
                    ranked_candidates=ranked_results,
                )

                st.success(
                    f"✅ **Discovery Complete!** Retrieved **{len(ranked_results)}** candidates "
                    f"evaluated against reference profile. Search session #{session_id} persisted in SQLite."
                )
                st.info("👉 Switch to the **'3. Ranked Results'** tab to inspect the ranked table and analytics.")


# ==============================================================================
# TAB 3: RANKED RESULTS
# ==============================================================================
elif nav_choice == "3. Ranked Results":
    st.markdown('<div class="main-header">Ranked Discovery Results</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Inspect, sort, and export candidate influencers discovered based on the reference profile.</div>',
        unsafe_allow_html=True,
    )

    results: List[RankedCandidate] = st.session_state.last_ranked_results
    if not results:
        st.info("ℹ️ No search results currently in memory. Please go to **'2. Discover Profiles'** and trigger a search.")
    else:
        st.write(f"Showing **{len(results)}** ranked candidates:")

        # Summary Metrics
        m1, m2, m3, m4 = st.columns(4)
        top_score = results[0].relevance_score if results else 0.0
        avg_score = sum(r.relevance_score for r in results) / len(results) if results else 0.0
        m1.metric("Total Candidates Returned", len(results))
        m2.metric("Top Relevance Score", f"{top_score:.1f}%")
        m3.metric("Average Score", f"{avg_score:.1f}%")
        m4.metric("Active ML Engine", results[0].ranking_method.split("+")[0].strip())

        # CSV Export Button
        csv_str = CSVExporter.to_csv_string(results)
        st.download_button(
            label="📥 Export Results to CSV (with Full Metadata & Scores)",
            data=csv_str,
            file_name="school_influencer_discovery_results.csv",
            mime="text/csv",
            type="primary",
        )

        # Plotly Score Distribution Chart
        scores_data = [
            {"Rank": c.rank, "Name": c.profile.name, "Score": c.relevance_score, "Role": c.profile.current_role}
            for c in results
        ]
        df_scores = pd.DataFrame(scores_data)
        fig_bar = px.bar(
            df_scores,
            x="Name",
            y="Score",
            color="Score",
            color_continuous_scale="Viridis",
            hover_data=["Rank", "Role"],
            title="Candidate Relevance Scores Across Discovered Influencers",
        )
        fig_bar.update_layout(xaxis_tickangle=-45, height=350, margin=dict(l=20, r=20, t=40, b=100))
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        st.markdown("### Discovered Profiles Table")

        # Display each candidate card
        for cand in results:
            p = cand.profile
            badge_class = "badge-synthetic" if "SYNTHETIC" in p.source_type else "badge-imported"

            with st.container():
                st.markdown(
                    f"""
                    <div class="candidate-card">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <span style="font-size: 1.15rem; font-weight: 700; color: #1E293B;">#{cand.rank}. {p.name}</span>
                                &nbsp;&nbsp;<span class="{badge_class}">{p.source_type}</span>
                                <div style="color: #475569; font-weight: 500; margin-top: 4px;">
                                    💼 {p.current_role} &nbsp;|&nbsp; 🏛️ {p.organization} &nbsp;|&nbsp; 📍 {p.location}
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div class="score-pill">{cand.relevance_score:.1f}%</div>
                                <small style="color: #64748B;">Relevance Score</small>
                            </div>
                        </div>
                        <div style="margin-top: 10px; font-size: 0.92rem; color: #334155; background: #F8FAFC; padding: 8px 12px; border-radius: 6px; border-left: 3px solid #0D9488;">
                            <strong>Why matched:</strong> {cand.explanation}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                card_cols = st.columns([6, 1.5, 1.5])
                with card_cols[1]:
                    if st.button(f"🔎 Inspect Details #{cand.rank}", key=f"btn_inspect_{cand.rank}"):
                        st.session_state.selected_candidate_id = p.profile_id
                        st.info(f"Selected **{p.name}**. Switch to **'4. Profile Details'** to view full breakdown.")
                with card_cols[2]:
                    # Placeholder or lawful URL
                    st.link_button("🌐 Profile URL", p.linkedin_url)


# ==============================================================================
# TAB 4: PROFILE DETAILS
# ==============================================================================
elif nav_choice == "4. Profile Details":
    st.markdown('<div class="main-header">Candidate Profile Deep Dive</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Detailed component scoring breakdown, evidence verification, and data provenance.</div>',
        unsafe_allow_html=True,
    )

    results: List[RankedCandidate] = st.session_state.last_ranked_results
    if not results:
        st.info("ℹ️ No ranked results available. Please run discovery in **'2. Discover Profiles'** first.")
    else:
        # Selector for candidate
        id_options = [c.profile.profile_id for c in results]
        labels = [f"#{c.rank}: {c.profile.name} ({c.relevance_score:.1f}%)" for c in results]
        id_to_label = dict(zip(id_options, labels))

        default_idx = 0
        if st.session_state.selected_candidate_id in id_options:
            default_idx = id_options.index(st.session_state.selected_candidate_id)

        selected_id = st.selectbox(
            "Select Candidate to Inspect",
            id_options,
            index=default_idx,
            format_func=lambda x: id_to_label.get(x, x),
        )

        selected_candidate = next(c for c in results if c.profile.profile_id == selected_id)
        p = selected_candidate.profile
        scores = selected_candidate.component_scores

        col_left, col_right = st.columns([5, 5])
        with col_left:
            st.markdown(f"### {p.name}")
            st.markdown(f"**Role:** {p.current_role}")
            st.markdown(f"**Organization:** {p.organization}")
            st.markdown(f"**Location:** {p.location}")
            st.markdown(f"**Professional Experience:** {p.experience_years} years")
            st.markdown(f"**Provenance:** `{p.source_type}`")
            st.markdown(f"**Profile URL:** [{p.linkedin_url}]({p.linkedin_url})")

            st.markdown("#### Professional Summary")
            st.write(p.professional_summary if p.professional_summary else "Not specified")

            st.markdown("#### Explicit School Ecosystem Indicators")
            if p.school_ecosystem_indicators:
                for ind in p.school_ecosystem_indicators:
                    st.markdown(f"- {ind}")
            else:
                st.write("None specified")

            st.markdown("#### Skills & Competencies")
            if p.skills:
                st.write(", ".join(p.skills))
            else:
                st.write("None specified")

        with col_right:
            st.markdown("### Algorithmic Scoring Breakdown")
            st.metric("Overall Weighted Relevance", f"{selected_candidate.relevance_score:.1f}%")

            # Radar Chart for Component Scores
            categories = [
                "Semantic Sim",
                "School Ecosystem",
                "Leadership",
                "Multi-School Reach",
                "Skills & Exp",
            ]
            values = [
                scores.semantic_similarity,
                scores.school_ecosystem,
                scores.leadership,
                scores.multi_school,
                scores.skills_experience,
            ]

            fig_radar = go.Figure()
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name="Score (0-100)",
                    line_color="#0D9488",
                )
            )
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                height=320,
                margin=dict(l=40, r=40, t=20, b=20),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Component Table
            comp_df = pd.DataFrame(
                {
                    "Component": [
                        "1. Semantic Similarity (ML/TF-IDF)",
                        "2. School Ecosystem Relevance",
                        "3. Leadership & Decision-Maker",
                        "4. Multi-School & Advisory Reach",
                        "5. Skills & Experience Depth",
                    ],
                    "Score (/100)": [
                        scores.semantic_similarity,
                        scores.school_ecosystem,
                        scores.leadership,
                        scores.multi_school,
                        scores.skills_experience,
                    ],
                }
            )
            st.dataframe(comp_df, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("### Evidence-Based Match Explanation")
        st.info(f"📝 {selected_candidate.explanation}")

        # Data Uncertainties / Missing Fields
        if selected_candidate.uncertainties_or_missing:
            st.warning("⚠️ **Data Completeness & Uncertainty Disclosures:**\n" + "\n".join(f"- {u}" for u in selected_candidate.uncertainties_or_missing))


# ==============================================================================
# TAB 5: IMPORT DATASET
# ==============================================================================
elif nav_choice == "5. Import Dataset":
    st.markdown('<div class="main-header">Import Lawful Candidate Dataset (CSV)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Upload a lawfully acquired CSV of school influencers, principals, or education consultants for matching.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        **Required CSV Columns:**
        - `profile_id`: Unique record identifier (e.g. `IMP-001`)
        - `name`: Candidate's full name
        - `linkedin_url`: Public profile link or lawful directory URI
        - `current_role`: Candidate's current professional title
        
        **Recommended Optional Columns:**
        `organization`, `location`, `professional_summary`, `experience_years`,
        `school_ecosystem_indicators` (semicolon-delimited), `skills` (semicolon-delimited),
        `source_description`, `source_url`.
        """
    )

    # Sample template download
    sample_df = pd.read_csv(AppConfig.SAMPLE_CSV_FILE)
    csv_sample_str = sample_df.to_csv(index=False)
    st.download_button(
        label="📄 Download Sample CSV Template",
        data=csv_sample_str,
        file_name="sample_school_influencer_import.csv",
        mime="text/csv",
    )

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Profile CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            csv_provider = CSVProfileProvider(
                source=uploaded_file,
                source_description=f"User CSV: {uploaded_file.name}",
            )
            valid_profiles = csv_provider.get_profiles()
            st.session_state.custom_csv_provider = csv_provider
            st.session_state.operating_mode = OperatingMode.IMPORTED_DATA.value

            # Save to SQLite imported_profiles table
            saved_count = db.save_imported_profiles(valid_profiles)

            st.success(
                f"✅ **CSV Successfully Validated & Imported!** Loaded **{len(valid_profiles)}** valid profiles into database. "
                f"Operating mode automatically switched to **IMPORTED DATA MODE**."
            )

            # Display Preview
            preview_rows = [p.model_dump() for p in valid_profiles]
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True)

        except Exception as e:
            st.error(f"❌ **Schema Validation Error**: {e}")


# ==============================================================================
# TAB 6: SEARCH HISTORY
# ==============================================================================
elif nav_choice == "6. Search History":
    st.markdown('<div class="main-header">Persistent Search History</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">All discovery sessions and candidate scores are safely recorded in local SQLite database.</div>',
        unsafe_allow_html=True,
    )

    sessions = db.list_search_sessions(limit=30)
    if not sessions:
        st.info("No prior search sessions recorded in database yet.")
    else:
        st.write(f"Found **{len(sessions)}** previous search sessions:")
        sessions_df = pd.DataFrame(sessions)
        st.dataframe(sessions_df, use_container_width=True)

        st.markdown("#### Inspect Past Session Results")
        selected_session_id = st.selectbox(
            "Select Session ID to View Saved Results",
            [s["id"] for s in sessions],
            format_func=lambda x: f"Session #{x} ({next(s['created_at'] for s in sessions if s['id'] == x)} - {next(s['reference_role'] for s in sessions if s['id'] == x)})",
        )

        if st.button("Load Session Results"):
            session_results = db.get_session_results(selected_session_id)
            if session_results:
                st.write(f"Retrieved **{len(session_results)}** saved candidates for Session #{selected_session_id}:")
                st.dataframe(pd.DataFrame(session_results), use_container_width=True)
            else:
                st.warning("No candidate records found for this session.")


# ==============================================================================
# TAB 7: METHODOLOGY & LIMITATIONS
# ==============================================================================
elif nav_choice == "7. Methodology & Limitations":
    st.markdown('<div class="main-header">Methodology, Scoring Formula & Compliance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Complete transparency on the AI/ML similarity algorithm, domain weighting, and platform limitations.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### 1. Multi-Component Scoring Architecture
        Rather than relying solely on exact keyword matching, the tool evaluates candidates across 5 transparent dimensions:
        """
    )

    st.latex(
        r"\text{Relevance Score} = w_{\text{sem}} \cdot S_{\text{sem}} + w_{\text{eco}} \cdot S_{\text{eco}} + w_{\text{lead}} \cdot S_{\text{lead}} + w_{\text{multi}} \cdot S_{\text{multi}} + w_{\text{skill}} \cdot S_{\text{skill}}"
    )

    st.markdown(
        """
        - **Semantic Similarity ($S_{\text{sem}}$, 40%)**: Enriches reference and candidate profiles into comprehensive contextual text embeddings. Calculated using **Sentence Transformers (`all-MiniLM-L6-v2`)** when available, or **Scikit-Learn Sublinear TF-IDF + Cosine Similarity** as a guaranteed offline fallback.
        - **School Ecosystem Relevance ($S_{\text{eco}}$, 25%)**: Evaluates participation in K-12 education, boards (CBSE, ICSE, Cambridge, IB), and education councils.
        - **Leadership Relevance ($S_{\text{lead}}$, 15%)**: Assesses executive seniority (Principal, Headmistress, Trustee, Director, Dean).
        - **Multi-School / Advisory Reach ($S_{\text{multi}}$, 15%)**: Measures experience working across multiple school campuses, cluster director roles, academic audits, or education consultancy.
        - **Skills & Experience ($S_{\text{skill}}$, 5%)**: Overlap of domain competencies and tenure alignment.

        ---

        ### 2. Ethical Compliance & Data Safeguards
        1. **No Unauthorized Scraping**: This tool strictly respects LinkedIn's Terms of Service and Computer Fraud and Abuse policies. It does NOT bypass CAPTCHA, authentication, or use bot networks.
        2. **Synthetic Benchmark Transparency**: The default test dataset consists of 30 synthetic profiles constructed for testing. They are explicitly marked `[SYNTHETIC TEST DATA]` and never represented as real individuals.
        3. **No Fabricated Predictions**: Relevance scores reflect algorithmic alignment with the provided reference profile. They are **never** presented as real-world connection probabilities, response likelihoods, or lead guarantees.
        4. **Zero-LLM Core**: Ranking and explanations work deterministically without requiring paid or external LLM API keys.
        """
    )
