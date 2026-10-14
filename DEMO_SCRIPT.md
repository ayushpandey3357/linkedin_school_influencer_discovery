# DEMO SCRIPT: EDXSO School Influencer Discovery Tool

**Demonstration Duration**: 5 - 10 Minutes  
**Audience**: Technical Evaluators, Hiring Managers, and EdTech Stakeholders  
**Goal**: Demonstrate an end-to-end workflow discovering education influencers, former principals, and school trustees while showcasing ML ranking transparency, strict compliance, and evidence-grounded explanations.

---

## Preparation Checklist
- [x] Python 3.11+ installed.
- [x] Terminal open in project directory: `c:\Users\ayush\OneDrive\Desktop\linkedin_school_influencer_discovery`.
- [x] Dependencies installed: `pip install -r requirements.txt`.
- [x] Automated tests verified: `python -m pytest -v` (28/28 tests passing).
- [x] Application started: `streamlit run app.py` (or `python run.py`).

---

## Demonstration Timeline Overview

| Minute | Segment | Key Highlights & Screen Actions |
| :---: | :--- | :--- |
| **00:00 - 01:30** | **Problem Statement & Compliance** | Tab 1 (Overview), explain anti-scraping guardrails, synthetic vs imported modes. |
| **01:30 - 04:00** | **Discovery & ML Ranking Workflow** | Tab 2 (Discover Profiles), 1-click Preset 1, custom weighting, run ranking. |
| **04:00 - 06:30** | **Results & Explanation Inspection** | Tab 3 (Ranked Results) & Tab 4 (Profile Details), radar charts, explainability. |
| **06:30 - 08:00** | **CSV Ingestion & Export** | Tab 5 (Import Dataset), upload CSV, Tab 3 CSV export stream. |
| **08:00 - 09:30** | **SQLite Audit Trail & Architecture** | Tab 6 (Search History) & Tab 7 (Methodology & Formula), test suite demonstration. |
| **09:30 - 10:00** | **Q&A and Roadmap Summary** | Recap clean architecture, enterprise API path. |

---

## Detailed Step-by-Step Walkthrough

### Part 1: Introduction & Ethical Compliance Guardrails (1.5 Minutes)
1. **Show Tab 1: Overview**
   - Point out the active header: **EDXSO School Influencer Discovery**.
   - Direct attention to the yellow **Compliance & Provenance Guardrail banner**:
     > *"Notice that this system does not engage in unauthorized scraping of LinkedIn, cookie injection, or CAPTCHA bypass. Instead, it operates strictly on an audited, offline synthetic benchmark of 30 profiles or on lawful user-imported CSV datasets."*
   - Show the metric cards:
     - **Active Mode**: `TEST_DATASET`
     - **Candidate Pool**: `30 Profiles`
     - **ML Similarity Engine**: `TF-IDF + Cosine` (or Sentence Transformers if enabled)
   - Show the **Provenance Summary** JSON on the Overview page demonstrating full metadata auditability.

---

### Part 2: Reference Profile Input & AI/ML Similarity Ranking (2.5 Minutes)
1. **Navigate to Tab 2: Discover Profiles**
   - Click the 1-click quick evaluation preset:  
     👉 **"📋 Preset 1: Senior K-12 Consultant & Former Principal (Dr. Sunita Sharma)"**
   - Point out how the reference profile fields populate automatically:
     - **Role**: `Senior Education Consultant & K-12 School Advisor`
     - **Organization**: `Independent Educational Advisory`
     - **Experience**: `24 Years`
     - **Indicators**: `Former Principal of 2 flagship CBSE schools; Advisory board member for 4 school trusts; Consultant to 15+ K-12 schools`
     - **Skills**: `School Turnaround; Curriculum Strategy; CBSE Affiliation; Teacher Training`
   - Show the **Advanced Tunable Weights Expander**:
     - Explain the 5 transparent scoring components:
       - Semantic Sim (40%)
       - School Ecosystem (25%)
       - Leadership (15%)
       - Multi-School Reach (15%)
       - Skills & Experience (5%)
   - Click the primary action button:  
     👉 **"🔍 Discover & Rank Candidate Influencers"**
   - Show the success message:
     - `✅ Discovery Complete! Retrieved 25 candidates evaluated against reference profile. Search session #1 persisted in SQLite.`

---

### Part 3: Inspecting Ranked Results & Plotly Analytics (2.5 Minutes)
1. **Navigate to Tab 3: Ranked Results**
   - Highlight the summary metrics:
     - **Total Candidates Returned**: `25` (Exceeds the required minimum of 20 results)
     - **Top Relevance Score**: `~82% - 85%`
     - **Average Score**: `~55%`
   - Review the **Plotly Bar Chart**:
     - Visualizes candidate relevance scores descending smoothly.
   - Point out the **Discovered Profiles Table**:
     - Candidate cards display Rank, Name, Role, Organization, Location, and Relevance Score Pill.
     - Note the badge: `[SYNTHETIC TEST DATA]` and `example.com` placeholder URLs.
     - Note that the reference profile itself (`Dr. Sunita Sharma`) is **excluded** from its own recommendations (satisfying requirement 6 & 10).
   - Scroll down to show non-education control profiles (e.g., Software Engineers or Retail Store Managers) properly ranked at the bottom (<25%) with clear discrimination.
   - Click **"📥 Export Results to CSV"** to demonstrate instant CSV streaming with full scores and metadata.

---

### Part 4: Candidate Deep Dive & Evidence-Based Explanations (1.5 Minutes)
1. **Navigate to Tab 4: Profile Details**
   - Select candidate **#1 or #2** (e.g., `Meera Krishnan` or `Rajesh V. Patel`).
   - Show the **Radar Chart**:
     - Displays the candidate's shape across the 5 dimensions: Semantic Sim, School Ecosystem, Leadership, Multi-School Reach, Skills & Experience.
   - Show the **Component Breakdown Table** showing exact marks out of 100 for each dimension.
   - Point to the **Evidence-Based Match Explanation box**:
     > *"Notice that the explanation is not a vague LLM hallucination. It cites concrete profile achievements: 'Profile evidence highlights that candidate serves or served as Managing Director & Trustee at Vanguard Group of Schools, demonstrates Managing Trustee overseeing 8 K-12 schools. Primary algorithmic scoring drivers: School Ecosystem Relevance (92.0/100) and Multi-School / Advisory Reach (95.0/100).'"*
   - Show the **Data Completeness & Uncertainty Disclosures** showing transparency for any missing fields.

---

### Part 5: CSV Import, Persistent History & Automated Tests (2 Minutes)
1. **Navigate to Tab 5: Import Dataset**
   - Click **"📄 Download Sample CSV Template"** (`sample_import.csv`).
   - Drag and drop or browse the bundled `data/sample_import.csv`.
   - Show the green success message:
     - `✅ CSV Successfully Validated & Imported! Loaded 6 valid profiles into database. Operating mode automatically switched to IMPORTED DATA MODE.`
   - Point out that the dataset switches mode immediately and records are tagged `[IMPORTED DATA]`.
2. **Navigate to Tab 6: Search History**
   - Show the table of persisted sessions from SQLite (`data/search_history.db`).
   - Select the recent session to view its saved candidate rankings.
3. **Switch to PowerShell Terminal & Run Pytest**
   ```powershell
   python -m pytest -v
   ```
   - Show that **all 28 automated tests pass in ~3-4 seconds**, verifying:
     - Schema validation
     - Delimiter parsing
     - Exclusion of reference profile
     - Deterministic ranking
     - Minimum 20 results output
     - CSV export roundtrip
     - SQLite session persistence

---

## Feature Implementation Status Matrix

| Component | Status | Details |
| :--- | :---: | :--- |
| **Test Dataset Mode (30 profiles)** | **Implemented** | Bundled in `data/synthetic_profiles.json`, includes diverse K-12 backgrounds & controls. |
| **Imported Data Mode (CSV)** | **Implemented** | Strict schema validation, deduplication, provenance tracking, sample CSV included. |
| **Authorized Connector Mode** | **Implemented (Stub)** | Extensible interface in `app/providers/authorized_connector.py`; reports clear compliance messages. |
| **Profile Analysis Engine** | **Implemented** | Derives leadership, multi-school reach, flags missing attributes, prevents hallucinations. |
| **TF-IDF + Cosine NLP Similarity**| **Implemented** | Sublinear TF-IDF with unigrams & bigrams; works 100% offline without paid APIs. |
| **Sentence Transformers Embeddings**| **Implemented (Auto-detect)** | Automatically upgrades to `all-MiniLM-L6-v2` when installed, falls back to TF-IDF. |
| **Transparent 5-Component Matrix** | **Implemented** | Configurable weights (sum to 100%), normalized scores (0-100), self-exclusion rule. |
| **Evidence-Based Explanations** | **Implemented** | Zero-LLM deterministic generator citing factual fields and component drivers. |
| **Interactive Streamlit UI** | **Implemented** | Multi-tab dashboard with Plotly charts, candidate cards, radar plots, and responsive CSS. |
| **Persistent SQLite Storage** | **Implemented** | Sessions, candidate scores, and imported profiles stored via parameterized SQL. |
| **CSV Export Engine** | **Implemented** | Full data serialization with provenance metadata and component breakdowns. |
| **Automated Pytest Suite** | **Implemented** | 28 automated unit and integration tests; all 28 passing. |
| **Production Cloud Deployment** | **Ready** | Pinned dependencies in `requirements.txt` and `.streamlit/config.toml` for Streamlit Cloud. |
