# EDXSO School Influencer Discovery Tool

> **AI-Powered Discovery of School Education Consultants, Former Principals, Trustees, and K-12 Ecosystem Leaders**  
> *EDXSO AI Intern Assignment 4 - Production-Grade MVP*

---

## 1. Problem Statement

Identifying key influencers and institutional decision-makers within the school education ecosystem is a vital challenge for EdTech initiatives, school trusts, and curriculum providers. Standard LinkedIn search often yields either surface-level keyword hits or fails to identify professionals whose influence spans multiple schools (e.g., former principals turned advisors, multi-campus trustees, education entrepreneurs, and accreditation specialists).

Furthermore, commercial lead tools frequently violate platform terms through unauthorized web scraping or automated credential abuse.

The **EDXSO School Influencer Discovery Tool** addresses this problem:
- **Reference Profile Paradigm**: Accepts a single reference profile (role, organization, experience indicators, summary) and discovers other professionals with similar domain context, governance authority, and school network reach.
- **Contextual Understanding over Keyword Matching**: Evaluates candidates across 5 transparent dimensions, including semantic NLP similarity, leadership seniority, school ecosystem depth, and multi-school reach.
- **Strict Compliance & Data Provenance**: Fully offline by default, rejects unauthorized scraping, operates with an audited benchmark of synthetic education professionals (`[SYNTHETIC TEST DATA]`), and supports lawful CSV ingestion (`[IMPORTED DATA]`).

---

## 2. Core Features

- 🎯 **Multi-Mode Discovery Architecture**:
  - **Test Dataset Mode (Default)**: Completely offline benchmark of 30 synthetic school education professionals and non-education control profiles.
  - **Imported Data Mode**: High-speed CSV ingestion with schema validation, deduplication, and provenance tracking.
  - **Authorized Connector Mode**: Extensible provider stub ready for official enterprise OAuth APIs.
- 🧠 **Dual-Engine Semantic NLP Ranking**:
  - Primary: **Sentence Transformers (`all-MiniLM-L6-v2`)** for dense contextual embeddings.
  - Offline Fallback: **Scikit-Learn Sublinear TF-IDF + Cosine Similarity** (100% offline, zero paid API dependencies).
- ⚖️ **Transparent 5-Component Scoring Model**:
  - Semantic Similarity (40%)
  - School Ecosystem Relevance (25%)
  - Leadership & Authority (15%)
  - Multi-School & Advisory Reach (15%)
  - Skills & Experience Depth (5%)
  - *Weights are fully tunable in real-time and automatically normalized to 100%.*
- 📝 **Evidence-Based Match Explanations**:
  - Algorithmic, deterministic explanation generation citing factual profile indicators, component drivers, and data limitations without relying on external LLM hallucinations.
- 📊 **Interactive Streamlit Dashboard**:
  - Overview metrics, 1-click test presets, candidate cards, Plotly score distributions, and radar chart breakdowns.
- 💾 **Persistent SQLite Audit Trail**:
  - All discovery runs, reference profiles, candidate scores, and imported data are persisted using parameterized queries and relational schemas.
- 📥 **Comprehensive CSV Export**:
  - Instant export of ranked candidate lists with ranks, total scores, component breakdowns, profile URLs, explanations, and audit metadata.

---

## 3. Technology Stack

- **Language**: Python 3.11+ (Validated on Python 3.14.3)
- **Web Interface**: [Streamlit](https://streamlit.io/) (Polished responsive UI with custom CSS design tokens)
- **Data Engineering**: [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)
- **Machine Learning & NLP**: [scikit-learn](https://scikit-learn.org/) (`TfidfVectorizer`, `cosine_similarity`) & optional [Sentence Transformers](https://sbert.net/)
- **Data Modeling & Validation**: [Pydantic v2](https://docs.pydantic.dev/)
- **Storage**: [SQLite3](https://www.sqlite.org/) (Local relational database with foreign key integrity)
- **Visualizations**: [Plotly Express & Graph Objects](https://plotly.com/python/)
- **Testing**: [pytest](https://docs.pytest.org/) (28 comprehensive unit & integration tests)

---

## 4. Architecture

```text
               +--------------------------------------------------+
               |            Reference Profile Input               |
               |  (Manual Form / URL / 1-Click Evaluation Preset) |
               +------------------------+-------------------------+
                                        |
                                        v
               +--------------------------------------------------+
               |             Profile Analyzer Module              |
               | (Derives Leadership, Multi-School, Disclosures)  |
               +------------------------+-------------------------+
                                        |
                                        v
    +-----------------------------------------------------------------------+
    |                       Discovery Provider Layer                        |
    |  +--------------------+  +-------------------+  +-------------------+ |
    |  |  SyntheticDataset  |  |  CSVProfile       |  |  Authorized       | |
    |  |  Provider (JSON)   |  |  Provider (CSV)   |  |  Connector (Stub) | |
    |  +--------------------+  +-------------------+  +-------------------+ |
    +-----------------------------------+-----------------------------------+
                                        |
                                        v
    +-----------------------------------------------------------------------+
    |                     AI/ML Similarity & Ranking                        |
    |  +--------------------------------+  +------------------------------+ |
    |  | Dense Embeddings (SentenceTx)  |  | Sublinear TF-IDF + Cosine    | |
    |  +--------------------------------+  +------------------------------+ |
    |                  Weighted 5-Component Scoring Matrix                  |
    +-----------------------------------+-----------------------------------+
                                        |
                                        v
               +--------------------------------------------------+
               |            Evidence-Based Explanations           |
               |  (Factual citations, component driver analysis) |
               +------------------------+-------------------------+
                                        |
                    +-------------------+-------------------+
                    |                                       |
                    v                                       v
+---------------------------------------+ +----------------------------------+
|           Streamlit UI                | |         SQLite Database          |
| - Filter & Sort                       | | - search_sessions                |
| - Interactive Radar / Bar Charts      | | - ranked_candidates              |
| - CSV Download Stream                 | | - imported_profiles              |
+---------------------------------------+ +----------------------------------+
```

---

## 5. Setup Instructions (Windows PowerShell)

### Prerequisites
- Python 3.11 or higher installed on your system.
- Git (optional, for repository management).

### Step 1: Open PowerShell and Navigate to Workspace
```powershell
cd c:\Users\ayush\OneDrive\Desktop\linkedin_school_influencer_discovery
```

### Step 2: Create and Activate Virtual Environment (Recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

*(Note: Core dependencies include streamlit, pandas, numpy, scikit-learn, plotly, pydantic, and pytest. No paid API keys are required).*

### Step 4: Environment Configuration (Optional)
```powershell
Copy-Item .env.example .env
```

---

## 6. How to Run the Application

You can launch the application using either of the following commands:

### Option A: Using the Launcher Script
```powershell
python run.py
```

### Option B: Direct Streamlit Execution
```powershell
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## 7. How to Run Tests

Execute the automated test suite with full verbose output:

```powershell
python -m pytest -v
```

### Verified Test Results
```text
============================= test session starts =============================
platform win32 -- Python 3.14.3, pytest-9.1.1, pluggy-1.6.0
collected 28 items

tests/test_analyzer.py::test_analyzer_derives_leadership_and_multi_school PASSED [  3%]
tests/test_analyzer.py::test_analyzer_flags_missing_fields PASSED        [  7%]
tests/test_analyzer.py::test_analyzer_control_profile_non_education PASSED [ 10%]
tests/test_database.py::test_database_init_and_search_session_persistence PASSED [ 14%]
tests/test_database.py::test_database_imported_profiles_persistence PASSED [ 17%]
tests/test_explanations.py::test_explanation_generation_high_match PASSED [ 21%]
tests/test_explanations.py::test_explanation_generation_low_match PASSED [ 25%]
tests/test_exporter.py::test_exporter_to_dataframe_and_csv_string PASSED [ 28%]
tests/test_providers.py::test_synthetic_provider_loads_benchmark_dataset PASSED [ 32%]
tests/test_providers.py::test_synthetic_provider_deduplication PASSED    [ 35%]
tests/test_providers.py::test_synthetic_provider_filtering PASSED        [ 39%]
tests/test_providers.py::test_csv_provider_loads_valid_csv PASSED        [ 42%]
tests/test_providers.py::test_csv_provider_missing_required_column_raises_error PASSED [ 46%]
tests/test_providers.py::test_csv_provider_deduplication PASSED          [ 50%]
tests/test_providers.py::test_csv_provider_loads_sample_csv_file PASSED  [ 53%]
tests/test_providers.py::test_authorized_connector_stub_behavior PASSED  [ 57%]
tests/test_ranking.py::test_similarity_engine_tfidf_fallback PASSED      [ 60%]
tests/test_ranking.py::test_weights_normalization PASSED                 [ 64%]
tests/test_ranking.py::test_reference_profile_exclusion PASSED           [ 67%]
tests/test_ranking.py::test_minimum_20_results_for_test_reference PASSED [ 71%]
tests/test_ranking.py::test_score_normalization_and_ranges PASSED        [ 75%]
tests/test_ranking.py::test_deterministic_ranking PASSED                 [ 78%]
tests/test_ranking.py::test_ranking_discrimination_against_non_education_controls PASSED [ 82%]
tests/test_schema.py::test_valid_candidate_profile PASSED                [ 85%]
tests/test_schema.py::test_missing_required_fields_raises_validation_error PASSED [ 89%]
tests/test_schema.py::test_missing_optional_fields_normalization PASSED  [ 92%]
tests/test_schema.py::test_delimiter_parsing_for_indicators_and_skills PASSED [ 96%]
tests/test_schema.py::test_experience_years_parsing PASSED               [100%]

============================= 28 passed in 23.82s =============================
```

---

## 8. Data Sources, Ethics & Compliance Rules

1. **No Unauthorized Scraping**:
   - The application does NOT scrape LinkedIn web pages, bypass CAPTCHAs, hijack cookies, or violate platform rate limits.
2. **Explicit Labeling of Synthetic Data**:
   - In **Test Dataset Mode**, every profile is explicitly tagged with `source_type: "SYNTHETIC TEST DATA"` and includes `(Synthetic Profile)` in its name.
   - All profile URLs use reserved placeholder domains (e.g., `https://example.com/profiles/syn-001`).
3. **Lawful Import Path**:
   - In **Imported Data Mode**, users can import records obtained through lawful directories or consent-based records. Imported records are tagged `[IMPORTED DATA]`.
4. **No Inferred Private Relationships**:
   - The application assesses textual and contextual domain overlap. It never claims that a candidate has an active friendship, connection, or guaranteed lead response rate.

---

## 9. ML Similarity Method & Ranking Logic

### 1. Unified Text Representation
Each profile is mapped to an enriched textual document:
```text
Role: Senior Education Consultant & K-12 School Advisor.
Organization: Independent Educational Advisory.
Location: New Delhi, India.
Summary: Former CBSE Senior Principal with 24 years in K-12 school education...
School Ecosystem Experience: Former Principal of 2 flagship CBSE schools; Advisory board member for 4 school trusts...
Skills: School Turnaround; Curriculum Strategy; CBSE Affiliation...
Experience: 24.0 years.
```

### 2. Semantic Similarity Score ($S_{\text{sem}}$)
- Vectorized using sublinear term-frequency TF-IDF with unigrams and bigrams (`ngram_range=(1, 2)`).
- Cosine similarity between reference vector $\vec{r}$ and candidate vector $\vec{c}_i$ is computed and clipped:
  $$S_{\text{sem}} = \max(0, \min(100, \cos(\vec{r}, \vec{c}_i) \times 100))$$

### 3. Multi-Component Weighted Ranking Formula
$$R = w_{\text{sem}} \cdot S_{\text{sem}} + w_{\text{eco}} \cdot S_{\text{eco}} + w_{\text{lead}} \cdot S_{\text{lead}} + w_{\text{multi}} \cdot S_{\text{multi}} + w_{\text{skill}} \cdot S_{\text{skill}}$$

| Component | Default Weight | Description |
| :--- | :---: | :--- |
| **Semantic Similarity** ($S_{\text{sem}}$) | **40%** | Contextual NLP similarity across roles, summaries, and achievements |
| **School Ecosystem** ($S_{\text{eco}}$) | **25%** | Presence of K-12 keywords, boards (CBSE, ICSE, IB, Cambridge), school trusts |
| **Leadership Relevance** ($S_{\text{lead}}$) | **15%** | Executive decision-maker indicators (Principal, Trustee, Director, Founder) |
| **Multi-School Reach** ($S_{\text{multi}}$) | **15%** | Evidence of advising multiple schools, managing school networks or clusters |
| **Skills & Experience** ($S_{\text{skill}}$) | **5%** | Jaccard overlap of domain competencies and tenure alignment |

*Reference Profile Exclusion*: If a candidate matches the reference profile (by ID or normalized URL), it is automatically excluded from recommendations.

---

## 10. How to Import a CSV

1. Open the **5. Import Dataset** tab in the application.
2. Download the provided template: `sample_school_influencer_import.csv`.
3. Ensure your CSV contains the mandatory columns:
   - `profile_id` (e.g., `IMP-001`)
   - `name` (e.g., `Dr. Arvind Saxena`)
   - `linkedin_url` (e.g., `https://example.com/imported/arvind-saxena`)
   - `current_role` (e.g., `Former Principal & School Accreditation Specialist`)
4. Optional columns: `organization`, `location`, `professional_summary`, `experience_years`, `school_ecosystem_indicators` (semicolon-delimited), `skills` (semicolon-delimited), `source_description`, `source_url`.
5. Upload your CSV. The system validates the schema, deduplicates records, stores metadata in SQLite, and switches operating mode to **IMPORTED DATA MODE**.

---

## 11. Deployment Guide

### Deploying to Streamlit Community Cloud
1. Push this repository to GitHub:
   ```bash
   git init
   git add .
   git commit -m "feat: complete EDXSO school influencer discovery MVP"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
2. Visit [share.streamlit.io](https://share.streamlit.io) and log in.
3. Click **New app**, select your repository, branch (`main`), and set the main file path to:
   ```text
   app.py
   ```
4. Click **Deploy**. The application will read `requirements.txt` and `.streamlit/config.toml` automatically. No secrets or external API keys are required for the MVP.

---

## 12. Project Structure

```text
linkedin_school_influencer_discovery/
│
├── .env.example                     # Environment configuration template
├── .gitignore                       # Ignored patterns (databases, secrets, virtualenvs)
├── .streamlit/
│   └── config.toml                  # Streamlit UI theme and server configuration
├── requirements.txt                 # Pinned dependencies
├── run.py                           # CLI entrypoint to launch Streamlit
├── app.py                           # Complete Streamlit multi-tab dashboard
├── README.md                        # Project documentation and guide
├── TECHNICAL_NOTE.md                # In-depth architectural & algorithmic design note
├── DEMO_SCRIPT.md                   # 5-10 minute presentation & demonstration guide
│
├── app/
│   ├── __init__.py
│   ├── config.py                    # Constants, enums, AppConfig, and ScoringWeights
│   ├── database.py                  # SQLite storage layer with parameterized queries
│   ├── explanations.py              # Evidence-based explanation engine
│   ├── exporter.py                  # CSV export and Pandas serialization
│   ├── models.py                    # Pydantic v2 schemas and data validation
│   ├── profile_analyzer.py          # Modular profile analyzer
│   ├── ranking.py                   # NLP SimilarityEngine and RankingEngine
│   └── providers/
│       ├── __init__.py
│       ├── base.py                  # ProfileProvider abstract interface
│       ├── synthetic_provider.py    # Offline synthetic benchmark provider
│       ├── csv_provider.py          # Validated CSV import provider
│       └── authorized_connector.py  # Provider stub for official enterprise APIs
│
├── data/
│   ├── synthetic_profiles.json      # Benchmark of 30 diverse synthetic profiles
│   └── sample_import.csv            # Lawfully structured sample CSV for import testing
│
└── tests/
    ├── __init__.py
    ├── test_analyzer.py             # Tests for attribute extraction and missing fields
    ├── test_database.py             # Tests for SQLite session and profile storage
    ├── test_explanations.py         # Tests for explanation generator logic
    ├── test_exporter.py             # Tests for CSV export formatting and roundtripping
    ├── test_providers.py            # Tests for Synthetic, CSV, and Connector providers
    ├── test_ranking.py              # Tests for TF-IDF, weights, exclusion, top-20
    └── test_schema.py               # Tests for Pydantic models, validation, and parsing
```

---

## 13. Limitations & Future Roadmap

- **No Live Scraping**: By design, live LinkedIn profiles are not scraped directly via HTTP/Puppeteer due to compliance. Direct URL fetching is reserved for future official LinkedIn Partner APIs.
- **Lexical/Semantic Hybrid Optimization**: While TF-IDF provides robust, deterministic offline performance without GPU/network needs, Sentence Transformers can be enabled in production environments for subtle semantic cross-lingual matching.
- **Future Graph Integration**: Roadmapped integration with graph databases (e.g., Neo4j or BigQuery Property Graphs) for second-degree institutional alumni mapping once authorized data connectors are connected.
