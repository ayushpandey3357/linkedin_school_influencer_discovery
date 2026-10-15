# MVP TECHNICAL NOTE: Architecture, ML Methods, and Compliance Architecture

**Project**: EDXSO School Influencer Discovery Tool  
**Document Version**: 1.0.0 (MVP Technical Note)  
**Author**: Ayush Kumar Pandey  
**Date**: October 2026  

---

## 1. System Architecture Overview

The system is designed as a modular, decoupled data and ML pipeline conforming to clean architecture principles:

```text
 [Client Layer: Streamlit Local Dashboard / CLI Entrypoint]
                          |
                          v
 [Domain Controller / Business Logic Layer]
   ├── ProfileAnalyzer        (Structural attribute derivation & disclosure detection)
   ├── RankingEngine          (Multi-component score combination & normalization)
   ├── SimilarityEngine       (Sublinear TF-IDF + Cosine Similarity / Sentence-Transformers)
   └── ExplanationEngine      (Deterministic, evidence-grounded rationale synthesis)
                          |
                          v
 [Provider & Data Abstraction Layer]
   ├── ProfileProvider        (Base abstract interface)
   ├── SyntheticProvider      (Local JSON benchmark containing 30 audited records)
   ├── CSVProfileProvider     (Strict schema validation, delimiter cleaning & deduplication)
   └── AuthorizedConnector    (Enterprise API connector stub — non-scraping)
                          |
                          v
 [Persistence & Export Layer]
   ├── SQLite Database        (Parameterized queries, foreign-key relational integrity)
   └── CSVExporter            (Structured tabular serialization with audit metadata)
```

### Key Architectural Tenets
1. **Separation of Concerns**: Machine learning algorithms know nothing about UI components or storage implementations.
2. **Provider Polymorphism**: Adding a new lawful data vendor (e.g., an official LinkedIn Community Management API or an internal school directory database) requires subclassing `ProfileProvider` without changing ranking or UI code.
3. **Deterministic Core**: The scoring and explanation pipelines do not invoke external, nondeterministic generative APIs, ensuring reproducible results for auditability and compliance.
4. **Local Execution**: The application runs as a local Streamlit tool; it is not deployed as a cloud service and does not scrape live websites.

---

## 2. Data Retrieval & Ingestion Architecture

### 2.1 The Compliance Problem with Web Scraping
Direct web scraping of LinkedIn profiles violates Section 8 of LinkedIn's User Agreement, triggers automated bot mitigations (Cloudflare, PerimeterX, CAPTCHAs), risks IP blacklisting, and violates Computer Fraud and Abuse statutes. Furthermore, scraping unstructured HTML yields noisy or fabricated profile attributes. The application **does not automatically discover or scrape real LinkedIn profiles**.

### 2.2 Implemented Provider Architecture

The project cleanly distinguishes three explicit data sources:

#### 1. Synthetic Test Records (`SyntheticDatasetProvider`)
- Serves as the offline, zero-dependency benchmark harness.
- Bundles 30 curated synthetic records (`SYN-001` through `SYN-030`) spanning former principals, school owners, education consultants, network directors, trustees, and non-education control profiles.
- **Provenance Safeguards**:
  - `source_type`: strictly labeled as `"SYNTHETIC TEST DATA"`.
  - Names explicitly suffixed with `(Synthetic Profile)`.
  - URLs restricted to RFC 2606 example domains (e.g., `https://example.com/profiles/syn-001`).

#### 2. Imported CSV Data (`CSVProfileProvider`)
- Ingests lawful, user-supplied CSV files (e.g., opted-in directories or public institutional datasets).
- **Validation Pipeline**:
  - Schema validation via Pydantic v2 (`CandidateProfile`).
  - Required fields: `profile_id`, `name`, `linkedin_url`, `current_role`.
  - Multi-delimiter parsing for nested indicators and skills (handles `;`, `|`, and `,`).
  - Strict deduplication: Deduplicates on `profile_id` and normalized URL. Never merges two profiles based merely on similar names.
  - Error isolation: Invalid rows are captured into a validation error log without crashing the application.
  - `source_type`: labeled as `"IMPORTED CSV DATA"`.

#### 3. Authorized Connector Stub (`AuthorizedConnectorProvider`)
- Establishes a compliant, non-scraping stub interface for enterprise OAuth2 integrations (e.g., official LinkedIn Partner / Talent Solutions APIs).
- If API credentials (`LINKEDIN_CLIENT_ID`, `LINKEDIN_CLIENT_SECRET`) are absent, it gracefully returns an empty candidate list with an informative diagnostic message rather than attempting unauthorized crawling or faking live profile retrieval.

---

## 3. Machine Learning & Similarity Methodology

### 3.1 Profile Document Representation
To perform semantic similarity, structured profile attributes are synthesized into an information-dense textual representation:

$$\mathcal{D}(P) = \text{Concat}\Big(\text{Role}, \text{Organization}, \text{Location}, \text{Summary}, \text{Indicators}, \text{Skills}, \text{Experience}\Big)$$

Example synthesized document:
> `Role: Senior Education Consultant & K-12 School Advisor. Organization: Independent Educational Advisory. Location: New Delhi, India. Summary: Former CBSE Senior Principal with 24 years in K-12 school education... School Ecosystem Experience: Former Principal of 2 flagship CBSE schools... Skills: School Turnaround CBSE Affiliation... Experience: 24.0 years.`

### 3.2 Active Ranking Implementation: Sublinear TF-IDF + Cosine Similarity

The **active, verified ranking method** currently executed by the application is **Scikit-Learn Sublinear TF-IDF Vectorization with Cosine Similarity**:

- **Active Engine**: `TfidfVectorizer` configured with:
  - `ngram_range=(1, 2)`: Captures unigrams (e.g., "principal", "trustee") and informative bigrams (e.g., "school turnaround", "cbse affiliation").
  - `sublinear_tf=True`: Replaces raw term frequency $tf$ with $1 + \log(tf)$ to prevent repetitive tokens from dominating relevance.
  - `stop_words='english'`: Eliminates non-informative lexical noise.
  - `min_df=1`: Retains domain-specific educational terminology even in small corpora.
- **Similarity Calculation**: Cosine similarity is computed between the reference vector $\vec{u}$ and candidate vectors $\vec{v}$:
  $$\cos(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \|\vec{v}\|_2}$$
  The resulting value is scaled to $[0, 100]$:
  $$S_{\text{sem}} = \max\Big(0.0, \min\big(100.0, \cos(\vec{u}, \vec{v}) \times 100.0\big)\Big)$$
- **Execution Profile**: Runs completely offline with zero network calls, zero external API costs, and guaranteed deterministic outputs.

### 3.3 Optional Embedding Enhancement: Sentence Transformers

The codebase includes an optional `sentence-transformers` loader within `SimilarityEngine` (`app/ranking.py`):
- If the optional library is installed in the Python environment, the engine can load `all-MiniLM-L6-v2` (384-dimensional dense embeddings).
- In the project's base `requirements.txt`, `sentence-transformers` is commented out by default to keep the MVP lightweight, dependency-free, and instantly runnable across standard environments.
- When `sentence-transformers` is uninstalled or fails to load, `SimilarityEngine` automatically logs a warning and uses the active TF-IDF + Cosine Similarity engine.

---

## 4. Multi-Component Ranking Logic

Relying solely on textual cosine similarity produces false positives (e.g., a software developer mentioning a school management tool might match the word "school"). The application therefore couples semantic similarity with a **Domain-Specific Multi-Component Scoring Matrix**:

$$\text{Total Score} = \sum_{k \in \mathcal{K}} w_k \cdot S_k, \quad \text{where } \sum_{k \in \mathcal{K}} w_k = 1.0$$

### Verified Scoring Weights & Component Formulas

Verified directly against `app/config.py` and `app/ranking.py`:

| Component ($k$) | Code Field | Default Weight ($w_k$) | Component Range |
| :--- | :--- | :---: | :---: |
| **Semantic Similarity** | `semantic_similarity` | **0.40** (40%) | $[0.0, 100.0]$ |
| **School Ecosystem Relevance** | `school_ecosystem` | **0.25** (25%) | $[0.0, 100.0]$ |
| **Leadership Authority** | `leadership` | **0.15** (15%) | $[0.0, 100.0]$ |
| **Multi-School / Advisory Reach** | `multi_school` | **0.15** (15%) | $[0.0, 100.0]$ |
| **Skills & Experience Depth** | `skills_experience` | **0.05** (5%) | $[0.0, 100.0]$ |

The composite relevance formula implemented in `RankingEngine.rank_candidates` is:

$$\text{Relevance Score} = 0.40 \cdot S_{\text{sem}} + 0.25 \cdot S_{\text{eco}} + 0.15 \cdot S_{\text{lead}} + 0.15 \cdot S_{\text{multi}} + 0.05 \cdot S_{\text{skill}}$$

#### Component Calculation Details:
1. **Semantic Similarity ($S_{\text{sem}}$)**: Scaled cosine similarity of TF-IDF vectors between the reference profile text and candidate profile text.
2. **School Ecosystem Relevance ($S_{\text{eco}}$)**:
   - Matches against 23 curated ecosystem tokens (`school`, `k-12`, `cbse`, `icse`, `cambridge`, `ib`, `pedagogy`, `turnaround`, etc.): $\min(\text{tokens} \times 8.0, 50.0)$.
   - Structured indicator bonus: $\min(N_{\text{indicators}} \times 10.0, 30.0)$.
   - K-12 sector classification bonus: $+20.0$.
   - Capped at $100.0$.
3. **Leadership & Decision-Making Authority ($S_{\text{lead}}$)**:
   - Role seniority classification: $+45.0$ for owner/trustee/founder, or $+35.0$ for principal/head/director.
   - Leadership token density: $\min(N_{\text{lead\_tokens}} \times 12.0, 30.0)$.
   - Experience depth: $+25.0$ for $\ge 15$ years, or $+15.0$ for $\ge 8$ years.
   - Capped at $100.0$.
4. **Multi-School / Advisory Reach ($S_{\text{multi}}$)**:
   - Profile multi-school experience signal: $+40.0$.
   - Consultant / advisor classification: $+30.0$.
   - Regex detection of institutional scale (e.g., "15+ schools", "8 campuses"): $+30.0$ for $\ge 20$ schools, $+20.0$ for $\ge 5$ schools, $+10.0$ for $< 5$ schools.
   - Multi-school keyword matches: $\min(N_{\text{multi\_tokens}} \times 8.0, 25.0)$.
   - Capped at $100.0$.
5. **Skills & Experience Alignment ($S_{\text{skill}}$)**:
   - Overlap ratio of normalized skills: $\frac{|\mathcal{S}_{\text{ref}} \cap \mathcal{S}_{\text{cand}}|}{\max(|\mathcal{S}_{\text{ref}}|, 1)} \times 70.0$.
   - Tenure compatibility penalty: $\max(30.0 - (|\Delta \text{years}| \times 1.5), 0.0)$.
   - Fallback if reference has no skills: $\min(\text{Candidate Experience} \times 4.0, 100.0)$.
   - Capped at $100.0$.

### Score Normalization & Boundary Protections
- All components $S_k$ are clipped to $[0.0, 100.0]$.
- Configured weights are validated to sum to 1.0 ($\pm 10^{-4}$); if adjusted via UI, weights are normalized: $w'_k = \frac{w_k}{\sum_j w_j}$.
- Candidate sorting is deterministic: primary sort by `relevance_score` descending, secondary sort by `profile_id` ascending.

---

### 4.2 Reference Profile Exclusion Architecture & Root Cause Analysis

#### Root Cause Analysis of Previous 93.6% Self-Recommendation Bug
In earlier revisions, searching for "Dr. Sunita Sharma (Reference Profile)" returned "Dr. Sunita Sharma (Synthetic Profile)" ranked #1 with a 93.6% relevance score. Investigation revealed three contributing factors:
1. **Divergent URL Paths**: The reference preset used placeholder `https://example.com/profiles/reference-profile` while candidate SYN-001 used canonical `https://example.com/profiles/syn-001`. Naive equality `ref_url == cand_url` failed.
2. **Parenthetical Suffix Divergence**: The reference profile name included `(Reference Profile)` or `(Reference Test)`, whereas the candidate dataset record had `(Synthetic Profile)`. Exact string equality `ref_name == cand_name` failed.
3. **Missing Model Identifier**: The `ReferenceProfile` Pydantic model omitted an optional `profile_id` attribute, preventing direct ID-based identity tracking.

Because SYN-001 possessed near-identical text to the reference query, TF-IDF cosine similarity reached ~85% and domain heuristic components reached 90-95%, yielding a ~93.6% composite score and ranking #1.

#### Multi-Tier Exclusion Hierarchy
The verified `is_reference_profile_match` function in `app/ranking.py` implements a strict, multi-tiered exclusion hierarchy:

1. **Tier 1 — Stable Profile ID Match**:
   If `reference.profile_id` is populated, candidate matching this ID (case-insensitive) is immediately excluded.
2. **Tier 2 — Normalized Profile URL Match**:
   URLs are stripped of protocol (`http://`, `https://`), `www.`, tracking parameters (`?trk=...`), fragments (`#...`), and trailing slashes. Non-generic paths matching each other trigger exclusion. Generic placeholders (`example.com`, `linkedin.com/in`, etc.) are ignored.
3. **Tier 3 — Conservative Multi-Field Concordance**:
   - Base names are cleaned by removing honorifics (`Dr.`, `Mr.`, `Ms.`) and bracketed suffixes (`(Reference Profile)`, `(Synthetic Profile)`).
   - **Crucial Rule**: Unrelated candidates are **NEVER** excluded on name similarity alone.
   - Exclusion requires clean base name match **plus** concordant verification across multiple fields:
     - Role concordance (identical normalized title or high word overlap $\ge 0.50$)
     - Organization concordance (both specified and matching/substring overlap)
     - Location concordance (shared geographic location)
     - Experience tenure concordance ($|\Delta \text{exp}| \le 1.0\text{ year}$)
     - Summary lexical overlap ($\ge 0.50$ overlap on tokens $\ge 4$ characters)
   - A candidate is excluded if (Role + Organization match) OR if $\ge 3$ independent fields match among {role, organization, location, experience, summary_overlap}.

#### Documented Limitations of Conservative Duplicate Check
- **Homonym Collision with Shared Attributes**: If two genuinely distinct educators share the exact same name, work at the same institution, and hold the exact same role and tenure, the conservative check will treat them as duplicate records.
- **Unlinked Modified Profiles**: If an educator significantly updates their listed role and organization across different sources and provides neither stable profile ID nor profile URL, conservative matching will deliberately not exclude them to prioritize recall and avoid false-positive candidate suppression.

---

## 5. Evidence-Based Explanations

Rather than relying on black-box LLM prompts that risk fabricating achievements or personal connections, the `ExplanationEngine` synthesizes deterministic, evidence-grounded rationales:

### Generation Pipeline
1. **Score Driver Decomposition**: Identifies the top 2 mathematical contributors to the score (e.g., "Semantic Similarity (86.4/100)" and "School Ecosystem Relevance (92.0/100)").
2. **Profile Fact Extraction**: Pulls actual recorded strings from `current_role`, `organization`, and `school_ecosystem_indicators`.
3. **Data Completeness Audit**: Surfaces missing fields (e.g., "Organization not specified", "Tenure zero").
4. **Synthesis**: Produces a structured narrative:
   > `[SYNTHETIC TEST DATA] Exceptional match with a score of 82.8/100. Profile evidence highlights that candidate serves or served as Senior Education Consultant & K-12 School Advisor at Independent Educational Advisory, demonstrates 'Former Principal of 2 flagship CBSE schools (1,800+ students)'. Primary algorithmic scoring drivers: School Ecosystem Relevance (92.0/100) and Semantic Similarity (82.1/100).`

---

## 6. Data Provenance & Storage Architecture

### Relational SQLite Schema
The local SQLite database (`data/search_history.db` or configured path) implements three relational tables verified against `app/database.py`:

```sql
-- 1. Search Sessions Audit Log
CREATE TABLE search_sessions (
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

-- 2. Ranked Candidates Cache
CREATE TABLE search_results (
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

-- 3. Imported Profiles Persistent Store
CREATE TABLE imported_profiles (
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
```

Performance is reinforced via indices: `idx_search_results_session`, `idx_search_results_profile`, `idx_search_sessions_date`, and `idx_imported_profiles_url`. All queries use parameterized statements (`?`) to prevent SQL injection vulnerabilities.

---

## 7. Verification & Automated Testing

The implementation is verified through an automated test suite executed via `pytest`.

### Verified Test Run Result
```text
============================= test session starts =============================
platform win32 -- Python 3.14.3, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\ayush\OneDrive\Desktop\linkedin_school_influencer_discovery
plugins: anyio-4.14.2
collected 36 items

tests\test_analyzer.py ...                                               [  8%]
tests\test_database.py ....                                              [ 19%]
tests\test_explanations.py ...                                           [ 27%]
tests\test_exporter.py ..                                                [ 33%]
tests\test_providers.py ........                                         [ 55%]
tests\test_ranking.py ...........                                        [ 86%]
tests\test_schema.py .....                                               [100%]

============================= 36 passed in 1.89 seconds ==============================
```

### Covered Test Dimensions
- **Profile Analyzer**: Structural attribute derivation, seniority classification, missing field audits, and non-education control checks.
- **Database Persistence**: Session logging, multi-candidate result cascades, CSV profile persistence, and transaction isolation.
- **Explanation Synthesis**: Evidence grounding, driver decomposition, and lack-of-data disclosures.
- **Data Providers**: Synthetic benchmark parsing, Pydantic schema validation, invalid CSV row isolation, and authorized connector stub diagnostics.
- **Ranking & Similarity**: TF-IDF cosine similarity, dynamic weight normalization, boundary constraints, and all 3 tiers of reference-profile exclusion.

---

## 8. Project Setup & Local Execution

Setup commands verified against the actual repository files:

### Step 1: Environment Setup
```powershell
# Navigate to project directory
cd c:\Users\ayush\OneDrive\Desktop\linkedin_school_influencer_discovery

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install core dependencies
pip install -r requirements.txt
```

### Step 2: Run Application
You can launch the dashboard using either entry point:
```powershell
# Option A: Standard launcher script
python run.py

# Option B: Direct Streamlit command
streamlit run app.py
```
The interface opens locally at `http://localhost:8501`.

### Step 3: Run Tests
```powershell
pytest
```

---

## 9. Current Scope & Path to Production

| Dimension | Current MVP Implementation | Future Production Roadmap |
| :--- | :--- | :--- |
| **Data Ingestion** | Curated synthetic benchmark (30 records) & validated CSV import | Official LinkedIn Enterprise / Partner API with OAuth2 |
| **NLP Vectorization** | Sublinear TF-IDF + Cosine Similarity (100% offline, verified active) | Fine-tuned domain Sentence Transformer (`all-MiniLM-L6-v2` / `bge-small`) |
| **Profile Discovery** | Ingested & benchmarked profiles; no live web scraping | Compliant API webhooks with opt-in user consent |
| **Execution Environment** | Local desktop execution via Streamlit | Containerized cloud deployment with role-based access control |
| **Persistence** | Local SQLite (`data/search_history.db`) | Managed PostgreSQL / Cloud SQL with automated backups |
| **Scale** | In-memory evaluation (<10,000 profiles) | Vector database indexing (e.g., Qdrant / Milvus) |
| **Authentication** | Local desktop execution / Streamlit session state | Enterprise SSO (Okta / Google OAuth2) with Role-Based Access Control |
