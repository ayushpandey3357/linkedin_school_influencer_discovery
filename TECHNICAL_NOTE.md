# TECHNICAL NOTE: Architecture, ML Methods, and Compliance Architecture

**Project**: EDXSO School Influencer Discovery Tool  
**Document Version**: 1.0.0 (Production-Grade Architecture)  
**Author**: Senior AI & Data Engineering Consultant  
**Date**: October 2026  

---

## 1. System Architecture Overview

The system is designed as a modular, decoupled data and ML pipeline conforming to clean architecture principles:

```text
 [Client Layer: Streamlit Dashboard / CLI Entrypoint]
                          |
                          v
 [Domain Controller / Business Logic Layer]
   ├── ProfileAnalyzer        (Structural attribute derivation & disclosure detection)
   ├── RankingEngine          (Multi-component score combination & normalization)
   ├── SimilarityEngine       (Dense vector or Sublinear TF-IDF NLP representations)
   └── ExplanationEngine      (Deterministic, evidence-grounded rationale synthesis)
                          |
                          v
 [Provider & Data Abstraction Layer]
   ├── ProfileProvider        (Base abstract interface)
   ├── SyntheticProvider      (Local JSON benchmark containing 30 audited records)
   ├── CSVProfileProvider     (Strict schema validation, delimiter cleaning & deduplication)
   └── AuthorizedConnector    (Compliant OAuth2 / Enterprise API stub)
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

---

## 2. Data Retrieval & Ingestion Architecture

### 2.1 The Compliance Problem with Web Scraping
Direct web scraping of LinkedIn profiles violates Section 8 of LinkedIn's User Agreement, triggers bot mitigations (Cloudflare, PerimeterX, CAPTCHAs), risks IP blacklisting, and violates Computer Fraud and Abuse statutes. Furthermore, scraping unstructured HTML yields noisy or fabricated profile attributes.

### 2.2 Implemented Provider Architecture

#### 1. Synthetic Dataset Provider (`SyntheticDatasetProvider`)
- Serves as the zero-dependency test harness.
- Bundles 30 meticulously constructed profiles spanning former principals, school owners, education consultants, network directors, trustees, and non-education controls.
- **Provenance Safeguards**:
  - `source_type`: strictly set to `"SYNTHETIC TEST DATA"`.
  - Names explicitly suffixed with `(Synthetic Profile)`.
  - URLs restricted to RFC 2606 example domains (e.g., `https://example.com/profiles/syn-001`).

#### 2. CSV Profile Provider (`CSVProfileProvider`)
- Ingests lawful, consent-based or public directory data.
- **Validation Pipeline**:
  - Schema validation via Pydantic v2.
  - Required fields: `profile_id`, `name`, `linkedin_url`, `current_role`.
  - Multi-delimiter parsing for nested indicators and skills (handles `;`, `|`, and `,`).
  - Strict deduplication: Deduplicates on `profile_id` and normalized URL. Never merges two profiles based merely on similar names.
  - Error isolation: Invalid rows are captured into a validation error log without crashing the application.

#### 3. Authorized Connector Provider (`AuthorizedConnectorProvider`)
- Establishes the interface for enterprise OAuth2 integrations (e.g., LinkedIn API / Talent Solutions).
- If API credentials are not provisioned, gracefully rejects execution with an informative message rather than faking an integration.

---

## 3. Machine Learning & Similarity Methodology

### 3.1 Profile Document Representation
To perform semantic similarity, structured profile attributes are synthesized into an information-dense representation:

$$\mathcal{D}(P) = \text{Concat}\Big(\text{Role}, \text{Organization}, \text{Location}, \text{Summary}, \text{Indicators}, \text{Skills}, \text{Experience}\Big)$$

Example document:
> `Role: Senior Education Consultant & K-12 School Advisor. Organization: Independent Educational Advisory. Location: New Delhi, India. Summary: Former CBSE Senior Principal with 24 years in K-12 school education... School Ecosystem Experience: Former Principal of 2 flagship CBSE schools... Skills: School Turnaround CBSE Affiliation... Experience: 24.0 years.`

### 3.2 NLP Similarity Engine
The system employs a dual-engine architecture:

#### Primary Engine: Dense Semantic Embeddings (Sentence Transformers)
- Model: `all-MiniLM-L6-v2` (384-dimensional dense vectors).
- Cosine similarity:
  $$\cos(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \|\vec{v}\|_2}$$
- Scaled to $[0, 100]$:
  $$S_{\text{sem}} = \max(0, \min(100, \cos(\vec{u}, \vec{v}) \times 100))$$

#### Guaranteed Offline Fallback: Sublinear TF-IDF + Cosine Similarity
- Employs Scikit-Learn's `TfidfVectorizer`:
  - `ngram_range=(1, 2)`: Captures both unigrams (e.g., "principal", "trustee") and bigrams (e.g., "school turnaround", "cbse affiliation").
  - `sublinear_tf=True`: Replaces $tf$ with $1 + \log(tf)$ to prevent high-frequency term domination.
  - `stop_words='english'`: Eliminates non-informative lexical tokens.
- Calculates sparse cosine similarity across the vocabulary space.
- Zero network calls, zero GPU dependencies, and guaranteed deterministic output.

---

## 4. Multi-Component Ranking Logic

Relying solely on textual cosine similarity produces false positives (e.g., a software developer writing a school management tool might match the word "school"). Therefore, the system incorporates a **Domain-Specific Multi-Component Scoring Matrix**:

$$\text{Total Score} = \sum_{k \in \mathcal{K}} w_k \cdot S_k, \quad \text{where } \sum_{k} w_k = 1.0$$

### Scoring Components:

1. **Semantic Similarity ($S_{\text{sem}}$, Weight: 40%)**:
   Captures broader semantic context between reference and candidate documents.

2. **School Ecosystem Relevance ($S_{\text{eco}}$, Weight: 25%)**:
   Evaluates explicit involvement with school education boards (CBSE, ICSE, Cambridge, IB), K-12 affiliations, school societies, and pedagogical reforms.
   $$S_{\text{eco}} = \min\left(100, 30 + 15 \times N_{\text{eco\_indicators}} + \text{RoleBonus}\right)$$

3. **Leadership & Decision-Making Authority ($S_{\text{lead}}$, Weight: 15%)**:
   Evaluates institutional governance authority:
   - Tier 1 (100%): Principal, Headmistress, School Director, Managing Trustee, School Owner.
   - Tier 2 (80%): Vice Principal, Dean, Education Founder, Advisory Board Member.
   - Tier 3 (50%): Academic Coordinator, Consultant, Lead Auditor.
   - Tier 4 (15%): Teacher, Analyst, Non-education roles.

4. **Multi-School / Advisory Reach ($S_{\text{multi}}$, Weight: 15%)**:
   Evaluates whether the candidate operates across multiple campuses:
   - Verified multi-school keywords ("cluster", "network of 12 schools", "15+ private schools", "group of schools").
   - Explicit advisory or audit experience across independent institutions.

5. **Skills & Experience Alignment ($S_{\text{skill}}$, Weight: 5%)**:
   Combines tenure alignment with Jaccard overlap of extracted competency tokens:
   $$J(\mathcal{S}_{\text{ref}}, \mathcal{S}_{\text{cand}}) = \frac{|\mathcal{S}_{\text{ref}} \cap \mathcal{S}_{\text{cand}}|}{|\mathcal{S}_{\text{ref}} \cup \mathcal{S}_{\text{cand}}|}$$

### Score Normalization & Boundary Protections
- All components $S_k$ are guaranteed to lie within $[0.0, 100.0]$.
- Dynamic weight normalization: If a user specifies custom weights that do not sum to 100%, the engine normalizes:
  $$w'_k = \frac{w_k}{\sum_j w_j}$$
- **Self-Exclusion Rule**: A candidate identical to the reference profile is automatically filtered out ($R \neq \text{Ref}$).

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
The local SQLite database (`data/search_history.db`) implements three relational tables:

```sql
-- 1. Search Sessions Audit Log
CREATE TABLE search_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    dataset_mode TEXT NOT NULL,
    reference_name TEXT,
    reference_role TEXT NOT NULL,
    reference_org TEXT,
    reference_exp REAL,
    ranking_engine TEXT NOT NULL,
    weights_json TEXT NOT NULL,
    total_evaluated INTEGER NOT NULL,
    total_returned INTEGER NOT NULL
);

-- 2. Ranked Candidates Cache
CREATE TABLE ranked_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    profile_id TEXT NOT NULL,
    name TEXT NOT NULL,
    current_role TEXT NOT NULL,
    organization TEXT,
    location TEXT,
    relevance_score REAL NOT NULL,
    component_scores_json TEXT NOT NULL,
    explanation TEXT NOT NULL,
    source_type TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES search_sessions(id) ON DELETE CASCADE
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
    experience_years REAL,
    school_ecosystem_indicators TEXT,
    skills TEXT,
    source_type TEXT NOT NULL,
    source_description TEXT,
    source_url TEXT,
    last_verified_at TEXT,
    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

All queries use parameterized statements (`?`) to prevent SQL injection vulnerabilities.

---

## 7. Limitations & Compliant Path to Production

| Aspect | Current MVP Implementation | Enterprise Production Roadmap |
| :--- | :--- | :--- |
| **Data Ingestion** | Synthetic benchmark (30 records) & lawful CSV import | LinkedIn Learning Solutions / Official Partner Enterprise API |
| **NLP Vectorization** | Sublinear TF-IDF + Cosine Similarity (Offline) | Fine-tuned domain Sentence Transformer (`bge-small-en-v1.5` or `all-MiniLM-L6-v2`) |
| **Relationship Modeling**| 1st-degree profile attribute matching | Property Graph traversal (Neo4j / BigQuery Graph) linking shared alumni boards & trusts |
| **Search Scale** | In-memory NumPy/Pandas processing (<10,000 profiles) | Milvus / Qdrant Vector Indexing with BigQuery Data Pipeline |
| **Authentication** | Local desktop execution / Streamlit session state | Enterprise SSO (Okta / Google OAuth2) with Role-Based Access Control |
