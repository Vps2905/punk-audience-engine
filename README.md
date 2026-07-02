# Audience Intelligence Agents

## 1. Project Overview
This project is a standalone Audience Intelligence Agents / Audience Intelligence Engine demo that converts privacy-safe audience signals into clean features, embeddings/vectors, cohorts, lookalikes, synthetic seed profiles, and Meta-safe export packages.

## 2. What This Demo Covers
This v1 demo covers the following five core modules:
- Module 1: Ingestion & Privacy Layer
- Module 2: Embedding & Feature Store
- Module 3: Cohort Management & Lookalike
- Module 4: Meta Safe Export
- Module 5: Simple Conversational Trigger

*Note: Advanced multi-agent orchestration and swarm/background evolution are intentionally not included in this v1 demo.*

## 3. High-Level Architecture

```text
Data Source
  -> Ingestion & Privacy Layer
  -> Synthetic Data Generation
  -> Embedding & Feature Store
  -> Cohort Management & Lookalike
  -> Meta Safe Export
  -> Simple UI / Conversational Trigger
```

## 4. Current Real DB Demo Result
Example latest local run results:
- Raw rows loaded: 269
- Deduped sessions: 256
- Safe cohorts: 122
- Clusters: 8
- Synthetic seed profiles: 1000
- Export status: pending_approval

*Note: These are sample local run numbers and may change when DB data changes.*

## 5. Module-by-Module Details

### Module 1: Ingestion & Privacy Layer
**Goal:** Safely ingest and clean data.  
**Current output:**
- clean_feature_table.csv
- privacy_report.json
- hashing_manifest.json
- dummy_hashing_demo.csv
- k_anonymity_dp_report.csv
- lineage_report.json

**Details:**
- raw identifiers are not exported
- real hashed MAIDs are also not exported in review package
- k-anonymity uses min cohort size
- basic DP noise is applied
- lineage tracks source to transformation to output

### Module 2: Embedding & Feature Store
**Current output:**
- cohort_vectors.npy
- cohort_metadata.csv
- vector_preview.csv
- embedding_manifest.json
- similarity_search_demo.json

**Details:**
- current v1 uses TF-IDF vectors
- production plan supports sentence-transformers/OpenAI-compatible embeddings
- current vector store is local file
- future vector DB can be Weaviate/Pinecone/Qdrant/pgvector

### Module 3: Cohort Management & Lookalike
**Current output:**
- top_cohorts.csv
- cluster_summary.csv
- cohort_quality_report.json
- lookalike_demo.json

**Details:**
- clustering creates cohort groups
- quality scoring checks size/coherence
- current lookalike is basic same-cluster/safe-trait similarity
- future lookalike will be stronger vector/ML-based

### Module 4: Meta Safe Export
**Current output:**
- synthetic_safe_seed_profiles.csv
- meta_safe_export_manifest.json
- export_package_summary.json
- approval_request.json

**Details:**
- export is pending approval
- no automatic external upload
- no raw MAIDs, raw lat/lng, email, phone, or individual-level data
- designed as Meta Advantage+ compatible seed package concept

### Module 5: Simple Conversational Trigger
**Current output:**
- sample_user_requests.json
- chat_orchestration_demo.json
- endpoint_mapping.json

**Details:**
- basic deterministic trigger demo
- example: user asks “create montreal restaurant evening cohort and prepare safe export”
- flow maps request to ingestion, embedding, cohort, export modules
- full LLM/LangGraph orchestration is future phase

## 6. Repository Structure
Key directories in this project:
- `app/agents`
- `app/api`
- `app/static`
- `scripts`
- `samples`
- `tests`
- `data`

**Excluded Folders:**
The following generated data folders contain outputs that are ignored by Git to preserve privacy and prevent bloating:
- `data/review_packages/`
- `data/swarm_outputs/`
- `data/postgres_outputs/`
- `data/swarm_monitor/`

## 7. Running Locally

To run the application locally, execute the following commands:

```bash
cd ~/Downloads/punk-audience-engine
source .venv/bin/activate
python3 -m compileall app
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

- **UI:** http://127.0.0.1:8000/ui/audience-agents
- **API latest output:** http://127.0.0.1:8000/agents/latest-five-module-demo-output

## 8. Generate Five-Module Demo Outputs

Run these scripts to generate the output packages:

```bash
python3 scripts/generate_five_module_demo_outputs.py
python3 scripts/split_module_outputs.py
```

These scripts will generate output folders in the following structure:
```text
data/review_packages/five_module_demo_<timestamp>_<run_id>/
  01_ingestion_privacy/
  02_embeddings_feature_store/
  03_cohort_management/
  04_meta_safe_export/
  05_simple_conversational_trigger/
  separate_module_zips/
```

## 9. API Endpoints

### Latest Endpoints
- `POST /agents/generate-five-module-demo-output`
- `GET /agents/latest-five-module-demo-output`
- `GET /agents/download-five-module-demo/{artifact_name}`
- `POST /agents/run-maid-swarm`
- `POST /agents/monitor-swarm-source`

### Older/Demo Endpoints
- `POST /ingest`
- `POST /synthetic/generate`
- `POST /embed`
- `POST /search/similar`
- `POST /cluster`
- `POST /cohort/create`
- `POST /cohort/lookalike`
- `POST /export/meta/{cohort_id}`

## 10. Privacy and Safety Guarantees
- `.env` is never committed.
- DB credentials are never committed.
- Raw MAIDs are not exported.
- Hashed real MAIDs are not included in review outputs.
- Raw observations are not exported.
- Raw lat/lng is not exported.
- Individual-level user/device rows are not exported.
- Outputs are aggregated and/or synthetic only.
- Exports require manual approval.

## 11. What Is Demo v1 vs Production Pending

### Demo v1:
- real DB read flow
- privacy-safe cohorts
- synthetic seed profiles
- local vectors
- clustering
- basic lookalike
- export manifest
- simple UI

### Production pending:
- SDV DPGCSynthesizer integration
- sentence-transformer embeddings
- vector DB integration
- stronger lookalike agent
- production auth
- read-only DB credentials
- proper background jobs
- full approval audit trail
- real Meta API upload only after compliance approval
- advanced multi-agent orchestration
- swarm/background evolution

## 12. Git Safety
To maintain data privacy and security, do **NOT** commit the following:
- `.env`
- `data/review_packages/`
- `data/swarm_outputs/`
- `data/postgres_outputs/`
- `data/swarm_monitor/`
- `*.zip`
- raw data
- credentials

## 13. Roadmap
- **Phase 1:** v1 demo package and UI
- **Phase 2:** SyntheticEngineAgent with SDV fallback stack
- **Phase 3:** Embedding upgrade + vector DB
- **Phase 4:** LookalikeAgent
- **Phase 5:** SafeExportAgent + approval workflow
- **Phase 6:** Conversational orchestration
- **Phase 7:** Production hardening and monitoring
