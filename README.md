# Compliance Risk Prediction Model
### Predictive Risk Management for Financial Compliance
**Prepared by The Talent Grid · Prepared for Deloitte · Version 1.0**

---

> **TL;DR** — An AI-powered compliance risk scoring system that ingests financial transactions, enriches them with KYC/sanctions/policy context via MCP tools, scores them with XGBoost + SHAP, generates plain-English explanations via LLM, and routes high-risk cases to a human reviewer dashboard. Built as a production-architecture POC running at **$7/month**.

---

## Table of Contents

1. [What This Is](#1-what-this-is)
2. [Architecture Overview](#2-architecture-overview)
3. [GenAI / LLM Strategy](#3-genai--llm-strategy)
4. [MCP Integration](#4-mcp-integration)
5. [Tech Stack](#5-tech-stack)
6. [Repository Structure](#6-repository-structure)
7. [Quick Start](#7-quick-start)
8. [Workflow & Decision States](#8-workflow--decision-states)
9. [Cost Breakdown](#9-cost-breakdown)
10. [Key Design Decisions](#10-key-design-decisions)
11. [CV Talking Points](#11-cv-talking-points)
12. [Glossary](#12-glossary)

---

## 1. What This Is

A **POC implementation** of the Deloitte Compliance Risk Prediction Model SRS. The system:

- Ingests financial transaction events (real-time stream or batch)
- Enriches each transaction with customer risk, sanctions, PEP, jurisdiction, and policy context
- Scores each transaction using a trained XGBoost model with SHAP explainability
- Generates plain-English reviewer explanations using an LLM (Gemini Flash)
- Provides holistic advisory recommendations using a second LLM (Groq / Llama 70B)
- Routes HIGH and MEDIUM risk cases to a Streamlit reviewer dashboard
- Logs every inference, tool call, and reviewer action to an immutable audit trail

**This is a CV / portfolio project** demonstrating enterprise-grade architecture on a $7/month stack. The same architecture document describes how this scales to a production Deloitte deployment on AWS (Kafka, EKS, RDS, MLflow, HashiCorp Vault).

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                                 │
│   Synthetic transactions · JSON policy files · Mock sanctions  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│               DATA INGESTION (Upstash Kafka)                    │
│         Schema validation · lineage tagging · dead-letter      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│            FEATURE ENGINEERING (Pandas + Upstash Redis)         │
│   Velocity · deviation · geo · counterparty · KYC · behavioral │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────┼──────────────┐
              │             │              │
┌─────────────▼──┐  ┌───────▼──────┐  ┌───▼──────────────────┐
│  MCP Server    │  │  ML Scorer   │  │  LLM Components      │
│  (mcp SDK,     │  │  XGBoost +   │  │  Explainer (Gemini)  │
│   free)        │  │  SHAP        │  │  Judge (Groq/Llama)  │
└─────────────┬──┘  └───────┬──────┘  └───┬──────────────────┘
              └─────────────┼──────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│           UNIFIED RISK OUTPUT + GOVERNANCE GATE                 │
│    Score · class · explanation · policy citations · audit log   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│         DECISION WORKBENCH (Streamlit on HF Spaces)             │
│   Alert queue · evidence panel · reviewer actions · audit log   │
│              *** HUMAN ALWAYS IN THE LOOP ***                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. GenAI / LLM Strategy

Three-role hybrid model. Traditional ML handles scoring; LLMs handle communication and reasoning.

| Role | Component | Model (POC) | Model (Production) | Purpose |
|------|-----------|-------------|-------------------|---------|
| A | ML Scorer | XGBoost (local) | XGBoost on EKS | Numerical risk score, fast, auditable |
| B | LLM Explainer | Gemini 1.5 Flash (free API) | Claude Sonnet 4.6 | Plain-English explanation + policy citations |
| C | LLM-as-Judge | Groq Llama 3.1 70B (free) | Claude Sonnet 4.6 | Holistic advisory recommendation |

### When each LLM fires

```
Transaction scored
      │
      ├── risk_class == LOW  ──► Skip LLM · log · done
      │
      └── risk_class == HIGH or MEDIUM
                 │
                 ├──► LLM Explainer (async)
                 │     Input:  top-5 SHAP + policy passages from MCP
                 │     Output: explanation_text + reason_codes + policy_citations
                 │
                 └──► LLM-as-Judge (async)
                       Input:  full case + similar cases from MCP
                       Output: ESCALATE | FLAG | MONITOR | CLOSE + narrative
```

### LLM governance rules (all enforced in code)

- All LLM prompts are versioned in `prediction/prompts/`
- LLMs respond only in structured JSON — validated against Pydantic models before use
- Every prompt gets a SHA-256 hash stored in the audit log
- `hallucination_check.py` verifies every cited policy ID was actually supplied in the prompt
- LLM outputs are **advisory only** — they never trigger state changes autonomously

---

## 4. MCP Integration

MCP (Model Context Protocol) is the **controlled data access layer** between the LLMs and all enterprise data. No LLM ever queries a database directly.

### How it works in this POC

```
LLM (Gemini / Groq)
      │
      │  tool_call: get_transaction_context(transaction_id)
      ▼
MCP Server (mcp Python SDK, runs as sidecar on Render)
      │
      │  1. Auth check (role validation)
      │  2. Input sanitization (injection prevention)
      │  3. Query backend
      │  4. Field-level output filtering
      │  5. Audit log → Neon Postgres
      ▼
LLM receives filtered, permitted response
```

### MCP tools implemented

| Tool | Backend (POC) | Backend (Production) | Output |
|------|--------------|---------------------|--------|
| `get_transaction_context` | Neon Postgres | Core banking API | Amount, channel, timestamp, origin, destination |
| `get_customer_risk` | Neon Postgres | KYC / CRM system | Risk rating, onboarding status, prior alert count |
| `retrieve_policy` | JSON files in `/policies/` | Versioned policy repo | Policy passages with version IDs |
| `check_sanctions` | Mock fixture `/fixtures/sanctions_mock.json` | OFAC / UN / EU watchlist APIs | Match confidence, watchlist source |
| `get_similar_cases` | Neon Postgres | Case history DB | Prior cases with dispositions |
| `log_audit_event` | Neon Postgres (append-only) | S3 WORM + Kafka | Immutable event with hash chain |

### Running the MCP server

```bash
# Option A — sidecar alongside FastAPI on Render (no extra cost)
uvicorn mcp.mcp_server:app --port 8001 &
uvicorn api.main:app --port 8000

# Option B — separate Hugging Face Space (also free)
# Deploy mcp/ directory as its own HF Space
# Set MCP_SERVER_URL env var in main Render service
```

---

## 5. Tech Stack

### POC ($7/month)

| Layer | Tool | Cost | Notes |
|-------|------|------|-------|
| Message streaming | Upstash Kafka | $0 | Real Kafka API, serverless, pay-per-message |
| Feature cache | Upstash Redis | $0 | Serverless Redis, 10K cmds/day free |
| Database | Neon Serverless Postgres | $0 | Real PostgreSQL, scale-to-zero, never expires |
| ML model | XGBoost + SHAP | $0 | Open source, trained locally |
| LLM Explainer | Google Gemini 1.5 Flash | $0 | 1M tokens/day free tier |
| LLM Judge | Groq Llama 3.1 70B | $0 | 14,400 req/day free tier |
| MCP server | mcp Python SDK | $0 | Open source, runs as sidecar |
| API hosting | Render.com Starter | **$7** | Always-on FastAPI, 512MB RAM |
| Reviewer UI | Hugging Face Spaces | $0 | Streamlit, 2vCPU/16GB, public URL |
| Model registry | DagsHub (MLflow) | $0 | Free MLflow tracking server |
| CI/CD | GitHub Actions | $0 | 2,000 min/month free |
| Secrets | Render env vars | $0 | Encrypted, per-service |
| **Total** | | **$7/mo** | |

### Production equivalent (for reference)

| POC tool | Production replacement | Monthly cost |
|----------|----------------------|-------------|
| Upstash Kafka | AWS MSK (3-broker cluster) | ~$460 |
| Upstash Redis | AWS ElastiCache (cache.m5.large) | ~$115 |
| Neon Postgres | AWS RDS PostgreSQL Multi-AZ | ~$260 |
| Gemini / Groq | Claude Sonnet 4.6 API | Variable |
| Render | AWS EKS (3× m5.xlarge nodes) | ~$420 |
| HF Spaces | AWS EKS + ALB | included above |
| DagsHub | Self-hosted MLflow + S3 | ~$50 |
| GitHub Actions | GitHub Actions (same) | $0 |

---

## 6. Repository Structure

```
compliance-risk-poc/
│
├── ingestion/
│   ├── kafka_consumer.py          # Upstash Kafka consumer
│   ├── synthetic_generator.py     # Generates fake transactions for POC
│   ├── schema_validator.py        # Pydantic schema validation
│   └── schemas/
│       └── transaction_event.py   # TransactionEvent Pydantic model
│
├── features/
│   ├── feature_pipeline.py        # Orchestrates all feature builders
│   ├── velocity_features.py
│   ├── deviation_features.py
│   ├── geo_features.py
│   ├── kyc_features.py
│   ├── behavioral_features.py
│   └── redis_client.py            # Upstash Redis feature cache
│
├── mcp/
│   ├── mcp_server.py              # FastMCP server — all 6 tools
│   ├── tools/
│   │   ├── transaction_tool.py    # → Neon Postgres
│   │   ├── customer_tool.py       # → Neon Postgres
│   │   ├── policy_tool.py         # → /policies/*.json
│   │   ├── sanctions_tool.py      # → /fixtures/sanctions_mock.json
│   │   ├── case_similarity_tool.py# → Neon Postgres
│   │   └── audit_tool.py          # → Neon Postgres (append-only)
│   ├── auth/
│   │   └── role_validator.py      # RBAC — checks caller role before tool runs
│   └── middleware/
│       ├── input_sanitizer.py     # Strips injection attempts from params
│       └── output_filter.py       # Removes non-permitted fields from response
│
├── prediction/
│   ├── orchestrator.py            # Runs ML + both LLMs concurrently
│   ├── ml_scorer/
│   │   ├── xgboost_scorer.py
│   │   ├── shap_explainer.py
│   │   └── threshold_policy.py    # Configurable risk thresholds
│   ├── llm_explainer/
│   │   ├── explainer.py           # Gemini Flash API call
│   │   ├── prompt_builder.py
│   │   └── hallucination_check.py # Verifies policy citations
│   ├── llm_judge/
│   │   ├── judge.py               # Groq Llama 70B API call
│   │   ├── prompt_builder.py
│   │   └── output_parser.py
│   └── prompts/
│       ├── explainer_v1.txt       # Versioned prompt templates
│       └── judge_v1.txt
│
├── risk_output/
│   ├── aggregator.py              # Merges ML + LLM outputs
│   ├── governance_gate.py         # Hallucination + calibration checks
│   └── case_creator.py            # Opens workbench case for H/M
│
├── audit/
│   ├── audit_logger.py            # SHA-256 hash-chain logger
│   └── models.py                  # AuditEvent Pydantic model
│
├── api/
│   ├── main.py                    # FastAPI app
│   └── routers/
│       ├── score.py               # POST /score
│       ├── cases.py               # GET /cases, PATCH /cases/{id}
│       └── audit.py               # GET /audit/export
│
├── dashboard/
│   └── app.py                     # Streamlit reviewer dashboard
│                                  # Deployed on Hugging Face Spaces
│
├── policies/
│   ├── aml.json                   # AML policy passages (mock)
│   ├── kyc.json                   # KYC policy passages (mock)
│   └── jurisdiction_risk.json     # Jurisdiction risk scores
│
├── fixtures/
│   └── sanctions_mock.json        # Mock sanctions watchlist for POC
│
├── models/
│   └── xgboost_v1.json            # Trained model (committed to repo)
│
├── notebooks/
│   ├── 01_synthetic_data.ipynb    # Generate training data
│   ├── 02_model_training.ipynb    # Train XGBoost + SHAP analysis
│   └── 03_end_to_end_demo.ipynb   # Full pipeline walkthrough
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── config/
│   ├── settings.py                # Pydantic Settings (reads from env)
│   └── thresholds.yaml            # Risk score thresholds by product/jurisdiction
│
├── .github/
│   └── workflows/
│       └── deploy.yml             # GitHub Actions → auto-deploy to Render
│
├── render.yaml                    # Render deployment config
├── requirements.txt
├── pyproject.toml
├── .env.example                   # Template — never commit .env
└── README.md                      # This file
```

---

## 7. Quick Start

### Prerequisites

- Python 3.11+
- Git
- Accounts (all free): Upstash, Neon, Google AI Studio, Groq, Render, Hugging Face, DagsHub

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/compliance-risk-poc.git
cd compliance-risk-poc
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Fill in:
# UPSTASH_KAFKA_URL, UPSTASH_KAFKA_TOKEN
# UPSTASH_REDIS_URL, UPSTASH_REDIS_TOKEN
# NEON_DATABASE_URL
# GEMINI_API_KEY          (from Google AI Studio — free)
# GROQ_API_KEY            (from console.groq.com — free)
# MCP_SERVER_URL          (http://localhost:8001 for local dev)
```

### 3. Set up the database

```bash
python scripts/init_db.py
# Creates: transactions, cases, audit_events, reviewer_actions tables in Neon
```

### 4. Generate synthetic training data and train the model

```bash
jupyter notebook notebooks/01_synthetic_data.ipynb
jupyter notebook notebooks/02_model_training.ipynb
# Outputs: models/xgboost_v1.json + models/shap_explainer.pkl
```

### 5. Run locally

```bash
# Terminal 1 — MCP server
uvicorn mcp.mcp_server:app --port 8001 --reload

# Terminal 2 — FastAPI scoring API
uvicorn api.main:app --port 8000 --reload

# Terminal 3 — Streamlit dashboard
streamlit run dashboard/app.py

# Terminal 4 — Generate and score some transactions
python ingestion/synthetic_generator.py --count 50
```

### 6. Test the full pipeline

```bash
# Score a single transaction
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN-001",
    "customer_id": "CUST-42",
    "amount": 95000.00,
    "currency": "USD",
    "channel": "WIRE",
    "origin_account": "ACC-123",
    "destination_account": "ACC-456",
    "jurisdiction_origin": "US",
    "jurisdiction_destination": "AE",
    "product_type": "WIRE_TRANSFER",
    "timestamp": "2026-06-13T10:00:00Z"
  }'

# Expected response includes:
# risk_score, risk_class, shap_explanation,
# llm_explanation (for HIGH/MED), llm_judge_recommendation,
# case_id, audit_event_id
```

### 7. Deploy to production

```bash
# Push to GitHub → GitHub Actions auto-deploys to Render
git push origin main

# Dashboard deploys separately to HF Spaces
# (connect HF Space to your GitHub repo — auto-deploys on push)
```

---

## 8. Workflow & Decision States

```
RECEIVED → SCORED → CASE_CREATED → UNDER_REVIEW → [ESCALATED] → CLOSED
```

| State | Description | Exit criteria |
|-------|-------------|--------------|
| `RECEIVED` | Transaction accepted by ingestion | Schema valid |
| `SCORED` | ML + LLM outputs produced | All governance checks passed |
| `CASE_CREATED` | Alert opened for HIGH/MEDIUM | Evidence package attached |
| `UNDER_REVIEW` | Reviewer examining evidence | Decision selected |
| `ESCALATED` | Routed to manager / legal | Escalation disposition recorded |
| `CLOSED` | Final decision recorded | Reason code set, audit locked |

### Reviewer actions (all logged, immutable)

`APPROVE` · `REJECT` · `ESCALATE` · `HOLD` · `CLOSE`

Every action requires a `reason_code`. Every action is written to the audit chain immediately and cannot be modified.

---

## 9. Cost Breakdown

### POC — actual monthly spend

| Service | What it replaces | Cost |
|---------|-----------------|------|
| Render.com Starter (FastAPI) | AWS EKS | $7.00 |
| Upstash Kafka | AWS MSK | $0 |
| Upstash Redis | AWS ElastiCache | $0 |
| Neon Serverless Postgres | AWS RDS | $0 |
| Gemini 1.5 Flash | Claude Sonnet 4.6 | $0 |
| Groq Llama 70B | Claude Sonnet 4.6 | $0 |
| HF Spaces (Streamlit) | AWS EKS + ALB | $0 |
| DagsHub (MLflow) | Self-hosted MLflow | $0 |
| GitHub Actions | GitHub Actions | $0 |
| **Total** | | **$7/mo** |

### Production scale (reference)

At 500K transactions/month with 8% flagged HIGH/MEDIUM:
- Infrastructure (MSK + RDS + EKS + Redis): ~$1,200/mo
- LLM (Claude Sonnet 4.6, with prompt caching): ~$600–900/mo
- Tooling (MLflow, monitoring, secrets): ~$550/mo
- **Total: ~$2,400–2,600/mo**

Key cost lever: **flag rate**. Reducing false positives at the ML threshold level directly cuts LLM spend. The batch API (50% LLM discount) applies for non-real-time queues.

---

## 10. Key Design Decisions

### Why XGBoost and not a pure LLM for scoring?

XGBoost scores in under 50ms, produces auditable SHAP values, and regulators understand it. LLMs are slower, harder to audit at the feature level, and can hallucinate numerical outputs. The hybrid approach gives speed and auditability from ML with communication quality from LLMs.

### Why MCP instead of direct database queries from the LLM?

Direct DB access from LLMs risks: unscoped data access, PII leakage, prompt injection via data content, and no audit trail. MCP enforces a strict tool contract — every call is authenticated, parameter-validated, output-filtered, and logged before the LLM sees the result. This is the key architectural difference between this system and a naive RAG implementation.

### Why run LLM Explainer and LLM-as-Judge concurrently?

They have different inputs and neither depends on the other's output. Concurrent execution (`asyncio.gather`) cuts end-to-end latency roughly in half for HIGH/MEDIUM cases compared to sequential execution.

### Why is the LLM-as-Judge advisory only?

The SRS explicitly states model outputs are decision-support signals and do not constitute final regulatory judgment without authorized review. Any auto-closure would violate this. The judge's output is displayed alongside evidence in the reviewer's dashboard — the human decides and records the final disposition.

### Why a hash-chain audit log?

A simple append-only log can be backdated by a malicious insider. A SHA-256 hash chain means any insertion or modification of a past record breaks the chain and is immediately detectable. This satisfies NFR-06 (tamper-evident auditability) and regulator expectations.

### Why Gemini Flash + Groq instead of a single LLM?

Different tools for different roles. Gemini Flash is optimised for fast structured extraction (explanation + citation) — ideal for the Explainer. Groq's inference speed on Llama 70B makes it excellent for the more complex Judge reasoning. Using two different providers also demonstrates provider-agnostic LLM integration, which is itself a valuable architectural skill.

---

## 11. CV Talking Points

When discussing this project in interviews:

**On the architecture:**
> "I built a hybrid ML + LLM compliance risk system following a real Deloitte SRS. The ML layer — XGBoost with SHAP — handles fast, auditable risk scoring. Two LLM components handle explanation and advisory reasoning. The key architectural constraint I respected is that LLM outputs are advisory only — human reviewers always make the final call, which is a hard regulatory requirement."

**On MCP:**
> "I implemented the MCP layer as the controlled data access bus between the LLMs and all data sources. No LLM queries a database directly — every tool call is authenticated, input-sanitized, output-filtered, and audit-logged before the LLM sees the result. This prevents prompt injection, PII leakage, and unscoped data access — which are the three biggest failure modes in naive LLM-plus-database systems."

**On cost:**
> "The production architecture runs on AWS at roughly $2,400/month at mid-scale. For this POC I mapped every production component to a free-tier equivalent — Upstash for Kafka and Redis, Neon for PostgreSQL, Gemini Flash and Groq for the LLMs, Hugging Face Spaces for the dashboard. The only paid component is a $7/month Render instance to keep the API always-on for demos. That cost discipline is itself a design skill."

**On the audit trail:**
> "Every model inference, MCP tool call, and reviewer action is written to an immutable hash-chain audit log in Postgres. Any tampering with a past record breaks the chain. This matches what regulators expect from a production compliance system — complete traceability from transaction receipt to final disposition."

**On the LLM hallucination check:**
> "The LLM Explainer is explicitly instructed to cite only policy passages supplied in its prompt. I then run a post-response check that verifies every cited policy ID exists in what we actually sent. If the LLM fabricates a citation, the check fails and the explanation is flagged — it never reaches the reviewer. That's how you use LLMs safely in a regulated environment."

---

## 12. Glossary

| Term | Definition |
|------|-----------|
| AML | Anti-Money Laundering — controls for identifying suspicious financial activity |
| KYC | Know Your Customer — identity verification and customer risk assessment |
| PEP | Politically Exposed Person — individual with elevated financial crime risk |
| MCP | Model Context Protocol — open standard for controlled tool and context access by AI |
| SHAP | SHapley Additive exPlanations — framework for explaining individual ML predictions |
| Risk Score | Float 0.0–1.0 — probability that a transaction presents material compliance risk |
| Risk Class | HIGH / MEDIUM / LOW — categorical label derived from risk score + threshold policy |
| Evidence Pack | Structured export of transaction facts, model outputs, policy citations, and reviewer decisions |
| LLM-as-Judge | An LLM used to provide holistic advisory reasoning — not to make final decisions |
| Hash chain | Audit log where each event includes the SHA-256 hash of the previous event |
| Feature vector | Structured numerical inputs derived from a transaction, fed to the ML model |
| Hallucination check | Post-LLM validation that verifies all cited sources were actually supplied in the prompt |
| Governance gate | Pre-delivery validation layer that checks LLM output quality before it reaches reviewers |

---

## Links

| Resource | URL |
|----------|-----|
| Live demo (Streamlit dashboard) | `https://huggingface.co/spaces/YOUR_USERNAME/compliance-risk-dashboard` |
| API docs (FastAPI auto-generated) | `https://YOUR_APP.onrender.com/docs` |
| MLflow experiment tracking | `https://dagshub.com/YOUR_USERNAME/compliance-risk-poc/experiments` |
| Architecture document | `COMPLIANCE_RISK_MODEL_ARCHITECTURE.md` |
| Original SRS | `Deloitte_Compliance_Risk_Prediction_SRS_TalentGrid.pdf` |

---

*Built by The Talent Grid · Portfolio / CV project based on Deloitte Financial Compliance Risk Program SRS v1.0*
*Production architecture designed for enterprise deployment · POC runs at $7/month on free-tier cloud services*
