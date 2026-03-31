# StyleMind AI

An agentic LLM system that builds persistent taste profiles for personalized fashion recommendations.

**Course:** 11-766 Large Language Model Applications (CMU)  
**Author:** Tanay Mittal (tanaymit)

## Architecture: Tiered Model Approach

StyleMind uses a cost-optimized tiered setup across AWS Bedrock and OpenAI:

| Component | Tier | Model | Why |
|---|---|---|---|
| Intent Parser | Light | Claude Haiku 4.5 (Bedrock) | Structured JSON extraction — cheap & fast |
| Profile Updater | Light | Claude Haiku 4.5 (Bedrock) | Mechanical diff generation |
| Style Planner | Heavy | Claude Sonnet 4.6 (Bedrock) | Creative chain-of-thought reasoning |
| Lookbook Generator | Heavy | Claude Sonnet 4.6 (Bedrock) | Aspirational prose generation |
| LLM-as-Judge | Heavy | Claude Sonnet 4.6 (Bedrock) | Evaluation scoring |
| Embeddings | — | OpenAI text-embedding-3-small | Cheapest, high quality |

Estimated cost for full evaluation (50 sessions): **~$5-8** using $400 AWS credits.

## Project Structure

```
stylemind/
├── config.py                 # Central config (providers, models, paths)
├── setup.py                  # One-command setup (embeddings + FAISS)
├── demo.py                   # End-to-end midpoint demo
├── data_pipeline.py          # Raw CSV → cleaned dataset
├── requirements.txt
├── .env.example
│
├── agents/
│   ├── llm_client.py         # Unified LLM client (Bedrock / OpenAI)
│   ├── taste_profile.py      # Taste Profile schema (Pydantic)
│   ├── intent_parser.py      # Intent Parser (light tier)
│   ├── style_planner.py      # Style Planner (heavy tier)
│   └── retriever.py          # Product Retriever (FAISS + OpenAI embeddings)
│
├── data/
│   ├── catalog_cleaned.csv   # Cleaned dataset (41,888 items)
│   ├── catalog.json          # Sample records
│   └── embedding_strings.csv # Pre-built embedding strings
│
├── embeddings/               # FAISS index + vectors (built by setup.py)
├── profiles/                 # Persistent user profiles (JSON)
└── tests/
    ├── test_data.py          # Data verification (no API needed)
    ├── test_local.py         # Schema + client factory tests
    └── test_integration.py   # Full pipeline (needs credentials)
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure credentials
```bash
cp .env.example .env
# Edit .env:
#   - Add OPENAI_API_KEY (needed for embeddings)
#   - Add AWS credentials (or use IAM role)
#   - LLM_PROVIDER defaults to "bedrock"
```

### 3. Configure AWS CLI (if not using explicit keys)
```bash
aws configure
# Or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env
```

### 4. Enable Bedrock models
In the AWS Console → Bedrock → Model access, enable:
- `Anthropic Claude Sonnet 4.6`
- `Anthropic Claude Haiku 4.5`

### 5. Build embeddings + FAISS index
```bash
python setup.py
```
Takes ~3-5 min, costs ~$0.10 in OpenAI credits.

### 6. Run the demo
```bash
python demo.py                    # Full pipeline (Bedrock + FAISS)
python demo.py --no-retrieval     # Parser + Planner only
python demo.py --provider openai  # Use OpenAI instead of Bedrock
```

### 7. Run tests
```bash
python tests/test_data.py           # Data verification (always works)
python tests/test_local.py          # Schema + factory tests
python tests/test_integration.py    # Full pipeline (needs credentials)
```

## Provider Switching

Set `LLM_PROVIDER` in `.env` to switch between providers:

```bash
# AWS Bedrock (default) — uses your $400 AWS credits
LLM_PROVIDER=bedrock

# OpenAI fallback — all calls go through OpenAI
LLM_PROVIDER=openai
```

The `llm_client.py` abstraction means all agents work identically with either provider. Embeddings always use OpenAI regardless of LLM provider.

## How It Works

```
User: "smart casual dinner, London in October, ~$200"
         │
         ▼
┌──────────────────────────┐
│  Intent Parser            │ ← Taste Profile
│  (Haiku 4.5 — light tier) │
│  → structured JSON        │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Style Planner            │ ← Taste Profile
│  (Sonnet 4.6 — heavy tier)│
│  → outfit blueprints      │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Product Retriever        │
│  (OpenAI embeddings+FAISS)│
│  → real catalog items     │
└──────────┬───────────────┘
           ▼
   [End-of-semester work]
   Assembly → Reranker → Lookbook → Feedback → Profile Update
```

## Midpoint Deliverables (Complete)

- [x] Project structure with tiered Bedrock/OpenAI architecture
- [x] Unified LLM client abstraction (provider-agnostic)
- [x] Dataset cleaned (44K → 41,888 items) with simulated prices
- [x] FAISS index pipeline with OpenAI embeddings
- [x] Intent Parser (light tier, few-shot + Pydantic)
- [x] Taste Profile schema (Pydantic, save/load, summary generation)
- [x] Style Planner (heavy tier, chain-of-thought blueprints)
- [x] Product Retriever (semantic + hybrid metadata filters)

## End-of-Semester Work (Remaining)

- [ ] Outfit Assembly & Scoring (rule-based + LLM reranker, heavy tier)
- [ ] Lookbook Generator (aspirational prose, heavy tier)
- [ ] Profile Updater agent (feedback → structured diffs, light tier)
- [ ] Streamlit interface with feedback loop
- [ ] LLM-as-judge evaluation (50 inputs, heavy tier)
- [ ] Human study (15-20 participants)
- [ ] Ablation study (with/without narrative generation)
