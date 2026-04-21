# StyleMind AI

> *An AI that knows what to wear better than you do. (Probably.)*

An agentic LLM system that builds a persistent taste profile for each user and translates natural language fashion requests into real, scored, lookbook-quality outfit recommendations — complete with product images from a 41,000-item catalog.

**Course:** 11-766 Large Language Model Applications (CMU)  
**Author:** Tanay Mittal (tanaymit@andrew.cmu.edu)

---

## What it does

You type: *"Smart casual dinner in Pittsburgh, October, budget around $150"*

StyleMind:
1. Figures out what you actually mean (Intent Parser)
2. Designs 2-3 outfit concepts with chain-of-thought reasoning (Style Planner)
3. Finds real matching products via semantic search + filters (Retriever)
4. Assembles and scores complete outfits (Assembler)
5. Writes a mini lookbook blurb for each outfit (Lookbook Generator)
6. Remembers what you liked and updates your taste profile forever (Profile Updater)

Next time you ask, it already knows you hate cargo pants.

---

## Architecture: Tiered Model Approach

StyleMind is deliberately cheap. Creative tasks get the big model; mechanical tasks get the small one.

| Agent | Tier | Model | Reasoning |
|---|---|---|---|
| Intent Parser | Light | Claude Haiku 4.5 (Bedrock) | Structured JSON extraction — Haiku is plenty |
| Profile Updater | Light | Claude Haiku 4.5 (Bedrock) | Mechanical diff generation, not creative work |
| Style Planner | Heavy | Claude Sonnet 4.6 (Bedrock) | Chain-of-thought outfit design — needs the big brain |
| Lookbook Generator | Heavy | Claude Sonnet 4.6 (Bedrock) | Writing aspirational prose badly requires talent |
| Embeddings | — | text-embedding-3-small (OpenAI) | Cheapest good embedding model, ~$0.10 for the full catalog |

**Estimated cost for a full session (5 agent calls):** ~$0.03–0.08  
**Estimated cost to embed the whole 41K catalog:** ~$0.10 one-time

---

## How It Works (The Real Architecture)

```
Your fashion emergency
         │
         ▼
┌─────────────────────┐
│   Intent Parser      │ ◄─── Your taste profile (gender, colors, rejections, etc.)
│   Haiku 4.5 (light) │
│   → structured JSON  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Style Planner      │ ◄─── Taste profile again (it's important)
│  Sonnet 4.6 (heavy) │
│  → outfit blueprints │  (abstract concepts — no specific products yet)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Product Retriever   │  Parallel FAISS semantic search across all slots
│  FAISS + OpenAI emb  │  Hard slot→category filters (wallets ≠ outerwear)
│  41,888 real items   │  Gender + season filters that never get relaxed
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Outfit Assembler    │  Rule-based scoring: 50% semantic sim + 30% color
│  (no LLM needed)    │  harmony + 20% budget fit. Instant.
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Lookbook Generator  │  Writes the "editorial voice" prose for each outfit
│  Sonnet 4.6 (heavy) │  Both outfits generated in parallel
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Profile Updater     │  Converts your feedback into a structured diff
│  Haiku 4.5 (light)  │  and patches your taste profile JSON on disk
└─────────────────────┘
           │
           ▼
    You look great.
    (Probably.)
```

**Total wall-clock time: ~7–12 seconds** (parallel retrieval + parallel lookbook generation)

---

## Project Structure

```
stylemind/
├── app.py                    # Streamlit UI — the thing you actually see
├── pipeline.py               # Orchestrator: wires all agents together with parallelism
├── config.py                 # Central config (models, paths, API keys)
├── setup.py                  # One-command setup: builds FAISS index from catalog
├── data_pipeline.py          # Raw CSV → cleaned dataset (run once)
├── requirements.txt
├── .env.example
│
├── agents/
│   ├── llm_client.py         # Provider-agnostic LLM client (Bedrock or OpenAI)
│   ├── taste_profile.py      # The persistent brain: Pydantic schema, save/load
│   ├── intent_parser.py      # Light tier: NL request → structured JSON constraints
│   ├── style_planner.py      # Heavy tier: constraints → abstract outfit blueprints
│   ├── retriever.py          # FAISS semantic search + metadata hard filters
│   ├── outfit_assembler.py   # Rule-based assembly and scoring (no LLM)
│   ├── lookbook_generator.py # Heavy tier: outfits → editorial prose
│   └── profile_updater.py    # Light tier: feedback → profile diffs
│
├── data/
│   ├── catalog_cleaned.csv   # 41,888 fashion items with prices, colors, seasons
│   ├── images/               # 44,441 product JPEGs (referenced by item ID)
│   └── embedding_strings.csv # Pre-built strings for FAISS indexing
│
├── embeddings/               # FAISS index (built by setup.py, ~170MB)
├── profiles/                 # Persistent user profiles stored as JSON
└── tests/
    ├── test_data.py          # Data sanity checks (no API needed)
    ├── test_local.py         # Schema + factory tests
    └── test_integration.py   # Full pipeline test (needs credentials)
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure credentials
```bash
cp .env.example .env
```
Edit `.env`:
```
OPENAI_API_KEY=sk-...          # Required for embeddings
LLM_PROVIDER=bedrock           # or "openai"

# If using Bedrock:
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1

# If using OpenAI for everything:
# LLM_PROVIDER=openai
```

### 3. Enable Bedrock model access (if using Bedrock)
AWS Console → Bedrock → Model access → Request access for:
- `Anthropic Claude Sonnet 4.6` (heavy tier)
- `Anthropic Claude Haiku 4.5` (light tier)

This takes a few minutes and is free. AWS just wants you to click "agree."

### 4. Build the FAISS index
```bash
python setup.py
```
Takes ~3–5 minutes, costs ~$0.10 in OpenAI embedding credits. Do this once.

### 5. Run the app
```bash
streamlit run app.py
```

### 6. Use it
- Enter a **User ID** in the sidebar (anything, e.g. `alex` or `tanay`)
- Select your **gender** preference so the catalog is filtered correctly
- Type your styling request in the main box and hit **Style me**
- Accept an outfit to save it to your profile for future personalization

---

## Provider Switching

```bash
# Default: AWS Bedrock (uses your AWS credits)
LLM_PROVIDER=bedrock

# Fallback: everything through OpenAI
LLM_PROVIDER=openai
```

The `llm_client.py` abstraction means all agents work identically with either provider. Embeddings always go through OpenAI regardless (Bedrock doesn't offer text-embedding-3-small).

---

## Key Design Decisions

**Why tiered models?**  
Running Sonnet for everything would cost ~3× more per session. Intent parsing and profile diffing are mechanical structured extraction tasks — Haiku handles them fine at a fraction of the cost.

**Why FAISS + metadata filters instead of pure vector search?**  
Pure semantic search will happily return a wallet when you ask for outerwear. Hard categorical filters (slot → masterCategory/subCategory/articleType) prevent this. Semantic search handles the "what kind of shirt" question; the metadata filters handle "but actually a shirt, not a blazer."

**Why never relax season or gender in retrieval fallback?**  
A summer outfit with a winter coat is worse than no outfit at all. The fallback cascade only relaxes soft constraints (formality, exact color, budget) — never the hard constraints.

**Why persist gender as a code-level override in the pipeline?**  
LLMs are probabilistic. If the user said "Men's" in the sidebar, we enforce it in code after parsing, regardless of what the intent LLM outputs. Belt and suspenders.

**Why color family compatibility scoring instead of exact color matching?**  
Navy and charcoal are compatible. Red and yellow-gold are not. A rigid string-match approach can't know this. Color family membership + a compatibility matrix gives much better outfit coherence.

---

## Taste Profile

Every user gets a `profiles/{user_id}.json` that persists across sessions:

```json
{
  "user_id": "alex",
  "style_identity": "Minimalist smart casual with a streetwear edge",
  "gender_expression": "masculine",
  "color_preferences": { "preferred": ["navy", "charcoal"], "avoided": ["bright yellow"] },
  "fit_preferences": { "preferred_fits": ["slim", "tapered"], "avoided_fits": ["baggy"] },
  "rejections": [{ "item_type": "chunky sneakers", "count": 2 }],
  "profile_confidence": 0.23,
  "total_sessions": 3
}
```

The profile influences: intent parsing (context), style planning (constraints), retrieval (gender filter), and assembler (color preferences). It gets smarter with every session.

---

## Running Tests

```bash
python tests/test_data.py           # Always works — no API calls
python tests/test_local.py          # Schema + factory tests
python tests/test_integration.py    # Full pipeline — needs credentials
```

---

## Known Limitations

- **Catalog is from Kaggle** — prices are simulated, product names can be weird, images cover ~41K of 44K items.
- **No real e-commerce integration** — this is a research prototype, not a shopping app.
- **LLM outputs are non-deterministic** — the same request may produce slightly different outfits on repeated runs. That's a feature, not a bug. Mostly.
- **Cold start** — new profiles with no history get broadly appealing outfits. The more you use it, the better it gets.

---

*Built for CMU 11-766 · Spring 2026*
