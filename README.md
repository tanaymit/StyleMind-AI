# StyleMind AI

> *An AI that knows what to wear better than you do. (Probably.)*

StyleMind is a personal AI stylist. You describe what you need ("smart casual dinner, Pittsburgh, October, around $150") and it figures out the outfit, finds real products from a 44K-item catalog, and writes you a little lookbook. It also remembers your taste across sessions, so it gets more useful the more you use it.

**Course:** 11-766 Large Language Model Applications (CMU)  
**Author:** Tanay Mittal (tanaymit@andrew.cmu.edu)

---

## What it does

You type: *"Smart casual dinner in Pittsburgh, October, budget around $150"*

StyleMind runs through six steps:

1. **Intent Parser** - reads your request and pulls out occasion, formality, budget, season, gender
2. **Style Planner** - designs outfit concepts with reasoning for why each item works
3. **Product Retriever** - searches a 41K-item catalog using semantic search + category filters
4. **Outfit Assembler** - picks the best product per slot and scores the outfit
5. **Lookbook Generator** - writes a short editorial blurb for each outfit
6. **Profile Updater** - saves what you liked so future recommendations feel personal

Next time you ask, it already knows you hate cargo pants.

---

## Models used

StyleMind uses two tiers of models. Creative work gets the bigger model; mechanical extraction gets the smaller one.

| Agent | Tier | Model |
|---|---|---|
| Intent Parser | Light | Claude Haiku 4.5 (Bedrock) |
| Profile Updater | Light | Claude Haiku 4.5 (Bedrock) |
| Style Planner | Heavy | Claude Sonnet 4.6 (Bedrock) |
| Lookbook Generator | Heavy | Claude Sonnet 4.6 (Bedrock) |
| Embeddings | - | text-embedding-3-small (OpenAI) |

**Cost per session (5 agent calls):** ~$0.03 to $0.08  
**Cost to build the FAISS index from the full catalog:** ~$0.10, one time

---

## How it works

```
Your fashion emergency
         |
         v
+---------------------+
|   Intent Parser     |  <-- taste profile (gender, colors, past rejections)
|   Haiku 4.5         |
|   -> structured JSON|
+----------+----------+
           |
           v
+---------------------+
|   Style Planner     |  <-- taste profile again (it matters)
|   Sonnet 4.6        |
|   -> outfit concepts|  abstract ideas, no specific products yet
+----------+----------+
           |
           v
+---------------------+
|  Product Retriever  |  parallel FAISS search across all outfit slots
|  FAISS + embeddings |  hard filters: wallets can't be outerwear, etc.
|  41,888 real items  |  gender + season filters never get relaxed
+----------+----------+
           |
           v
+---------------------+
|  Outfit Assembler   |  scores: 50% semantic match + 30% color harmony
|  (no LLM needed)   |  + 20% budget fit. Rule-based, basically instant.
+----------+----------+
           |
           v
+---------------------+
|  Lookbook Generator |  writes editorial-style prose for each outfit
|  Sonnet 4.6         |  both outfits generated in parallel
+----------+----------+
           |
           v
+---------------------+
|  Profile Updater    |  turns your feedback into a structured update
|  Haiku 4.5          |  and saves it to your profile JSON
+---------------------+
           |
           v
    You look great.
    (Probably.)
```

**Total time: about 7 to 12 seconds** (parallel retrieval + parallel lookbook generation)

---

## Project structure

```
stylemind/
+-- app.py                    # Streamlit UI
+-- pipeline.py               # wires all agents together, handles parallelism
+-- config.py                 # models, paths, API keys (gitignored, copy from config.py.example)
+-- setup.py                  # builds the FAISS index from the catalog (run once)
+-- data_pipeline.py          # cleans the raw Kaggle CSV (run once)
+-- requirements.txt
+-- .env.example
+-- config.py.example
|
+-- agents/
|   +-- llm_client.py         # unified Bedrock / OpenAI client
|   +-- taste_profile.py      # user profile schema and save/load
|   +-- intent_parser.py      # natural language -> structured constraints
|   +-- style_planner.py      # constraints -> outfit blueprints
|   +-- retriever.py          # FAISS search + metadata filters
|   +-- outfit_assembler.py   # picks products and scores outfits
|   +-- lookbook_generator.py # writes the outfit prose
|   +-- profile_updater.py    # updates profile from feedback
|
+-- data/                     # catalog CSVs and product images (gitignored)
+-- embeddings/               # FAISS index, built by setup.py (gitignored)
+-- profiles/                 # user profiles saved as JSON (gitignored)
+-- tests/
    +-- test_data.py          # data sanity checks, no API needed
    +-- test_local.py         # schema and factory tests
    +-- test_integration.py   # full pipeline test, needs credentials
```

---

## Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up config files
```bash
cp .env.example .env
cp config.py.example config.py
```

Fill in your credentials in `.env`:
```
OPENAI_API_KEY=sk-...

LLM_PROVIDER=bedrock

# if using Bedrock:
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1

# if using OpenAI for everything instead:
# LLM_PROVIDER=openai
```

### 3. Enable Bedrock model access (Bedrock users only)
Go to AWS Console > Bedrock > Model access and request access for:
- Anthropic Claude Sonnet 4.6
- Anthropic Claude Haiku 4.5

Takes a few minutes and is free. AWS just wants you to click agree.

### 4. Get the catalog data
Download the [Fashion Product Images dataset from Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) and put the CSVs in `data/`. Then run:
```bash
python data_pipeline.py
```

### 5. Build the FAISS index
```bash
python setup.py
```
Takes about 3 to 5 minutes and costs ~$0.10 in OpenAI embedding credits. Only needs to run once.

### 6. Run the app
```bash
streamlit run app.py
```

### 7. Use it
- Enter a **User ID** in the sidebar (anything works, like `alex` or `tanay`)
- Set your **gender** preference so the catalog filters correctly
- Type what you need and hit **Style me**
- Accept an outfit to save it to your profile

---

## Switching providers

```bash
# default: AWS Bedrock
LLM_PROVIDER=bedrock

# fallback: everything through OpenAI
LLM_PROVIDER=openai
```

Embeddings always go through OpenAI regardless of this setting (Bedrock does not offer text-embedding-3-small).

---

## Design decisions

**Why two model tiers?**  
Running Sonnet for every single agent call would cost about 3x more per session. Parsing a sentence into JSON and diffing a profile are mechanical tasks that Haiku handles fine. Sonnet is only used where creative reasoning actually matters.

**Why FAISS plus metadata filters instead of pure vector search?**  
Pure semantic search will happily return a wallet when you ask for outerwear. The metadata filters (slot maps to allowed categories) prevent that. Semantic search figures out "what kind of item"; the filters ensure it is actually that category.

**Why are gender and season never relaxed in the fallback cascade?**  
A summer outfit with a winter coat is worse than showing fewer results. Other constraints like exact color, formality level, and budget can be relaxed when results are thin. Gender and season cannot.

**Why enforce gender in code after parsing?**  
LLMs are probabilistic. Even with the profile loaded, the intent parser might output "unspecified" for gender. So after parsing, the pipeline checks and overwrites it if the profile has a gender set. The LLM is the first line; the code check is the backup.

**Why color family scoring instead of string matching?**  
Navy and charcoal go together. Red and yellow-gold do not. A string match cannot know this. The color family compatibility matrix captures these relationships so outfit color coherence is scored meaningfully.

---

## Taste profile

Each user gets a `profiles/{user_id}.json` that persists across sessions:

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

The profile feeds into intent parsing, style planning, retrieval, and outfit scoring. It gets more accurate with each session.

---

## Running tests

```bash
python tests/test_data.py           # no API calls needed
python tests/test_local.py          # schema and factory tests
python tests/test_integration.py    # full pipeline, needs credentials
```

---

## Known limitations

- **Prices are simulated.** The Kaggle catalog has no real prices, so they are generated from category-based ranges (e.g. shirts $25-$80, blazers $50-$200). Not real retail prices.
- **No e-commerce integration.** This is a research prototype. You cannot actually buy anything.
- **LLM outputs vary.** The same request can produce different outfits on repeated runs. Mostly fine, occasionally surprising.
- **Cold start.** New profiles get broadly appealing suggestions. It improves as you use it.

---

*Built for CMU 11-766 · Spring 2026*
