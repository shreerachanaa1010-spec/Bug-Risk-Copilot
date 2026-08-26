# Bug-Risk Copilot

**A production-grade AI system that predicts bug-introducing commits, retrieves historical context, and reasons through evidence to guide code review.**

## Project Overview

This is a **full-stack AI engineering portfolio piece** that demonstrates four distinct, production-level capabilities in one coherent system:

1. **Classical ML** — Train and evaluate a bug-risk classifier (XGBoost)
2. **Vector Search/RAG** — Retrieve similar historical bug commits using embeddings
3. **Agentic Orchestration** — Multi-step reasoning with conditional loops (LangGraph)
4. **Production Engineering** — REST API, testing, CI/CD integration, explainability

### Why This Project Matters for Interviews

Most portfolio projects are "thin LLM wrappers" — they slap a prompt into an LLM and call it ML. This project is fundamentally different:

- **The ML model makes the decision** (0-1 risk score from a trained classifier).
- **The LLM explains** the decision using SHAP features + retrieved context.
- **The agent orchestrates** the investigation, handling ambiguous cases with conditional loops.

This separation of concerns is what real production systems do. Interviewers specifically look for this.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     New Commit / PR Diff                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────▼───────────────┐
         │  Feature Extraction (15 feat) │
         │  • Churn, complexity          │
         │  • File/author history        │
         └───────────────┬───────────────┘
                         │
         ┌───────────────▼───────────────┐
         │  ML MODEL (XGBoost)           │────► risk_score (0-1)
         │  Trained offline              │      + SHAP features
         └───────────────┬───────────────┘
                         │
         ╔═══════════════▼═══════════════╗
         ║   LANGGRAPH AGENT             ║
         ║  ┌─ fetch_context             ║
         ║  │  ├─ investigate◄──┐        ║  FAISS vector search
         ║  │  │   (retrieval)  │loop    ║  over past bug commits
         ║  │  ├─────────────────┘       ║
         ║  │  ├─ explain (LLM)          ║
         ║  │  └─ decide_action          ║
         ║  │     ├─ low → comment       ║
         ║  │     └─ high → escalate     ║
         ╚═▼════════════════════════════╝
                         │
         ┌───────────────▼───────────────┐
         │  FastAPI Response             │
         │  + GitHub PR Comment (optional)│
         └───────────────────────────────┘
```

**Key Design Principle:** The ML model's score is *never* overridden by the LLM. The LLM's job is to *explain* a decision already made by a trained classifier. This is defensive and auditable.

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Data Mining | `pydriller` | Walk git history, extract commits/diffs/metadata |
| Labeling | SZZ Algorithm | Identify which commits *introduced* bugs (via git blame) |
| Storage | SQLite | Persist mined data; no external dependencies |
| Features | `radon`, `pandas` | Static analysis + feature engineering |
| Classical ML | `xgboost` | Imbalanced classification with feature importance |
| Explainability | `shap` | Per-prediction explanations (tree SHAP) |
| Embeddings | `sentence-transformers` | Free, local embeddings (no API) |
| Vector DB | `faiss` | CPU-only similarity search |
| Orchestration | `langgraph` | Agentic workflows with cycles |
| LLM | Gemini API (free tier) | Lightweight, cost-free alternative to GPT |
| API | `fastapi` | Modern, high-performance REST service |
| Testing | `pytest` | Unit + integration tests |

---

## Project Structure

```
bug-risk-copilot/
├── data/
│   ├── mining.py              # Phase 1: Extract commits
│   ├── szz_labeling.py        # Phase 1: Label bug-introducing commits
│   └── commits.db             # Mined commit data (SQLite)
│
├── features/
│   ├── build_features.py      # Phase 2: Feature engineering
│   └── features.csv           # Output: Feature matrix
│
├── embeddings/
│   └── build_faiss_index.py   # Phase 2: Vector index
│
├── models/
│   ├── train_model.py         # Phase 3: Baseline + XGBoost
│   ├── explain_predictions.py # Phase 3: SHAP analysis
│   └── artifacts/             # Saved model.pkl, faiss.index
│
├── agent/
│   └── agent.py               # Phase 4: LangGraph orchestration
│
├── api/
│   └── main.py                # Phase 5: FastAPI service
│
├── tests/
│   └── test_pipeline.py       # Phase 5: pytest unit tests
│
├── .github/workflows/
│   └── pr-risk-check.yml      # Stretch: GitHub Actions integration
│
└── README.md (this file)
```

---

## Implementation Phases

### Phase 1: Data Collection (Days 1–2)

**Goal:** Mine a real repo and label bug-introducing commits.

**Steps:**
1. Pick a mid-size, active Python repo (e.g., `pallets/flask`, 3000+ commits)
2. Mine commits with `pydriller.Repository()` → SQLite
3. Detect fix-commits via regex: `\b(fix|bug|resolve|closes)\b`
4. Run SZZ algorithm: for each fix-commit, use `git blame` to find which earlier commit introduced the bug
5. Store result: `is_bug_introducing = 1` or `0` label per commit

**Challenges & How We Handle Them:**

- **False positives in SZZ:** Blame sometimes points to whitespace-only or comment-only changes. **Solution:** Filter out trivial diffs.
- **Merge commits:** Blame can be misleading. **Solution:** Only blame non-merge commits.
- **Class imbalance:** Typically 5-15% are bug-introducing. **Solution:** Document this; we'll handle it in Phase 3 with `scale_pos_weight`.

**Deliverable:** SQLite database with `commits(hash, author, message, insertions, deletions, is_fix_commit, is_bug_introducing)`.

**Interview talking points:**
- Explain why we use SZZ (the standard algorithm in this space).
- What are known SZZ weaknesses? (Merge commits, whitespace, refactoring.)
- Why not just use issue tracker? (GitHub Issues API has rate limits, regex is faster for MVP.)

---

### Phase 2: Feature Engineering (Day 3)

**Goal:** Extract 15+ features capturing bug risk signals.

**Features extracted:**

| Category | Features |
|----------|----------|
| **Diffusion** | num_files, num_directories, num_subsystems |
| **Size** | lines_added, lines_deleted, total_churn, churn_ratio |
| **Complexity** | max_complexity_delta, avg_complexity_delta (via `radon`) |
| **Historical** | file_age_days, file_prior_bug_count, author_prior_bug_rate, author_commit_count |
| **Message** | message_length, has_issue_reference, hour_of_day, day_of_week |

**Key Detail: Temporal Causality**
- Author's bug rate is computed *only from commits before this one*.
- This prevents *data leakage* — a critical mistake in time-series ML.

**Also:** Embed all bug-introducing commits using `sentence-transformers` and build a FAISS index for later retrieval.

**Deliverable:** `features.csv` (one row per commit, ~15 columns) + `bug_commits.index` (FAISS) + metadata.

**Interview talking points:**
- Why is temporal split critical? (Leakage.)
- Which feature do you expect to be strongest? (File prior bug count — commits in "risky files" are more likely to be buggy.)
- How would you handle missing features? (Imputation strategy: median, domain knowledge, etc.)

---

### Phase 3: Model Training & Evaluation (Days 4–5)

**Goal:** Train a baseline and production model, evaluate rigorously.

**Baseline: LogisticRegression**
- Standardize features
- Use `class_weight='balanced'` for imbalance
- Evaluate on time-based test set

**Production: XGBoost**
- Time-based train/test split (DO NOT use random split)
- Tune `scale_pos_weight` for imbalance
- Light hyperparameter search with `TimeSeriesSplit`
- Evaluate with **PR-AUC** (not ROC-AUC)

**Why PR-AUC?**
- With 5-15% positive class, ROC-AUC is misleading (always ~0.9+).
- PR-AUC is more honest: it emphasizes precision/recall trade-off.
- Pick a threshold based on cost reasoning: false positive costs review time; false negative lets bugs ship.

**SHAP Explainability**
- Run `shap.TreeExplainer` on trained XGBoost
- Identify top-5 features by importance
- Generate individual prediction explanations

**Expected Finding:** `file_prior_bug_count` and `total_churn` dominate SHAP importance. This makes intuitive sense: files with prior bugs are risky, and large changes are risky.

**Deliverable:** Trained model artifact (`.pkl`), evaluation report (PR curve, metrics, SHAP plots), chosen operating threshold.

**Interview talking points:**
- Why time-based split, not random? (Leakage of future into training.)
- Why PR-AUC, not accuracy? (Imbalanced data; accuracy is misleading.)
- What would you change with 10x more data? (Could use deep learning; currently, XGBoost is best for imbalanced tabular data.)

---

### Phase 4: LangGraph Agent (Day 6)

**Goal:** Build a multi-step agentic workflow that investigates and explains.

**Graph Structure:**
```
fetch_context → investigate → explain → decide_action
                     ↑____________↓
                  (optional loop)
```

**Nodes:**

1. **`fetch_context`:** Retrieve model prediction and SHAP features.
2. **`investigate`:** Embed the diff, query FAISS for top-3 similar past bug commits. If similarity is weak, loop back for a broader search.
3. **`explain`:** Construct a prompt combining risk score, SHAP features, and retrieved commits. Call Gemini LLM to generate plain-English explanation.
4. **`decide_action`:** Check risk score against threshold:
   - Below threshold → `action = "comment"` (informational)
   - Above threshold → `action = "escalate"` (tag reviewer)

**Key Design:** The LLM never *decides* risk; it only *explains* a score already computed by the model.

**Deliverable:** Working LangGraph agent that takes `(commit_hash, diff_text, risk_score, top_features)` and returns `(explanation, action)`.

**Interview talking points:**
- Why use a graph (LangGraph) instead of a linear chain? (Handles conditional logic and loops cleanly.)
- Why embed diffs instead of keyword search? (Semantic similarity catches conceptually related bugs even with different keywords.)
- What happens if FAISS search returns weak matches? (Agent can loop for broader context — shows agentic reasoning, not just prompt chaining.)

---

### Phase 5: Serving & Integration (Day 7)

**Goal:** Expose the system as a REST API and optionally integrate with GitHub.

**FastAPI Service:**

Endpoints:
- `POST /predict` — ML model prediction (risk score + SHAP features)
- `POST /explain` — Agent explanation (calls LangGraph)
- `POST /analyze` — End-to-end (predict + explain)
- `GET /health` — Health check

**Pydantic Models:** Clean request/response schemas for type safety and documentation.

**Testing with pytest:**
- Unit tests for each component (mining, features, agent nodes)
- Mock external calls (LLM, FAISS) to keep tests fast
- Integration test for the full pipeline

**Stretch Goal: GitHub Actions**
- Workflow triggers on PR
- Extracts diff, calls `/analyze` endpoint
- Posts comment on PR with risk assessment and recommendation

**Deliverable:** Deployed FastAPI service (or locally runnable), pytest coverage, optionally wired GitHub Action.

**Interview talking points:**
- How would you handle model versioning? (Model registry, CI/CD pipeline.)
- What's the latency budget? (Most code reviews take 10+ minutes; model + agent should finish in <5 seconds.)
- How would you monitor for model drift? (Retrain periodically, track precision/recall on new data.)

---

## Key Interview Prep: Talking Points

### 1. Why This Project is Strong

**Narrative:** "This isn't just a chatbot with an LLM. It's a real system where the ML model makes the decision and the LLM explains it. That separation is production-grade."

### 2. Class Imbalance (Critical!)

**Question:** "How do you handle the fact that only 5-15% of commits are bugs?"

**Answer:**
- Use `scale_pos_weight` in XGBoost to penalize false negatives.
- Evaluate with PR-AUC, not accuracy (which is 85%+ just from the majority class).
- Choose operating threshold based on cost: if false positive is cheap (just a comment), optimize for recall.

### 3. Data Leakage

**Question:** "How do you avoid leaking future data into training?"

**Answer:**
- Time-based split: earliest 80% for training, most recent 20% for test.
- Never shuffle commits — that breaks causality.
- Author's bug rate computed only from *prior* commits.

### 4. SZZ Algorithm

**Question:** "What's a known weakness of SZZ?"

**Answer:**
- Blame can point to whitespace-only or comment-only changes.
- Merge commits confuse blame.
- We mitigate by filtering trivial diffs and non-merge commits.

**Alternative answer:** "For production, we'd cross-reference with issue tracker + code review comments to improve precision."

### 5. Why Separate ML from LLM?

**Question:** "Why not just have the LLM decide if it's risky?"

**Answer:**
- LLMs are unpredictable; you can't audit or explain decisions.
- ML model is deterministic and trained on historical data.
- LLM adds context *given* the model's score, not overriding it.
- This is what production systems do (e.g., fraud detection at banks).

### 6. FAISS vs. Other Vector DBs

**Question:** "Why FAISS instead of Pinecone or Weaviate?"

**Answer:**
- FAISS is CPU-only, free, and local — no external dependencies or API costs.
- For a portfolio project, this demonstrates systems thinking: minimal moving parts.
- In production, you'd evaluate trade-offs: latency, scale, availability.

### 7. What Would You Change?

**Question:** "With infinite compute/data, what would you do differently?"

**Answer:**
- Collect more data (aim for 10k+ commits with labels).
- Try deep learning: transformer-based model on raw diffs (RoBERTa fine-tuned).
- Add online learning: retrain weekly on new bugs found in production.
- Build a feature store (Feast, Tecton) for scalable feature serving.
- Add causal inference: understand which features actually cause bugs (vs. correlate).

---

## Running the Project

### Setup

```bash
# Clone and navigate
cd bug-risk-copilot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Phase 1: Data Collection

```bash
# Mine a repository (example: Flask)
python data/mining.py --repo /path/to/flask --db data/commits.db

# Label bug-introducing commits (requires git repo access)
python data/szz_labeling.py --repo /path/to/flask --db data/commits.db
```

### Phase 2: Features & Embeddings

```bash
# Extract features
python features/build_features.py --db data/commits.db --output data/features.csv

# Build FAISS index
python embeddings/build_faiss_index.py --db data/commits.db --output models/artifacts
```

### Phase 3: Training

```bash
# Train models
python models/train_model.py --features data/features.csv --model-dir models/artifacts

# Generate SHAP explanations
python models/explain_predictions.py --model models/artifacts/xgboost_model.pkl --features data/features.csv
```

### Phase 4: Test Agent

```bash
python agent/agent.py
```

### Phase 5: Start API & Test

```bash
# Start API server
uvicorn api.main:app --reload

# Test in another terminal
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "diff": "+(risky change here)",
    "commit_hash": "abc123"
  }'
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Expected Outputs

### Phase 1
- SQLite database with 3000+ commits labeled as bug-introducing or not
- Example: 5-15% are bugs, 85-95% are clean

### Phase 2
- CSV with 15+ features per commit
- FAISS index with embeddings of all bug commits

### Phase 3
- XGBoost model achieving PR-AUC ~0.65-0.75 (depending on repo)
- SHAP analysis showing top predictive features
- Example: "file_prior_bug_count" is the strongest signal

### Phase 4
- LangGraph agent that generates explanations like:
  > "Flagged high-risk (0.82). This commit touches 8 files with no accompanying tests, and closely resembles commit a3f21 (Jan 2025), which modified the same authentication module and later caused production incident."

### Phase 5
- FastAPI service with `/predict`, `/explain`, `/analyze` endpoints
- Pytest tests passing
- GitHub Action commenting on PRs (optional)

---

## Limitations & Future Work

1. **Limited training data:** Start with 3000 commits; in production, aim for 10k+.
2. **SZZ precision:** Our regex-based fix-detection is ~80% accurate; cross-reference with issues for higher precision.
3. **Feature engineering:** Currently hand-crafted; in production, consider automated feature discovery (e.g., AutoML).
4. **LLM cost:** Gemini free tier has rate limits; in production, cache explanations or use cheaper models.
5. **Real-time serving:** FastAPI is single-instance; use load balancer + model serving infra (Ray, KServe) for high scale.

---

## Repo Structure Summary

| File | Purpose | Key Learnings |
|------|---------|---------------|
| `data/mining.py` | Commit extraction | `pydriller`, database schema |
| `data/szz_labeling.py` | Bug-introducing labels | SZZ algorithm, git blame, temporal causality |
| `features/build_features.py` | Feature engineering | Imbalanced data, feature importance |
| `embeddings/build_faiss_index.py` | Vector search index | Embeddings, similarity search |
| `models/train_model.py` | ML model training | Time-based split, class imbalance, PR-AUC |
| `models/explain_predictions.py` | SHAP explanations | Tree SHAP, feature importance |
| `agent/agent.py` | Agentic orchestration | LangGraph, multi-step reasoning, separation of concerns |
| `api/main.py` | REST API | FastAPI, Pydantic, production patterns |
| `tests/test_pipeline.py` | Unit + integration tests | pytest, mocking, edge cases |
| `.github/workflows/pr-risk-check.yml` | CI/CD integration | GitHub Actions, automation |

---

## Interview Narrative (Elevator Pitch)

> "I built a bug-risk prediction system that combines classical ML, vector search, and agentic reasoning. The core innovation is the separation: a trained XGBoost model predicts risk (0-1 score), then a LangGraph agent investigates and explains that decision using SHAP features and historical context. It's not a chatbot—it's a real system where the ML makes decisions and the LLM adds transparency.
>
> The technical challenges I solved: handling class imbalance in the training data, avoiding temporal leakage in time-series ML, implementing the SZZ algorithm to label historical bugs, building a FAISS index for semantic search over past commits, and orchestrating multi-step reasoning with conditional loops in LangGraph.
>
> It integrates as a REST API and can comment on GitHub PRs, turning it from a research project into a tool that actually touches real workflows."

---

## References & Resources

- **SZZ Algorithm:** [When Do Changes Induce Fixes?](https://dl.acm.org/doi/abs/10.1145/1083142.1083155)
- **PR-AUC:** [The precision-recall trade-off](https://en.wikipedia.org/wiki/Precision_and_recall)
- **SHAP:** [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- **LangGraph:** [LangChain Docs](https://python.langchain.com/docs/langgraph)
- **FAISS:** [Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)

---

**Author:** Built as a full-stack AI engineering portfolio piece.

**Last Updated:** 2026-08-26
