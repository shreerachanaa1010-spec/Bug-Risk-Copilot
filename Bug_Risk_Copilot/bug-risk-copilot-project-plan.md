# Bug-Risk Copilot — Complete End-to-End Project Plan

**Goal:** A system that predicts whether a git commit is likely to introduce a bug (real ML), retrieves similar historical bug-causing commits (vector search / RAG), and reasons through the evidence with an agentic workflow to produce a human-readable risk explanation and routing decision (LangGraph).

**Why this project:** It's the single most complete portfolio piece for a FAANG/MAANG AI engineering interview — it proves four distinct skills in one coherent system: classical ML (training + evaluation), vector search/RAG, agentic orchestration, and production engineering. Nothing about it is a thin LLM wrapper.

---

## 1. Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Data mining | `pydriller` | Walk git history, extract commits/diffs/metadata |
| Storage | SQLite (or Parquet via `pandas`) | Cache mined commit data locally |
| Static analysis | `radon` | Cyclomatic complexity metrics |
| Data wrangling | `pandas`, `numpy` | Feature engineering |
| Classical ML | `scikit-learn` (LogisticRegression baseline) | Baseline classifier |
| Classical ML | `xgboost` or `lightgbm` | Main classifier |
| Explainability | `shap` | Feature importance per prediction |
| Embeddings | `sentence-transformers` (e.g. `all-MiniLM-L6-v2`) | Embed commit diffs/messages |
| Vector DB | `faiss` (CPU) | Similarity search over embedded past bug commits |
| Agent orchestration | `langgraph` | Multi-step, conditional agent graph |
| LLM calls | `langchain` + Gemini API (free tier, reuse Sentinel setup) | Explanation generation node |
| API layer | `FastAPI` | Serve predictions + agent output |
| CI/CD (stretch) | GitHub Actions | Auto-comment on real PRs |
| Testing | `pytest` | Unit tests for pipeline components |
| Experiment tracking (optional) | `mlflow` or simple CSV logging | Track model versions/metrics |

---

## 2. Architecture Overview

```
                    ┌─────────────────────────┐
                    │   New Commit / PR Diff   │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Feature Extraction        │
                    │  (churn, complexity,       │
                    │   author/file history)     │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  ML MODEL (XGBoost)        │───► risk_score (0-1)
                    │  trained offline           │      + SHAP top features
                    └────────────┬─────────────┘
                                 │
                    ╔════════════▼═════════════╗
                    ║      LANGGRAPH AGENT       ║
                    ║                            ║
                    ║  [fetch_context]           ║
                    ║        │                   ║
                    ║  [investigate] ◄──┐         ║   FAISS similarity search
                    ║        │          │loop     ║   over past bug commits
                    ║        │ (if unclear)        ║
                    ║        ▼──────────┘         ║
                    ║  [explain]                  ║
                    ║        │                    ║
                    ║  [decide_action]            ║
                    ║   ├── low/med → comment      ║
                    ║   └── high → escalate        ║
                    ╚════════════┬═══════════════╝
                                 │
                    ┌────────────▼─────────────┐
                    │   FastAPI response /       │
                    │   GitHub PR comment        │
                    └───────────────────────────┘
```

**Key design principle:** the ML model makes the *prediction*. The agent only *investigates and explains* — it never overrides the model's score. This separation is what you defend in interviews: "the LLM doesn't decide risk, it explains a decision already made by a trained classifier."

---

## 3. Detailed Step-by-Step Plan

### Phase 1 — Data Collection (Days 1–2)

**Step 1.1 — Pick target repository**
- Use a mid-size, active, well-documented public Python repo (e.g. `pallets/flask`, `psf/requests`, or `scikit-learn/scikit-learn`).
- Needs: 3,000+ commits, active issue tracker, consistent commit message conventions (for fix-detection regex to work well).

**Step 1.2 — Mine commit history**
- Use `pydriller.Repository(path).traverse_commits()` to iterate every commit.
- For each commit, extract: hash, author, author date, committer date, files modified, per-file diff, added/deleted line counts, commit message, parent hash(es).
- Persist to SQLite immediately (mining is slow — don't re-run from scratch each time). Table: `commits(hash, author, date, message, files_json, insertions, deletions)`.

**Step 1.3 — Identify fix-commits**
- Regex match commit messages against fix-indicating patterns: `\b(fix|fixes|fixed|bug|bugfix|resolve|resolves|closes)\b`, plus issue-reference patterns (`#\d+`).
- Cross-reference with the repo's issue tracker via GitHub API if you want higher precision (optional — regex alone is a reasonable v1).
- Store as `is_fix_commit` flag.

**Step 1.4 — SZZ labeling (bug-introducing commit identification)**
- For every fix-commit, extract the exact lines the fix changed (from the diff).
- Run `git blame` (via `pydriller`'s `modified_files[i].diff_parsed` or `GitPython`) on those lines **as they existed just before the fix** to find which earlier commit last touched them.
- That earlier commit gets labeled `is_bug_introducing = 1`.
- Filter out trivial SZZ false positives: ignore blame results pointing to whitespace-only changes, comment-only changes, or the fix-commit's own initial commit (cosmetic SZZ noise reduction — mention this in interviews, it shows you know SZZ's known weaknesses).
- All other commits default to `is_bug_introducing = 0`.
- **Expect class imbalance** — typically 5–15% positive. Record the exact ratio; you'll need it for Phase 3.

**Deliverable:** a labeled dataset — one row per commit with `is_bug_introducing` target.

---

### Phase 2 — Feature Engineering (Day 3)

**Step 2.1 — Diffusion features**
- `num_files_changed`, `num_directories_touched`, `num_subsystems_touched` (top-level package boundaries).

**Step 2.2 — Size features**
- `lines_added`, `lines_deleted`, `total_churn = added + deleted`.

**Step 2.3 — Complexity features**
- Run `radon.complexity.cc_visit` on the file's content before and after the commit; compute `complexity_delta` per file, then aggregate (sum/max) across files in the commit.

**Step 2.4 — Historical/risk features**
- `file_age_days` (time since file's first commit).
- `file_prior_bug_count` (how many times this file has previously been marked bug-introducing — this is your strongest signal, expect it to dominate SHAP importance).
- `author_prior_bug_rate` (this author's historical bug-introduction rate, computed only from commits *before* the current one — critical to avoid leakage).
- `author_commit_count` (experience proxy).

**Step 2.5 — Message/metadata features**
- `message_length`, `has_issue_reference` (bool), `commit_hour` (time-of-day — late-night commits sometimes correlate with risk, worth testing), `day_of_week`.

**Step 2.6 — Embed diffs for FAISS (parallel track)**
- For every commit labeled `is_bug_introducing = 1`, embed the **diff text + commit message** using `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim, free/local — no API cost).
- Build a FAISS index (`IndexFlatL2` or `IndexIVFFlat` if the dataset is large) over these embeddings.
- Store a parallel metadata list (commit hash → original diff snippet, date, file) so retrieval results can be mapped back to human-readable context.
- Persist the FAISS index to disk (`faiss.write_index`).

**Deliverable:** a clean `pandas` DataFrame (one row per commit, ~15 features) ready for modeling, plus a persisted FAISS index of historical bug-commit embeddings.

---

### Phase 3 — Model Training & Evaluation (Days 4–5)

**Step 3.1 — Time-based train/test split**
- Sort commits chronologically. Use the earliest ~80% for training, most recent ~20% for testing.
- **Never use a random split** — it leaks future information into training and is a common mistake interviewers specifically probe for. Be ready to explain this choice unprompted.

**Step 3.2 — Baseline model**
- `LogisticRegression(class_weight='balanced')` from `scikit-learn`.
- Standardize numeric features first (`StandardScaler`).
- Evaluate: precision, recall, F1, and **PR-AUC** (not ROC-AUC — with 5–15% positive class, ROC-AUC is misleadingly optimistic; PR-AUC reflects real performance). Know this cold for interviews.

**Step 3.3 — Main model**
- `XGBClassifier` (or `LightGBM`), tuned via `scale_pos_weight` to handle imbalance.
- Light hyperparameter search (`RandomizedSearchCV` with time-series-aware CV via `TimeSeriesSplit` — not standard k-fold, same leakage reasoning as 3.1).
- Compare PR-AUC against baseline; plot both precision-recall curves on one chart.

**Step 3.4 — Explainability**
- Run `shap.TreeExplainer` on the trained XGBoost model.
- Generate a global feature-importance summary plot and a few individual-prediction waterfall plots.
- Expect `file_prior_bug_count` and `total_churn` to dominate — this becomes your headline finding.

**Step 3.5 — Threshold selection**
- Don't just use 0.5. Plot precision/recall at multiple thresholds and pick one deliberately, justified by cost reasoning: a false positive wastes reviewer time; a false negative lets a bug ship. State your assumption (e.g., "I optimized for recall ≥ 0.6 while keeping precision ≥ 0.4, because in this workflow a flagged commit only adds a review comment, not a blocked merge — so false positives are cheap").

**Deliverable:** trained model artifact (`model.pkl` or `model.json` for XGBoost), evaluation report (metrics + PR curve + SHAP plots), chosen operating threshold.

---

### Phase 4 — LangGraph Agent (Day 6)

**Step 4.1 — Define state schema**
```python
class AgentState(TypedDict):
    commit_hash: str
    diff_text: str
    risk_score: float
    shap_top_features: list[dict]
    similar_commits: list[dict]   # filled by investigate node
    needs_more_context: bool
    explanation: str
    action: str                   # "comment" | "escalate"
```

**Step 4.2 — Node: `fetch_context`**
- Pulls the commit's diff, the model's `risk_score`, and top SHAP features from Phase 3 outputs.

**Step 4.3 — Node: `investigate`**
- Embeds the current diff with the same `sentence-transformers` model used in Phase 2.
- Queries the FAISS index for top-3 nearest historical bug-introducing commits.
- Populates `similar_commits` with hash, date, file, and a short diff snippet for each.
- Sets `needs_more_context = True` if similarity scores are weak/ambiguous (e.g. all distances above a threshold), which routes back through a second, broader FAISS query (e.g. relaxing to file-level rather than diff-level embedding) — this is the conditional loop that specifically demonstrates LangGraph's cyclic-graph capability over a plain LangChain chain.

**Step 4.4 — Node: `explain`**
- Constructs a prompt combining: risk score, top SHAP features, and the retrieved similar commits.
- Calls the LLM (Gemini via `langchain`) to generate a 2–3 sentence, plain-English explanation.
- Example target output: *"Flagged high-risk (0.82). This commit touches 8 files with no accompanying tests, and closely resembles commit a3f21 (Jan 2025), which modified the same authentication module and later caused a production incident."*

**Step 4.5 — Node: `decide_action`**
- Conditional edge based on `risk_score` against the Phase 3 threshold:
  - Below threshold → `action = "comment"` (informational PR comment only)
  - Above threshold → `action = "escalate"` (comment + tag a human reviewer)

**Step 4.6 — Wire the graph**
```python
graph = StateGraph(AgentState)
graph.add_node("fetch_context", fetch_context)
graph.add_node("investigate", investigate)
graph.add_node("explain", explain)
graph.add_node("decide_action", decide_action)

graph.set_entry_point("fetch_context")
graph.add_edge("fetch_context", "investigate")
graph.add_conditional_edges(
    "investigate",
    lambda state: "investigate" if state["needs_more_context"] else "explain"
)
graph.add_edge("explain", "decide_action")
graph.add_edge("decide_action", END)

app = graph.compile()
```

**Deliverable:** a working, testable LangGraph agent that takes a `commit_hash` + pre-computed `risk_score`/SHAP output and returns an explanation + routing decision.

---

### Phase 5 — Serving & Integration (Day 7)

**Step 5.1 — FastAPI service**
- `POST /predict` — accepts a diff or commit hash, runs feature extraction + XGBoost model, returns `risk_score` + SHAP features.
- `POST /explain` — accepts the above output, runs it through the LangGraph agent, returns explanation + action.
- `POST /analyze` — convenience endpoint that chains both for a single commit end-to-end.

**Step 5.2 — Input validation**
- Use `pydantic` models for request/response schemas (clean FastAPI practice, worth mentioning in interviews as good API design).

**Step 5.3 — Stretch: GitHub Action integration**
- A workflow that triggers on `pull_request`, extracts the diff, calls your deployed `/analyze` endpoint, and posts the returned explanation as a PR comment via the GitHub API.
- This turns the project from "a model in a notebook" into "a tool that touches a real workflow" — the single biggest credibility jump for a portfolio piece.

**Step 5.4 — Testing**
- `pytest` unit tests for: feature extraction correctness (given a mock diff, do you get expected churn/complexity numbers), SZZ labeling edge cases, and a mocked LangGraph run (mock the LLM call to keep tests fast/free).

**Deliverable:** deployed (or locally runnable) FastAPI service, optionally wired into a real GitHub Action, with test coverage.

---

## 4. Evaluation & Presentation Checklist (for interviews)

Be ready to answer, in one or two sentences each:
- Why time-based split, not random?
- Why PR-AUC, not accuracy or ROC-AUC?
- What does SHAP tell you was the strongest predictor, and does that make intuitive sense?
- Why did you separate the ML model (decision) from the LLM (explanation)? What would go wrong if the LLM also decided the risk score?
- Why FAISS/embeddings instead of just keyword search over past commits?
- What's a known weakness of the SZZ algorithm, and how did you mitigate it?
- What would you change with 10x more data or compute?

---

## 5. Suggested Repo Structure

```
bug-risk-copilot/
├── data/
│   ├── mining.py          # pydriller extraction
│   ├── szz_labeling.py
│   └── commits.db
├── features/
│   └── build_features.py
├── models/
│   ├── train_baseline.py
│   ├── train_xgboost.py
│   ├── evaluate.py
│   └── artifacts/          # saved model.pkl, faiss.index
├── embeddings/
│   └── build_faiss_index.py
├── agent/
│   ├── state.py
│   ├── nodes.py
│   └── graph.py
├── api/
│   └── main.py             # FastAPI app
├── .github/workflows/
│   └── pr-risk-check.yml   # stretch goal
├── tests/
└── README.md                # write this last — it's your interview pitch
```

---

## 6. Timeline Summary

| Day | Deliverable |
|---|---|
| 1 | Raw commit dataset mined and cached |
| 2 | SZZ-labeled dataset with bug-introducing flags |
| 3 | Feature-engineered DataFrame + FAISS index of past bug commits |
| 4 | Baseline LogisticRegression model + eval report |
| 5 | XGBoost model + SHAP analysis + chosen threshold |
| 6 | Working LangGraph agent (fetch → investigate → explain → decide) |
| 7 | FastAPI service + tests + optional GitHub Action |

Start with Day 1 whenever you're ready — the `pydriller` mining script is the natural first piece of code to write.
