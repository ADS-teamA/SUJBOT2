# RAG Confidence Score – Training & Tuning Setup

This document describes how to **develop, train, and tune** a RAG confidence model using the six regressors defined in `rag_confidence_regressors.md`. It is written to be machine- and human-friendly, so it can later be used as context for a coding agent that implements the full pipeline.

---

## 1. Goal

We want a function that, given a **query** and its **retrieval results**, returns a scalar confidence score that approximates:

> the probability that the **relevant context** is present in the top-\(K\) results

Concretely, we treat **recall@10** as the ground-truth label and train a regression model that predicts this value from cheap retrieval-side features (the six regressors).

At runtime this becomes the `overall_confidence` used in `RAGConfidenceScore` and mapped to HIGH / MEDIUM / LOW confidence bands.

---

## 2. Available Data

From the existing `dataset.json` we assume the following structure (per query):

- `query`: the query string
- `relevant_chunk_ids`: list of chunk IDs judged as relevant (for training only)
- `candidates`: list of up to 100 candidate chunks, each with:
  - `id`: chunk ID (e.g. `BZ_VR1_L3_sec_60_chunk_1`)
  - `retriever_score`: raw dense retriever score
  - `retriever_score_norm`: normalized score in [0, 1] per query
  - `rank`: rank of this candidate (1 = best)

From this we can compute both labels and the six regressors.

---

## 3. Label Definition (Target Variable)

For each query, we define the training label as **recall@10** based on `relevant_chunk_ids` and the top-10 candidates.

### 3.1 Top-10 candidate set

Let:

- `C_10` = the list of top-10 candidates, sorted by ascending `rank`
- `top10_ids = {c["id"] for c in C_10}`

### 3.2 Relevant IDs

Let:

- `R = set(relevant_chunk_ids)`

### 3.3 Recall@10

The per-query label is:

\[
\text{recall@10} = 
\begin{cases}
0 & \text{if } |R| = 0 \\
\frac{|R \cap \text{top10\_ids}|}{|R|} & \text{otherwise}
\end{cases}
\]

This value lies in \([0, 1]\) and is used as the **regression target**.

> Note: for queries with no labeled relevant chunks (|R| = 0), you can either drop them from training or treat recall@10 as 0.0. Dropping them is usually safer.

---

## 4. Feature Set (Regressors)

We use the **top 6 regressors** from `rag_confidence_regressors.md`. Each is computed from the **query text**, **IDs**, and **scores** of the top-\(K\) candidates (we will use K=10 consistently).

Let, for the top-10 candidates:

- `C_10` = candidates sorted by `rank`
- `s_i` = `retriever_score_norm` of candidate i (i = 1..10)
- `S_i` = raw `retriever_score` of candidate i (if needed)
- `id_i` = chunk ID string of candidate i

Let:

- `K = 10`

### 4.1 Regressor 1: Normalized Score Standard Deviation (σ_10)

Captures how spread out the normalized scores are in the top-10.

\[
\mu_{10} = \frac{1}{10} \sum_{i=1}^{10} s_i
\]

\[
\sigma_{10} = \sqrt{\frac{1}{10} \sum_{i=1}^{10} (s_i - \mu_{10})^2}
\]

**Feature name:** `std_norm_top10 = σ_10`

---

### 4.2 Regressor 2: Normalized Score Entropy (H_10)

Converts scores into a probability distribution and measures entropy.

\[
p_i = \frac{s_i}{\sum_{j=1}^{10} s_j + \varepsilon}
\]

\[
H_{10} = - \sum_{i=1}^{10} p_i \log p_i
\]

**Feature name:** `entropy_norm_top10 = H_10`

(Using natural log is fine; base choice is irrelevant for regression.)

---

### 4.3 Regressor 3: Score Slope vs Rank (b_10)

Fits a straight line for normalized score as a function of rank.

Let ranks be `x_i = i` and scores `y_i = s_i`. Then:

\[
\bar{x} = \frac{1}{10} \sum_{i=1}^{10} x_i
\quad , \quad
\bar{y} = \frac{1}{10} \sum_{i=1}^{10} y_i
\]

\[
b_{10} = \frac{\sum_{i=1}^{10} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{10} (x_i - \bar{x})^2}
\]

**Feature name:** `slope_norm_top10 = b_10`

Typically `b_10` is negative; more negative often indicates a strong top hit and quickly decreasing scores.

---

### 4.4 Regressor 4: Top-vs-Rest Normalized Score Ratio (TV_10)

Measures how much stronger the top result is than the rest of the top-10.

Let:

\[
\text{mean\_rest}_{10} = \frac{1}{9} \sum_{i=2}^{10} s_i
\]

\[
\text{TV}_{10} = \frac{s_1}{\text{mean\_rest}_{10} + \varepsilon}
\]

**Feature name:** `top_vs_rest_ratio_top10 = TV_10`

---

### 4.5 Regressor 5: Section Diversity in Top-10 (div_10)

Chunk IDs look like:

```text
BZ_VR1_L3_sec_60_chunk_1
```

We treat everything before `_chunk_` as a **section ID**:

```python
section_id = id.split("_chunk_")[0]
```

For the top-10 candidates, let:

- `section_i` = section ID of candidate i
- `U_10` = set of distinct section IDs among `section_1..section_10`

Then:

\[
M_{10} = |U_{10}|
\quad , \quad
\text{div}_{10} = \frac{M_{10}}{10}
\]

**Feature name:** `section_diversity_top10 = div_10`

---

### 4.6 Regressor 6: Query Token Length (L_q)

Given the query string `q`, tokenize by whitespace:

```python
tokens = q.split()
```

Let:

\[
L_q = |\text{tokens}|
\]

**Feature name:** `query_token_len = L_q`

---

## 5. Feature Extraction Pipeline

We define a small, self-contained feature extraction module that:

1. Accepts a **single query entry** from the dataset (or from live retrieval).
2. Extracts the top-10 candidates.
3. Computes the six regressors above.
4. Returns a fixed-length feature vector `X`.

### 5.1 Suggested interface

Python-style:

```python
@dataclass
class RetrievalCandidate:
    id: str
    retriever_score: float
    retriever_score_norm: float
    rank: int

@dataclass
class QueryRetrievalExample:
    query: str
    candidates: list[RetrievalCandidate]
    relevant_chunk_ids: list[str] | None  # used only in training
```

Feature extractor:

```python
class ConfidenceFeatureExtractor:
    def __init__(self, k: int = 10, eps: float = 1e-12):
        self.k = k
        self.eps = eps

    def extract_features(self, example: QueryRetrievalExample) -> dict[str, float]:
        # returns a dict mapping feature_name -> value
        ...
```

The coding agent should implement `extract_features` according to the formulas in sections 4.1–4.6.

---

## 6. Model Training Pipeline

We use the dataset to train a regression model that maps features `X` to recall@10 labels.

### 6.1 Data preparation steps

1. **Load dataset**
   - Read `dataset.json` into memory.

2. **Filter queries**
   - Optionally drop queries where `relevant_chunk_ids` is empty.

3. **Compute labels**
   - For each query:
     - Build `QueryRetrievalExample` from its `query` and `candidates`.
     - Compute `recall_at_10` as in section 3.

4. **Compute features**
   - Use `ConfidenceFeatureExtractor` to obtain a dict of feature values.
   - Convert to a numeric vector in a consistent feature order.

5. **Split into sets**
   - Train / validation / test split, e.g. 70 / 15 / 15 by query.

### 6.2 Model choice

We start with a simple **regularized linear regression**:

- Ridge regression (L2) or Elastic Net (L1 + L2).
- Input: feature matrix `X` (n_queries × 6).
- Output: predicted `recall_at_10` in ℝ (we can clip to [0,1] afterward).

Rationale:

- It’s fast, interpretable, and works well with small feature sets.
- Coefficients can be inspected; non-informative features will end up with near-zero weights.

Alternative models (optional future work):

- Gradient-boosted trees (e.g. XGBoost, LightGBM) for non-linear patterns.
- Beta regression if strict [0,1] outputs are desired.

### 6.3 Hyperparameter tuning

On the validation set:

- Perform grid-search or random-search over regularization strength (e.g. alpha for Ridge).
- Evaluate using metrics such as:
  - Pearson correlation between predicted and true recall@10.
  - Mean squared error (MSE).
  - Calibration plots (predicted vs actual recall@10 binned by score).

Pick the model with best validation performance and good calibration.

---

## 7. Confidence Thresholds & Interpretation

Once we have a model that outputs predicted recall@10, we map it to discrete confidence bands.

### 7.1 Example thresholds

For example, using validation data to decide cutoffs:

- HIGH confidence: `pred_recall >= 0.9`
- MEDIUM confidence: `0.75 <= pred_recall < 0.9`
- LOW confidence: `0.5 <= pred_recall < 0.75`
- VERY LOW confidence: `pred_recall < 0.5`

These ranges can be tuned based on:

- Desired trade-off between automation and human review.
- Empirical error rates per bin (e.g., what fraction of queries in the HIGH band actually had recall@10 ≥ 0.8).

### 7.2 Mapping into RAGConfidenceScore

The runtime scoring function should return a structure like:

```python
@dataclass
class RAGConfidenceScore:
    overall_confidence: float      # predicted recall@10, clipped to [0,1]
    interpretation: str            # "HIGH", "MEDIUM", "LOW", "VERY_LOW"
    should_flag: bool              # e.g. True if LOW or VERY_LOW
    details: dict[str, float]      # raw features & intermediate metrics
```

- `overall_confidence` comes from the regression model.
- `interpretation` is set via thresholds.
- `should_flag` is driven by business logic (e.g. flag if `overall_confidence < 0.75`).

---

## 8. Runtime Integration Flow

At runtime (serving time), for each incoming query:

1. **Run retrieval**
   - Use current ensemble dense search to obtain candidate list with:
     - `id`, `retriever_score`, `retriever_score_norm`, `rank`.
   - Keep at least top-10 candidates.

2. **Build example**
   - Create a `QueryRetrievalExample` with:
     - `query` = user query string.
     - `candidates` = converted top candidates.
     - `relevant_chunk_ids` = `None` (not available at runtime).

3. **Extract features**
   - Call `ConfidenceFeatureExtractor.extract_features(example)`.

4. **Predict confidence**
   - Feed the feature vector into the trained regression model.
   - Clip predictions to `[0, 1]`:
     - `overall_confidence = max(0.0, min(1.0, prediction))`.

5. **Interpret & attach to response**
   - Convert `overall_confidence` to `interpretation` and `should_flag`.
   - Attach as `rag_confidence` metadata to retrieval results.
   - Optionally, prepend a textual warning (e.g., “⚠️ Low retrieval confidence, review recommended”) if `should_flag` is True.

This preserves low latency, because computing all six regressors is O(K) with `K = 10` and only uses data that is already available from the retriever.

---

## 9. Model & Config Versioning

For maintainability and future automation, we should:

- Store trained model artifacts with explicit versioning, e.g.:
  - `rag_confidence_model_v1.pkl`
- Store configuration in a separate JSON/YAML:
  - feature list and order,
  - K (top-k used),
  - epsilon for numerical stability,
  - thresholds for confidence bands.

Example config snippet:

```yaml
model_version: v1
top_k: 10
epsilon: 1e-12
features:
  - std_norm_top10
  - entropy_norm_top10
  - slope_norm_top10
  - top_vs_rest_ratio_top10
  - section_diversity_top10
  - query_token_len
thresholds:
  high: 0.9
  medium: 0.75
  low: 0.5
```

A coding agent can read this config to ensure consistency between training code and runtime scoring code.

---

## 10. Monitoring & Iteration

After deployment:

- Log, for each query:
  - predicted `overall_confidence`,
  - band (`HIGH` / `MEDIUM` / `LOW` / `VERY_LOW`),
  - whether the answer was later marked correct / incorrect (if available).
- Periodically recompute actual recall-based metrics on held-out queries (e.g., by collecting new labeled data).
- Retrain or retune:
  - thresholds (for confidence bands),
  - regression model (if new features or data are added).

This feedback loop lets the confidence model adapt as retrieval quality, corpus, or user behavior changes.

---

## 11. Summary

The setup is:

1. Use `dataset.json` to compute **recall@10** labels per query.
2. Extract six light-weight features (regressors) from normalized scores, ranks, IDs, and query text.
3. Fit a regularized linear regression model to predict recall@10 from these features.
4. Map predicted recall@10 into interpretable confidence bands at runtime.
5. Integrate the scoring into the RAG pipeline via a `RAGConfidenceScore` object and monitor performance over time.

This document should provide enough structure and detail for a coding agent to implement:
- feature extraction,
- model training,
- model serving,
- configuration and thresholding,
with minimal ambiguity.
