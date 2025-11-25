# RAG Confidence – Top 6 Regressor Proposals

This document lists six concrete regressor candidates for predicting retrieval quality (e.g., recall@10) **using only the data available in the `dataset.json` file**:

- query text
- candidate chunk IDs
- per-candidate `retriever_score`, `retriever_score_norm`, and `rank`

For each regressor we give:
- a short intuition
- a formula you can implement directly.

Throughout, let:

- `K` = number of top results you use for features (typically `K = 10`).
- `C_K` = the top-`K` candidates sorted by ascending `rank`.
- For candidate `i` (1-indexed) in `C_K`:
  - `s_i` = `retriever_score_norm` of candidate `i`
  - `S_i` = raw `retriever_score` of candidate `i`
  - `r_i` = `rank` of candidate `i` (so typically `r_i = i` if already sorted)

---

## 1. Normalized Score Standard Deviation (NQC-style)

**Intuition**

Measures how *spread out* the normalized scores of top results are. A higher spread (one or a few strong winners, weaker tail) tends to correlate with better retrieval quality than a flat, uncertain distribution.

**Formula**

For the top-`K` normalized scores:

- Mean score  
  \[
  \mu_K = \frac{1}{K} \sum_{i=1}^K s_i
  \]

- Standard deviation  
  \[
  \sigma_K = \sqrt{\frac{1}{K} \sum_{i=1}^K (s_i - \mu_K)^2}
  \]

**Regressor**

Use `σ_K` for a fixed `K` (e.g. `K = 10`). You can optionally compute it for multiple K (5, 10, 20) and let the regression decide which matters.

---

## 2. Normalized Score Entropy

**Intuition**

Converts normalized scores into a probability distribution over top-`K` results and measures its entropy. Low entropy (one/few high-probability results) typically indicates confident retrieval; high entropy (almost uniform) indicates uncertainty.

**Formula**

Convert scores to probabilities:

\[
p_i = \frac{s_i}{\sum_{j=1}^K s_j + \varepsilon}
\]

where \(\varepsilon\) is a small constant, e.g. \(10^{-12}\), to avoid division by zero.

Entropy of top-`K`:

\[
H_K = - \sum_{i=1}^K p_i \log p_i
\]

You may use natural log; the base doesn’t matter for regression (it’s just a scale factor).

**Regressor**

Use `H_K` for `K = 10`. Optionally you can normalize by `log K` to get entropy in `[0,1]`, but it’s not required.

---

## 3. Score Slope vs Rank

**Intuition**

Fits a straight line to normalized scores as a function of rank. A steep negative slope means the ranking quickly drops from a strong best match to weaker ones; a flat slope means all scores look similar.

**Formula**

Let the ranks be `x_i = i` for `i = 1..K`, and `y_i = s_i`. Define:

- Mean rank  
  \[
  \bar{x} = \frac{1}{K} \sum_{i=1}^K x_i
  \]

- Mean score  
  \[
  \bar{y} = \frac{1}{K} \sum_{i=1}^K y_i
  \]

The least-squares slope `b_K` of scores vs rank is:

\[
b_K = \frac{\sum_{i=1}^K (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^K (x_i - \bar{x})^2}
\]

**Regressor**

Use `b_K` (typically negative). The regression can learn that more negative slopes often correlate with better recall@10.

---

## 4. Top-vs-Rest Normalized Score Ratio

**Intuition**

Measures how much stronger the top result is compared to the rest of the top-`K`. Large ratio → a very confident top hit; small ratio → many competing results of similar strength.

**Formula**

Let `s_1` be the top normalized score, and define the mean of the remaining top-`K` scores:

\[
\text{mean\_rest}_K = \frac{1}{K - 1} \sum_{i=2}^K s_i
\]

Then the top-vs-rest ratio is:

\[
\text{TV}_K = \frac{s_1}{\text{mean\_rest}_K + \varepsilon}
\]

with a small \(\varepsilon\) (e.g. \(10^{-12}\)) to avoid division by zero.

**Regressor**

Use `TV_K` with `K = 10`.

---

## 5. Section Diversity in Top-K (from Chunk IDs)

**Intuition**

Your chunk IDs have the form:

```text
BZ_VR1_L3_sec_60_chunk_1
```

The substring before `"_chunk_"` can be treated as a **section ID** (a proxy for a document or logical region). Section diversity in top-k tells you whether the retriever is concentrating on a few sections (low diversity) or scattering across many (high diversity).

**Deriving section IDs**

For each candidate ID:

```python
section_id = id.split("_chunk_")[0]
```

**Formula**

For top-`K`:

- Let `section_i` be the section ID for candidate `i`.
- Let `U_K` be the set of distinct section IDs among the top-`K`.

Then:

\[
M_K = |U_K|, \quad
\text{diversity}_K = \frac{M_K}{K}
\]

**Regressor**

Use `diversity_K` with `K = 10`. The regression can learn whether lower or higher diversity corresponds to better recall in your corpus.

---

## 6. Query Token Length

**Intuition**

Short queries (few tokens) tend to be more ambiguous; extremely long queries may include a lot of noise. Query length is a classic, cheap predictor of retrieval difficulty.

**Formula**

Given the query string `q` from `dataset.json`:

1. Tokenize by simple whitespace:

   ```python
   tokens = q.split()
   ```

2. Define:

   \[
   L_q = |\text{tokens}|
   \]

**Regressor**

Use `L_q` (the number of tokens in the query) as a scalar feature. You can also add a binary flag like “very short query” (e.g., `L_q ≤ 3`) if you want an explicit non-linear signal, but `L_q` alone is enough for a linear regression to start with.

---

## Summary

These six regressors are all:

- **Computable directly from your current dataset and runtime data**  
  (no need for embeddings, graph metadata, or extra retrieval passes).
- **Supported by IR/QPP intuition**  
  (score dispersion, entropy, slope, contrast between top and tail, locality, and query difficulty).
- **Safe inputs** to a regression model predicting recall@10, where non-significant features can be dropped after fitting.
