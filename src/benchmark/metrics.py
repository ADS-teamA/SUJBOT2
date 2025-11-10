"""
Evaluation metrics for benchmark.

Implements LegalBench-RAG standard metrics:
- Exact Match (EM): Binary exact string match
- F1 Score: Token-level harmonic mean of precision/recall
- Precision: Correct tokens / predicted tokens
- Recall: Correct tokens / ground truth tokens
- Embedding Similarity: Semantic similarity using cosine similarity in embedding space
  (replaces SequenceMatcher - captures semantic meaning vs structural patterns)

All metrics use simple functions (not classes) for easy testing and reuse.
"""

import re
import logging
from typing import List, Dict, Set, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded embedding model for semantic similarity
_EMBEDDER = None

# Metric name abbreviations for display
METRIC_ABBREVIATIONS = {
    "exact_match": "EM",
    "f1_score": "F1",
    "precision": "P",
    "recall": "R",
    "embedding_similarity": "EMB",
    "combined_f1": "F1C",
    "rag_confidence": "RAG",
}


def _get_embedder():
    """
    Get or initialize the embedding model for semantic similarity metrics.

    Uses all-MiniLM-L6-v2:
    - Small (22M parameters)
    - Fast inference
    - Optimized for semantic textual similarity
    - State-of-the-art for STS benchmarks

    Returns:
        SentenceTransformer model (singleton)
    """
    global _EMBEDDER

    if _EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model for semantic similarity: all-MiniLM-L6-v2")
            _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Embedding model loaded successfully")

        except ImportError as e:
            import time

            error_id = f"ERR_EMBEDDER_MISSING_{int(time.time())}"
            logger.error(f"[{error_id}] sentence-transformers not installed", exc_info=True)
            raise ImportError(
                f"[{error_id}] Missing dependency: sentence-transformers\n"
                f"Install with: pip install sentence-transformers\n"
                f"Or disable embedding metrics in config"
            ) from e
        except Exception as e:
            import time

            error_id = f"ERR_EMBEDDER_LOAD_{int(time.time())}"
            logger.error(f"[{error_id}] Failed to load all-MiniLM-L6-v2: {e}", exc_info=True)
            raise RuntimeError(
                f"[{error_id}] Embedding model loading failed. Possible causes:\n"
                f"  - Network issues (model downloads from HuggingFace)\n"
                f"  - Insufficient disk space (~100MB needed)\n"
                f"  - CUDA/GPU compatibility issues\n"
                f"Check logs for details."
            ) from e

    return _EMBEDDER


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    Normalization:
    - Lowercase
    - Remove punctuation
    - Collapse whitespace
    - Strip leading/trailing whitespace

    Args:
        text: Raw text

    Returns:
        Normalized text

    Example:
        >>> normalize_text("Hello, World!  ")
        'hello world'
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation (keep only alphanumeric and spaces)
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Strip
    return text.strip()


def tokenize(text: str) -> Set[str]:
    """
    Tokenize text into set of words.

    Uses simple whitespace splitting after normalization.

    Args:
        text: Text to tokenize

    Returns:
        Set of tokens (lowercased, normalized)

    Example:
        >>> tokenize("Hello, World!")
        {'hello', 'world'}
    """
    normalized = normalize_text(text)
    if not normalized:
        return set()
    return set(normalized.split())


# ==============================================================================
# CORE METRICS (LegalBench-RAG Standard)
# ==============================================================================


def _calculate_token_metrics(predicted: str, references: List[str]) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score in one pass.

    This helper function computes all token-level metrics for the best matching
    reference to avoid code duplication in calculate_precision(), calculate_recall(),
    and calculate_f1_score().

    Args:
        predicted: Model's predicted answer
        references: List of ground truth answers

    Returns:
        Dict with 'precision', 'recall', 'f1_score' for the best matching reference

    Example:
        >>> _calculate_token_metrics("hello world", ["hello world"])
        {'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0}
    """
    pred_tokens = tokenize(predicted)

    if not pred_tokens:
        # Empty prediction - all metrics are zero
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    best_f1 = 0.0
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    for ref in references:
        ref_tokens = tokenize(ref)

        if not ref_tokens:
            # Empty reference - skip
            continue

        # Calculate token overlap
        common_tokens = pred_tokens & ref_tokens

        if not common_tokens:
            # No overlap - metrics are zero
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            # Calculate precision and recall
            precision = len(common_tokens) / len(pred_tokens)
            recall = len(common_tokens) / len(ref_tokens)

            # Calculate F1 score
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

        # Track best metrics (by F1 score)
        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {"precision": precision, "recall": recall, "f1_score": f1}

    return best_metrics


def calculate_exact_match(predicted: str, references: List[str]) -> float:
    """
    Exact match metric (case-insensitive).

    Returns 1.0 if prediction exactly matches ANY reference, else 0.0.

    Args:
        predicted: Model's predicted answer
        references: List of ground truth answers

    Returns:
        1.0 if exact match, 0.0 otherwise

    Example:
        >>> calculate_exact_match("hello world", ["hello world", "hi"])
        1.0
        >>> calculate_exact_match("hello", ["hello world"])
        0.0
    """
    pred_norm = normalize_text(predicted)

    for ref in references:
        ref_norm = normalize_text(ref)
        if pred_norm == ref_norm:
            return 1.0

    return 0.0


def calculate_f1_score(predicted: str, references: List[str]) -> float:
    """
    Token-level F1 score.

    Computes F1 for each reference and returns the maximum.
    F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        predicted: Model's predicted answer
        references: List of ground truth answers

    Returns:
        Max F1 score across all references [0.0, 1.0]

    Example:
        >>> calculate_f1_score("hello world", ["hello world"])
        1.0
        >>> calculate_f1_score("hello", ["hello world"])
        0.6666...  # precision=1.0, recall=0.5 -> F1=0.67
    """
    return _calculate_token_metrics(predicted, references)["f1_score"]


def calculate_precision(predicted: str, references: List[str]) -> float:
    """
    Token-level precision.

    Precision = (correct tokens) / (predicted tokens)
    Returns max precision across all references.

    Args:
        predicted: Model's predicted answer
        references: List of ground truth answers

    Returns:
        Max precision [0.0, 1.0]

    Example:
        >>> calculate_precision("hello world", ["hello"])
        0.5  # 1 correct out of 2 predicted
    """
    return _calculate_token_metrics(predicted, references)["precision"]


def calculate_recall(predicted: str, references: List[str]) -> float:
    """
    Token-level recall.

    Recall = (correct tokens) / (ground truth tokens)
    Returns max recall across all references.

    Args:
        predicted: Model's predicted answer
        references: List of ground truth answers

    Returns:
        Max recall [0.0, 1.0]

    Example:
        >>> calculate_recall("hello", ["hello world"])
        0.5  # 1 correct out of 2 ground truth
    """
    return _calculate_token_metrics(predicted, references)["recall"]


def calculate_embedding_similarity(predicted: str, references: List[str]) -> float:
    """
    Semantic similarity using cosine similarity in embedding space.

    Embeds both predicted answer and ground truth answers using a semantic
    similarity model (all-MiniLM-L6-v2), then computes cosine similarity.

    This measures semantic similarity rather than structural/lexical similarity:
    - Paraphrases score high (semantic match)
    - Word order doesn't matter (meaning-based)
    - Captures conceptual similarity

    Uses sentence-transformers with cosine similarity:
    cos_sim = (A · B) / (||A|| × ||B||)

    Args:
        predicted: Model's predicted answer
        references: List of ground truth answers

    Returns:
        Max cosine similarity across all references [0.0, 1.0]

    Example:
        >>> calculate_embedding_similarity("waste disposal", ["garbage removal"])
        ~0.85  # High semantic similarity despite different words
        >>> calculate_embedding_similarity("hello world", ["goodbye moon"])
        ~0.3   # Low semantic similarity
    """
    # Normalize text (but keep semantic content)
    pred_norm = normalize_text(predicted)

    if not pred_norm:
        return 0.0

    # Get embedder (lazy-loaded singleton)
    embedder = _get_embedder()

    # Embed predicted answer
    pred_embedding = embedder.encode(pred_norm, convert_to_tensor=False, normalize_embeddings=True)

    max_similarity = 0.0

    for ref in references:
        ref_norm = normalize_text(ref)

        if not ref_norm:
            continue

        # Embed reference answer
        ref_embedding = embedder.encode(
            ref_norm, convert_to_tensor=False, normalize_embeddings=True
        )

        # Cosine similarity (embeddings are already normalized, so just dot product)
        # For normalized vectors: cos_sim = A · B
        similarity = float(np.dot(pred_embedding, ref_embedding))

        max_similarity = max(max_similarity, similarity)

    return max_similarity


# ==============================================================================
# COMBINED METRICS (for multi-part ground truth answers)
# ==============================================================================


def calculate_combined_f1(predicted: str, references: List[str]) -> float:
    """
    Calculate F1 score against COMBINED ground truth answers.

    When there are multiple GT answers, they are concatenated into a single
    reference. This is useful when each GT answer contains different parts
    of information and the ideal answer should cover ALL parts.

    Differs from calculate_f1_score() which takes MAX over individual references.

    Args:
        predicted: Model's predicted answer
        references: List of ground truth answers (will be concatenated)

    Returns:
        F1 score against combined reference [0.0, 1.0]

    Example:
        >>> refs = ["IP address and cookies", "username and photos"]
        >>> # MAX approach: F1 with best match
        >>> calculate_f1_score("username and photos", refs)  # ~1.0
        >>> # COMBINED approach: F1 with concatenated refs
        >>> calculate_combined_f1("username and photos", refs)  # ~0.5 (missing IP address)
    """
    # Concatenate all references with space separator
    combined_ref = " ".join(references)

    # Calculate metrics against combined reference
    metrics = _calculate_token_metrics(predicted, [combined_ref])

    return metrics["f1_score"]


# ==============================================================================
# METRIC AGGREGATION
# ==============================================================================


def compute_all_metrics(predicted: str, references: List[str]) -> Dict[str, float]:
    """
    Compute all standard metrics at once.

    Args:
        predicted: Model's predicted answer
        references: List of ground truth answers

    Returns:
        Dict mapping metric name -> score

    Example:
        >>> compute_all_metrics("hello world", ["hello world", "hi there"])
        {
            'exact_match': 1.0,
            'f1_score': 1.0,
            'precision': 1.0,
            'recall': 1.0,
            'embedding_similarity': 1.0,
            'combined_f1': 1.0
        }
    """
    return {
        "exact_match": calculate_exact_match(predicted, references),
        "f1_score": calculate_f1_score(predicted, references),
        "precision": calculate_precision(predicted, references),
        "recall": calculate_recall(predicted, references),
        "embedding_similarity": calculate_embedding_similarity(predicted, references),
        "combined_f1": calculate_combined_f1(predicted, references),
    }


def aggregate_metrics(metric_results: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate per-query metrics into mean scores.

    Args:
        metric_results: List of metric dicts (one per query)

    Returns:
        Dict with mean scores per metric

    Example:
        >>> results = [
        ...     {"exact_match": 1.0, "f1_score": 0.8},
        ...     {"exact_match": 0.0, "f1_score": 0.6}
        ... ]
        >>> aggregate_metrics(results)
        {'exact_match': 0.5, 'f1_score': 0.7}
    """
    if not metric_results:
        return {}

    # Collect scores per metric
    metric_scores = {}
    for result in metric_results:
        for metric_name, score in result.items():
            if metric_name not in metric_scores:
                metric_scores[metric_name] = []
            metric_scores[metric_name].append(score)

    # Compute means
    aggregated = {}
    for metric_name, scores in metric_scores.items():
        if scores:
            aggregated[metric_name] = sum(scores) / len(scores)
        else:
            aggregated[metric_name] = 0.0

    return aggregated


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    Format metrics as readable string.

    Args:
        metrics: Dict of metric scores
        precision: Decimal places

    Returns:
        Formatted string

    Example:
        >>> format_metrics({"em": 0.75, "f1": 0.8123}, precision=2)
        'EM: 0.75 | F1: 0.81'
    """
    parts = []
    for name, score in sorted(metrics.items()):
        abbrev = METRIC_ABBREVIATIONS.get(name, name.upper())
        parts.append(f"{abbrev}: {score:.{precision}f}")

    return " | ".join(parts)
