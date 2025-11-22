"""
Generate short Czech paraphrases for hybrid-search benchmarking.

- Samples random Layer 3 chunks from the vector store
- Generates a single concise paraphrase (not a question) with GPT-5-mini
- Saves a dataset compatible with existing synthetic question benchmarks

If the OpenAI API key is missing or the call fails, you can enable
`--allow-fallback` to fall back to a deterministic chunk snippet so the
pipeline can still run end-to-end.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock

# Add project root to import path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Mock heavy optional dependencies so the script can run in minimal environments
MODULES_TO_MOCK = [
    "unstructured",
    "unstructured.partition",
    "unstructured.partition.auto",
    "unstructured.partition.pdf",
    "unstructured.partition.pptx",
    "unstructured.partition.docx",
    "unstructured.partition.image",
    "unstructured.partition.html",
    "unstructured.partition.xml",
    "unstructured.partition.md",
    "unstructured.partition.text",
    "unstructured.documents",
    "unstructured.documents.elements",
]
for module_name in MODULES_TO_MOCK:
    sys.modules[module_name] = MagicMock()

# Optional: load .env if available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.faiss_vector_store import FAISSVectorStore
from src.utils.api_clients import APIClientFactory

logger = logging.getLogger(__name__)


def _basic_tokens(text: str) -> set[str]:
    """Lightweight tokenization for fallback keyword generation."""
    import re

    text = text.lower()
    text = re.sub(r"[^0-9a-zá-žěščřďťňůúýíäöüßA-ZÁ-Ž]+", " ", text)
    return {t for t in text.split() if t}


def _enforce_word_limit(text: str, max_words: int = 10) -> str:
    """Trim text to a maximum number of words (default 10)."""
    words = text.replace("\n", " ").split()
    if not words:
        return ""
    trimmed = words[:max_words]
    return " ".join(trimmed)


def is_valid_chunk(chunk: Dict, target_document_id: Optional[str]) -> bool:
    """
    Filter out degenerate chunks.

    Criteria:
    - Matches target document if provided
    - Content present, >= 80 characters and >= 12 words
    - At least 10 unique words to avoid ultra-short/garbled snippets
    """
    content = (chunk.get("content") or "").strip()
    document_id = chunk.get("document_id", "")

    if target_document_id and target_document_id not in document_id:
        return False

    if len(content) < 80:
        return False

    words = content.split()
    if len(words) < 12:
        return False

    if len(set(words)) < 10:
        return False

    return True


def fallback_paraphrase(text: str, max_words: int = 10) -> str:
    """Fallback: short declarative sentence summarizing key terms."""
    tokens = list(_basic_tokens(text))
    if not tokens:
        return ""
    random.shuffle(tokens)
    keywords = ", ".join(tokens[:3])
    sentence = f"Sentence highlights {keywords}."
    return _enforce_word_limit(sentence, max_words=max_words)

def generate_paraphrase(
    client,
    model: str,
    chunk_content: str,
    allow_fallback: bool = False,
    max_attempts: int = 2,
    max_words: int = 10,
) -> str:
    """Generate a short Czech paraphrase limited to max_words."""
    base_prompt = f"""Write one meaningful sentence in Czech.
- Maximum of {max_words} words
- Use different wording than the source, no questions or quotes
- Preserve the main requirement or claim from the text
Text:
{chunk_content}
Sentence:"""

    if client:
        last_text = ""
        for attempt in range(1, max_attempts + 1):
            try:
                attempt_prompt = (
                    base_prompt
                    if attempt == 1
                    else f"{base_prompt}\n\nWrite another Czech sentence, still at most {max_words} words."
                )
                completion_params = {
                    "model": model,
                    "messages": [{"role": "user", "content": attempt_prompt}],
                }

                if "gpt-5" in model or model.startswith("o1"):
                    completion_params["max_completion_tokens"] = 80
                else:
                    completion_params["max_tokens"] = 80
                    completion_params["temperature"] = 0.35

                response = client.chat.completions.create(**completion_params)
                text = (response.choices[0].message.content or "").strip().strip('"“”')
                last_text = text

                if not text:
                    continue

                text = _enforce_word_limit(text, max_words=max_words)
                if not text:
                    continue

                return text
            except Exception as e:
                logger.warning(
                    "Paraphrase generation failed (attempt %d/%d), will %s: %s",
                    attempt,
                    max_attempts,
                    "fallback" if allow_fallback else "skip",
                    e,
                )

        if last_text:
            logger.info("Using best available paraphrase after %d attempts.", max_attempts)
            return _enforce_word_limit(last_text, max_words=max_words)

    if allow_fallback:
        return fallback_paraphrase(chunk_content, max_words=max_words)

    return ""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate hybrid-search benchmark paraphrases (Czech).")
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of paraphrases to generate (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_dataset/synthetic_paraphrases_bz_vr.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--vector-store",
        type=str,
        default="vector_db",
        help="Path to vector store directory (default: vector_db)",
    )
    parser.add_argument(
        "--document-id",
        type=str,
        default="BZ_VR1",
        help="Document ID filter; substring match (default: BZ_VR1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="OpenAI chat model to use (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for chunk sampling (default: 42)",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=float,
        default=3.0,
        help="How many candidate chunks to sample relative to requested count (default: 3.0)",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Use deterministic chunk snippets if the OpenAI call fails or no key is available",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    random.seed(args.seed)

    vector_store_path = Path(args.vector_store)
    if not vector_store_path.exists():
        raise FileNotFoundError(f"Vector store not found at {vector_store_path}")

    logger.info("Loading vector store from %s", vector_store_path)
    vector_store = FAISSVectorStore.load(vector_store_path)

    chunks = vector_store.metadata_layer3
    if not chunks:
        raise RuntimeError("No Layer 3 chunks found in the vector store.")

    valid_chunks: List[Dict] = [
        c for c in chunks if is_valid_chunk(c, args.document_id)
    ]
    if not valid_chunks:
        raise RuntimeError("No valid chunks found after filtering.")

    sample_size = min(
        len(valid_chunks),
        max(args.count, int(args.count * max(1.0, args.candidate_multiplier))),
    )
    sampled_chunks = random.sample(valid_chunks, sample_size)
    logger.info(
        "Sampling %d candidate chunks (need %d paraphrases) from %d valid chunks (of %d total layer3).",
        sample_size,
        args.count,
        len(valid_chunks),
        len(chunks),
    )

    client = None
    try:
        client = APIClientFactory.create_openai()
    except Exception as e:
        logger.warning("OpenAI client not available; generation will rely on fallback only: %s", e)
        if not args.allow_fallback:
            raise

    paraphrases: List[Dict] = []

    for idx, chunk in enumerate(sampled_chunks, start=1):
        if len(paraphrases) >= args.count:
            break

        chunk_id = chunk.get("chunk_id")
        document_id = chunk.get("document_id")
        content = (chunk.get("content") or "").strip()

        logger.info("[%d/%d] Generating paraphrase for chunk %s", idx, sample_size, chunk_id)
        paraphrase = generate_paraphrase(client, args.model, content, allow_fallback=args.allow_fallback)

        if not paraphrase:
            logger.warning("Skipping chunk %s: no paraphrase generated.", chunk_id)
            continue

        paraphrases.append(
            {
                "question": paraphrase,
                "ground_truth_chunk_id": chunk_id,
                "ground_truth_document_id": document_id,
                "chunk_content_preview": content[:200] + ("..." if len(content) > 200 else ""),
            }
        )

    if len(paraphrases) < args.count:
        logger.warning(
            "Generated %d/%d paraphrases. Increase --candidate-multiplier or enable --allow-fallback for more.",
            len(paraphrases),
            args.count,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(paraphrases, f, indent=2, ensure_ascii=False)

    logger.info("Generated %d paraphrases. Saved to %s", len(paraphrases), output_path)


if __name__ == "__main__":
    main()
