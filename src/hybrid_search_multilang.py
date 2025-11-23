"""
Multi-language BM25 support extension.

Adds Czech + English bilingual tokenization to BM25Store.
"""

import logging
from pathlib import Path
from typing import List, Set, Optional
from src.hybrid_search import BM25Store, load_nltk_stopwords, CZECH_STOP_WORDS

logger = logging.getLogger(__name__)


def load_combined_stopwords(languages: List[str]) -> Set[str]:
    """
    Load and combine stop words from multiple languages.

    Args:
        languages: List of language codes (e.g., ['cs', 'en'])

    Returns:
        Combined set of stop words from all languages
    """
    combined = set()

    for lang in languages:
        if lang == 'cs':
            # Use hardcoded Czech stop words
            combined.update(CZECH_STOP_WORDS)
            logger.info(f"Added {len(CZECH_STOP_WORDS)} Czech stop words")
        else:
            # Use NLTK for other languages
            stopwords = load_nltk_stopwords(lang)
            if stopwords:
                combined.update(stopwords)
                logger.info(f"Added {len(stopwords)} {lang} stop words")

    logger.info(f"Combined stop words: {len(combined)} total from {len(languages)} languages")
    return combined


class MultiLangBM25Store(BM25Store):
    """
    Multi-language BM25 store supporting Czech + English simultaneously.

    Usage:
        # Create bilingual BM25 store
        bm25_store = MultiLangBM25Store(languages=['cs', 'en'])
        bm25_store.build_from_chunks(chunks_dict)

        # Save with multi-language config
        bm25_store.save(Path('vector_db'))
    """

    def __init__(self, languages: List[str] = None):
        """
        Initialize multi-language BM25 store.

        Args:
            languages: List of language codes (e.g., ['cs', 'en'])
                      Default: ['cs', 'en'] for Czech + English
        """
        if languages is None:
            languages = ['cs', 'en']

        self.languages = languages

        # Combine stop words from all languages
        combined_stopwords = load_combined_stopwords(languages)

        # Initialize parent class with combined stop words
        # Set lang to first language for display purposes
        super().__init__(lang=languages[0])

        # Override stop words with combined set
        self.stop_words = combined_stopwords

        # Update all index stop words
        self.index_layer1.stop_words = combined_stopwords
        self.index_layer2.stop_words = combined_stopwords
        self.index_layer3.stop_words = combined_stopwords

        logger.info(
            f"MultiLangBM25Store initialized: "
            f"languages={languages}, "
            f"combined_stopwords={len(combined_stopwords)}"
        )

    def save(self, output_dir: Path) -> None:
        """
        Save with multi-language configuration.

        Args:
            output_dir: Directory to save indexes
        """
        from src.utils.persistence import PersistenceManager

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving multi-language BM25 store to {output_dir}")

        # Save indexes (same as parent)
        self.index_layer1.save(output_dir / "bm25_layer1.pkl")
        self.index_layer2.save(output_dir / "bm25_layer2.pkl")
        self.index_layer3.save(output_dir / "bm25_layer3.pkl")

        # Save multi-language configuration
        config = {
            "languages": self.languages,  # NEW: List of languages
            "lang": self.lang,  # Keep for backward compatibility
            "format_version": "3.0"  # v3.0 with multi-language support
        }
        PersistenceManager.save_json(output_dir / "bm25_store_config.json", config)

        logger.info(f"Multi-language BM25 store saved (languages={self.languages})")

    @classmethod
    def load(cls, input_dir: Path) -> "MultiLangBM25Store":
        """
        Load multi-language BM25 store.

        Args:
            input_dir: Directory containing indexes

        Returns:
            MultiLangBM25Store instance
        """
        from src.utils.persistence import PersistenceManager
        from src.hybrid_search import BM25Index

        input_dir = Path(input_dir)
        logger.info(f"Loading multi-language BM25 store from {input_dir}")

        # Load configuration
        config_path = input_dir / "bm25_store_config.json"
        if config_path.exists():
            config = PersistenceManager.load_json(config_path)

            # Check if multi-language config (v3.0)
            if config.get("format_version") == "3.0" and "languages" in config:
                languages = config["languages"]
                logger.info(f"Loading multi-language config: {languages}")
            else:
                # Single language config - convert to multi-language
                lang = config.get("lang", "en")
                languages = [lang]
                logger.info(f"Converting single-language config ({lang}) to multi-language")
        else:
            # Default to Czech + English
            languages = ['cs', 'en']
            logger.warning(
                "No BM25 config found. Using default: Czech + English"
            )

        # Create multi-language store
        store = cls(languages=languages)

        # Load index data
        store.index_layer1 = BM25Index.load(input_dir / "bm25_layer1.pkl")
        store.index_layer2 = BM25Index.load(input_dir / "bm25_layer2.pkl")
        store.index_layer3 = BM25Index.load(input_dir / "bm25_layer3.pkl")

        # Re-inject combined stop words
        combined_stopwords = load_combined_stopwords(languages)
        for index in [store.index_layer1, store.index_layer2, store.index_layer3]:
            index.nlp = store.nlp_model
            index.stop_words = combined_stopwords

        # Re-tokenize with combined stop words
        logger.info("Re-tokenizing with multi-language stop words...")
        for layer_name, index in [
            ("L1", store.index_layer1),
            ("L2", store.index_layer2),
            ("L3", store.index_layer3)
        ]:
            from rank_bm25 import BM25Okapi
            index.tokenized_corpus = [index._tokenize(doc) for doc in index.corpus]
            if index.tokenized_corpus:
                index.bm25 = BM25Okapi(index.tokenized_corpus)
            else:
                index.bm25 = None
            logger.info(f"Rebuilt BM25 for {layer_name} ({len(index.corpus)} docs)")

        logger.info(f"Multi-language BM25 store loaded (languages={languages})")

        return store


# Example usage
if __name__ == "__main__":
    print("=== Multi-Language BM25 Example ===\n")

    # Create bilingual BM25 store
    print("1. Creating Czech + English BM25 store...")
    store = MultiLangBM25Store(languages=['cs', 'en'])
    print(f"   Stop words: {len(store.stop_words)}")
    print(f"   Languages: {store.languages}")
    print()

    # Save
    print("2. Saving multi-language store...")
    print("   store.save(Path('vector_db'))")
    print()

    # Load
    print("3. Loading multi-language store...")
    print("   loaded = MultiLangBM25Store.load(Path('vector_db'))")
    print()

    print("=== Implementation complete! ===")
