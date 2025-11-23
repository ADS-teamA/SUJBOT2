"""
PostgreSQL BM25 Store - Load BM25 data from PostgreSQL database.

Loads BM25 sparse retrieval data from PostgreSQL (no pickle - uses JSONB).
Compatible with existing BM25Store interface.
"""

import asyncpg
import json
import logging
from typing import List, Dict, Optional, Set
from rank_bm25 import BM25Okapi

from src.hybrid_search import BM25Index
from src.hybrid_search_multilang import load_combined_stopwords

logger = logging.getLogger(__name__)


class PostgresBM25Store:
    """BM25 store loading from PostgreSQL (JSONB, no pickle)."""

    def __init__(self, pool: asyncpg.Pool):
        """Initialize with asyncpg connection pool."""
        self.pool = pool
        self.languages: List[str] = []
        self.lang: str = "en"
        self.nlp_model = None
        self.stop_words: Set[str] = set()

        # BM25 indexes
        self.index_layer1 = BM25Index()
        self.index_layer2 = BM25Index()
        self.index_layer3 = BM25Index()

    async def load(self):
        """Load BM25 from PostgreSQL database."""
        logger.info("Loading BM25 from PostgreSQL...")

        # Load config
        async with self.pool.acquire() as conn:
            config_row = await conn.fetchrow("SELECT * FROM bm25_config ORDER BY id DESC LIMIT 1")

            if config_row:
                self.languages = config_row["languages"]
                self.lang = config_row["primary_language"]
                logger.info(f"BM25 config: languages={self.languages}, primary={self.lang}")
            else:
                self.languages = ["en"]
                self.lang = "en"
                logger.warning("No BM25 config - using defaults")

        # Load stop words
        self.stop_words = load_combined_stopwords(self.languages)
        logger.info(f"Loaded {len(self.stop_words)} stop words")

        # Update indexes
        for idx in [self.index_layer1, self.index_layer2, self.index_layer3]:
            idx.stop_words = self.stop_words

        # Load layers
        await self._load_layer(1, self.index_layer1, "bm25_layer1")
        await self._load_layer(2, self.index_layer2, "bm25_layer2")
        await self._load_layer(3, self.index_layer3, "bm25_layer3")

        logger.info(
            f"PostgreSQL BM25 loaded: "
            f"L1={len(self.index_layer1.corpus)}, "
            f"L2={len(self.index_layer2.corpus)}, "
            f"L3={len(self.index_layer3.corpus)}"
        )

    async def _load_layer(self, layer_num: int, index: BM25Index, table: str):
        """Load one layer from PostgreSQL."""
        logger.info(f"Loading {table}...")

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(f"SELECT chunk_id, corpus, metadata FROM {table} ORDER BY id")

            # Populate (parse metadata from JSONB string to dict)
            index.corpus = [row["corpus"] for row in rows]
            index.chunk_ids = [row["chunk_id"] for row in rows]
            index.metadata = [
                json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
                for row in rows
            ]

            # Build doc_id map
            index.doc_id_map = {}
            for i, meta in enumerate(index.metadata):
                doc_id = meta.get("document_id")
                if doc_id:
                    if doc_id not in index.doc_id_map:
                        index.doc_id_map[doc_id] = []
                    index.doc_id_map[doc_id].append(i)

            # Tokenize and build BM25
            index.tokenized_corpus = [index._tokenize(doc) for doc in index.corpus]
            if index.tokenized_corpus:
                index.bm25 = BM25Okapi(index.tokenized_corpus)
            else:
                index.bm25 = None

        logger.info(f"Loaded {table}: {len(index.corpus)} docs")

    def search_layer1(self, query: str, k: int = 1) -> List[Dict]:
        """Search Layer 1."""
        return self.index_layer1.search(query, k)

    def search_layer2(self, query: str, k: int = 3, document_filter: Optional[str] = None) -> List[Dict]:
        """Search Layer 2."""
        return self.index_layer2.search(query, k, document_filter)

    def search_layer3(self, query: str, k: int = 50, document_filter: Optional[str] = None) -> List[Dict]:
        """Search Layer 3."""
        return self.index_layer3.search(query, k, document_filter)
