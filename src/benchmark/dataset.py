"""
Dataset loading and validation.

Loads QA pairs from privacy_qa.json format:
{
    "tests": [
        {
            "query": "...",
            "snippets": [
                {
                    "file_path": "privacy_qa/Fiverr.txt",
                    "span": [start, end],
                    "answer": "..."
                }
            ]
        }
    ]
}
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthSnippet:
    """
    Ground truth answer snippet from source document.

    Attributes:
        file_path: Relative path to source document
        span_start: Character start position in document
        span_end: Character end position in document
        answer: Ground truth answer text
    """

    file_path: str
    span_start: int
    span_end: int
    answer: str

    def __post_init__(self):
        """Validate snippet."""
        if self.span_end <= self.span_start:
            raise ValueError(f"Invalid span: end ({self.span_end}) <= start ({self.span_start})")
        if not self.answer.strip():
            raise ValueError("Answer text cannot be empty")


@dataclass
class QueryExample:
    """
    Single query-answer pair from benchmark dataset.

    Attributes:
        query_id: Unique query identifier (1-indexed)
        query: Question text
        snippets: List of ground truth answer snippets
    """

    query_id: int
    query: str
    snippets: List[GroundTruthSnippet]

    def __post_init__(self):
        """Validate query example."""
        if not self.query.strip():
            raise ValueError(f"Query {self.query_id}: Query text cannot be empty")
        if not self.snippets:
            raise ValueError(f"Query {self.query_id}: Must have at least one snippet")

    def get_expected_answers(self) -> List[str]:
        """
        Get list of all ground truth answers for this query.

        Returns:
            List of answer texts (one per snippet)
        """
        return [snippet.answer for snippet in self.snippets]

    def get_source_documents(self) -> List[str]:
        """
        Get list of unique source document paths.

        Returns:
            Unique file paths referenced in snippets
        """
        return list(set(snippet.file_path for snippet in self.snippets))


class BenchmarkDataset:
    """
    Loads and manages benchmark dataset.

    Usage:
        dataset = BenchmarkDataset.from_json("benchmark_dataset/privacy_qa.json")
        queries = dataset.get_queries(max_queries=10)  # First 10 queries
    """

    def __init__(self, queries: List[QueryExample]):
        """
        Initialize dataset.

        Args:
            queries: List of validated query examples
        """
        self.queries = queries
        logger.info(f"Loaded {len(queries)} queries from dataset")

    @classmethod
    def from_json(cls, json_path: str) -> "BenchmarkDataset":
        """
        Load dataset from privacy_qa.json format.

        Args:
            json_path: Path to JSON file

        Returns:
            BenchmarkDataset instance

        Raises:
            FileNotFoundError: If JSON file not found
            ValueError: If JSON format is invalid
        """
        json_path = Path(json_path)

        if not json_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {json_path}\n" f"Expected path: {json_path.absolute()}"
            )

        logger.info(f"Loading dataset from {json_path}")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")

        # Parse JSON structure
        if "tests" not in data:
            raise ValueError(
                "Invalid format: Expected 'tests' key in JSON.\n"
                "Format should be: {'tests': [{'query': '...', 'snippets': [...]}]}"
            )

        tests = data["tests"]
        if not isinstance(tests, list):
            raise ValueError("Invalid format: 'tests' must be a list")

        # Parse each query example
        queries = []
        valid_id = 1  # Separate counter for valid queries (no gaps)
        for i, test in enumerate(tests, start=1):
            try:
                # Parse snippets
                snippets = []
                for snippet_data in test.get("snippets", []):
                    span = snippet_data.get("span", [])
                    if len(span) != 2:
                        raise ValueError(f"Span must have exactly 2 elements, got {len(span)}")

                    snippet = GroundTruthSnippet(
                        file_path=snippet_data["file_path"],
                        span_start=span[0],
                        span_end=span[1],
                        answer=snippet_data["answer"],
                    )
                    snippets.append(snippet)

                # Create query example (use valid_id for sequential IDs)
                query = QueryExample(
                    query_id=valid_id,
                    query=test["query"],
                    snippets=snippets,
                )
                queries.append(query)
                valid_id += 1  # Increment only on success

            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping invalid query at index {i}: {e}")
                continue

        if not queries:
            raise ValueError("No valid queries found in dataset")

        logger.info(f"Successfully loaded {len(queries)} queries")

        return cls(queries)

    def get_queries(self, max_queries: int = None) -> List[QueryExample]:
        """
        Get query examples (optionally limited).

        Args:
            max_queries: Maximum number of queries to return (None = all)

        Returns:
            List of query examples
        """
        if max_queries is None:
            return self.queries
        return self.queries[:max_queries]

    def get_query_by_id(self, query_id: int) -> QueryExample:
        """
        Get specific query by ID.

        Args:
            query_id: Query identifier (1-indexed)

        Returns:
            QueryExample

        Raises:
            ValueError: If query_id not found
        """
        for query in self.queries:
            if query.query_id == query_id:
                return query

        raise ValueError(f"Query ID {query_id} not found. Valid IDs: 1-{len(self.queries)}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dict with statistics (total queries, avg snippets per query, etc.)
        """
        total_queries = len(self.queries)
        total_snippets = sum(len(q.snippets) for q in self.queries)
        avg_snippets = total_snippets / total_queries if total_queries > 0 else 0

        # Get unique source documents
        all_docs = set()
        for query in self.queries:
            all_docs.update(query.get_source_documents())

        return {
            "total_queries": total_queries,
            "total_snippets": total_snippets,
            "avg_snippets_per_query": round(avg_snippets, 2),
            "unique_source_documents": len(all_docs),
            "source_documents": sorted(all_docs),
        }
