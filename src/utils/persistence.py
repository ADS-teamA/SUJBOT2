"""
Persistence utilities for MY_SUJBOT pipeline.

Provides unified save/load patterns with hybrid serialization:
- JSON for configuration and small metadata (human-readable)
- Pickle for large arrays and complex objects (performance)

Replaces duplicated save/load/merge code in:
- faiss_vector_store.py (save, load, merge)
- hybrid_search.py (BM25Index, BM25Store, HybridVectorStore)

Features:
- Backward compatibility (supports old pickle-only format)
- Helper functions for common patterns (doc_id_to_indices updates)
- Directory creation and validation

Usage:
    from src.utils import PersistenceManager

    # Save JSON config
    PersistenceManager.save_json(path, {"dimensions": 3072})

    # Save pickle arrays
    PersistenceManager.save_pickle(path, large_array)

    # Update doc_id indices during merge
    PersistenceManager.update_doc_id_indices(target, source, offset)
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class Serializable(Protocol):
    """
    Protocol for objects that can be serialized to/from dictionary.

    Implementing classes should provide:
    - to_dict(): Serialize to dictionary
    - from_dict(cls, data): Deserialize from dictionary
    """

    def to_dict(self) -> Dict[str, Any]:
        """Serialize object to dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Serializable":
        """Deserialize object from dictionary."""
        ...


class PersistenceManager:
    """
    Unified persistence manager with hybrid serialization.

    Provides:
    - JSON save/load (for config, small metadata)
    - Pickle save/load (for arrays, complex objects)
    - Common merge patterns (doc_id_to_indices updates)
    - Directory management
    """

    @staticmethod
    def save_json(path: Path, data: Any, **kwargs) -> None:
        """
        Save data as JSON (human-readable).

        Best for:
        - Configuration (dimensions, model names, etc.)
        - Small metadata (counts, stats)
        - Data that needs human inspection

        Args:
            path: Path to JSON file
            data: Data to save (must be JSON-serializable)
            **kwargs: Additional arguments for json.dump (e.g., indent, ensure_ascii)

        Example:
            >>> PersistenceManager.save_json(
            >>>     Path("config.json"),
            >>>     {"dimensions": 3072, "model": "bge-m3"},
            >>>     indent=2
            >>> )
        """
        # Ensure parent directory exists
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Default JSON options
        json_kwargs = {"indent": 2, "ensure_ascii": False, **kwargs}

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, **json_kwargs)

        logger.debug(f"Saved JSON: {path}")

    @staticmethod
    def load_json(path: Path) -> Any:
        """
        Load JSON data.

        Args:
            path: Path to JSON file

        Returns:
            Loaded data

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON

        Example:
            >>> config = PersistenceManager.load_json(Path("config.json"))
            >>> print(config["dimensions"])
        """
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.debug(f"Loaded JSON: {path}")
        return data

    @staticmethod
    def save_pickle(path: Path, data: Any) -> None:
        """
        Save data as pickle (fast, binary).

        Best for:
        - Large arrays (embeddings, vectors)
        - Complex Python objects
        - Performance-critical data

        WARNING: Pickle is NOT secure for untrusted data. Only load pickle
        files from trusted sources.

        Args:
            path: Path to pickle file
            data: Data to save (must be pickle-able)

        Example:
            >>> PersistenceManager.save_pickle(
            >>>     Path("embeddings.pkl"),
            >>>     {"layer1": large_array, "layer2": another_array}
            >>> )
        """
        # Ensure parent directory exists
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.debug(f"Saved pickle: {path}")

    @staticmethod
    def load_pickle(path: Path) -> Any:
        """
        Load pickle data.

        Args:
            path: Path to pickle file

        Returns:
            Loaded data

        Raises:
            FileNotFoundError: If file doesn't exist
            pickle.UnpicklingError: If file is not valid pickle

        Example:
            >>> embeddings = PersistenceManager.load_pickle(Path("embeddings.pkl"))
            >>> print(embeddings["layer1"].shape)
        """
        path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)

        logger.debug(f"Loaded pickle: {path}")
        return data

    @staticmethod
    def update_doc_id_indices(
        target_map: Dict[str, List[int]], source_map: Dict[str, List[int]], offset: int
    ) -> None:
        """
        Update doc_id_to_indices mapping with offset (common merge pattern).

        This is used during vector store merging to update index mappings
        when combining multiple indexes.

        Args:
            target_map: Target doc_id_to_indices dict (modified in-place)
            source_map: Source doc_id_to_indices dict
            offset: Offset to add to all indices (typically len(target.metadata))

        Example:
            >>> # Merging two vector stores
            >>> target = {"doc1": [0, 1, 2]}
            >>> source = {"doc1": [0, 1], "doc2": [0]}
            >>> offset = 3  # len(target.metadata)
            >>>
            >>> PersistenceManager.update_doc_id_indices(target, source, offset)
            >>> # Result: {"doc1": [0, 1, 2, 3, 4], "doc2": [3]}
        """
        for doc_id, indices in source_map.items():
            # Create entry if doesn't exist
            if doc_id not in target_map:
                target_map[doc_id] = []

            # Add offset indices
            target_map[doc_id].extend([idx + offset for idx in indices])

        logger.debug(f"Updated doc_id indices with offset={offset} for {len(source_map)} documents")

    @staticmethod
    def ensure_directory(path: Path) -> None:
        """
        Ensure directory exists (create if needed).

        Args:
            path: Path to directory

        Example:
            >>> PersistenceManager.ensure_directory(Path("output/vector_store"))
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path}")


class VectorStoreLoader:
    """
    Backward-compatible vector store loader.

    Supports both old (pickle-only) and new (hybrid JSON+pickle) formats.

    Old format (legacy):
    - layer1.index, layer2.index, layer3.index (FAISS)
    - metadata.pkl (pickle)

    New format (current):
    - faiss_layer1.index, faiss_layer2.index, faiss_layer3.index (FAISS)
    - faiss_metadata.json (config: dimensions, counts)
    - faiss_arrays.pkl (large arrays: metadata lists, doc_id_to_indices)
    """

    @staticmethod
    def detect_format(path: Path) -> str:
        """
        Detect vector store format.

        Args:
            path: Path to vector store directory

        Returns:
            "new" (hybrid JSON+pickle) or "old" (pickle-only)

        Raises:
            ValueError: If format cannot be determined

        Example:
            >>> format_type = VectorStoreLoader.detect_format(Path("output/vector_store"))
            >>> print(f"Format: {format_type}")
        """
        path = Path(path)

        # Check for new format files
        if (path / "faiss_metadata.json").exists():
            return "new"

        # Check for old format files
        elif (path / "metadata.pkl").exists():
            return "old"

        else:
            raise ValueError(
                f"Cannot determine vector store format at {path}. "
                f"Expected either 'faiss_metadata.json' (new) or 'metadata.pkl' (old)."
            )

    @staticmethod
    def load_with_backward_compatibility(
        path: Path, loader_class, prefer_format: Optional[str] = None
    ):
        """
        Load vector store with automatic format detection and migration.

        Args:
            path: Path to vector store directory or connection string
            loader_class: Class with load() classmethod
            prefer_format: Optional format preference ("new" or "old")

        Returns:
            Loaded vector store instance

        Example:
            >>> from src.storage import load_vector_store_adapter
            >>> store = await load_vector_store_adapter(
            >>>     backend="postgresql",
            >>>     connection_string="postgresql://..."
            >>> )
        """
        format_type = VectorStoreLoader.detect_format(path)

        if format_type == "old":
            logger.info(
                f"Detected old format at {path}. " f"Loading with backward compatibility..."
            )
        else:
            logger.info(f"Detected new format at {path}. Loading...")

        # Use existing load() method
        return loader_class.load(path)


# Example usage
if __name__ == "__main__":
    import tempfile
    import numpy as np

    print("=== Persistence Manager Examples ===\n")

    # Create temp directory for examples
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Example 1: Save/load JSON config
        print("1. JSON save/load...")
        config = {"dimensions": 3072, "model": "bge-m3", "layers": [1, 2, 3]}
        json_path = temp_path / "config.json"
        PersistenceManager.save_json(json_path, config)
        loaded_config = PersistenceManager.load_json(json_path)
        print(f"   ✓ Config: {loaded_config}")

        # Example 2: Save/load pickle arrays
        print("\n2. Pickle save/load...")
        arrays = {
            "layer1": np.random.rand(10, 3072),
            "layer2": np.random.rand(50, 3072),
        }
        pickle_path = temp_path / "arrays.pkl"
        PersistenceManager.save_pickle(pickle_path, arrays)
        loaded_arrays = PersistenceManager.load_pickle(pickle_path)
        print(f"   ✓ Arrays: layer1 shape={loaded_arrays['layer1'].shape}")

        # Example 3: Update doc_id indices
        print("\n3. Update doc_id indices (merge pattern)...")
        target = {"doc1": [0, 1, 2], "doc2": [3, 4]}
        source = {"doc1": [0, 1], "doc3": [0]}
        offset = 5  # Simulating merge
        PersistenceManager.update_doc_id_indices(target, source, offset)
        print(f"   ✓ Merged: {target}")

        # Example 4: Format detection
        print("\n4. Format detection...")
        # Create a fake old format
        (temp_path / "old_format" / "metadata.pkl").parent.mkdir(parents=True, exist_ok=True)
        PersistenceManager.save_pickle(temp_path / "old_format" / "metadata.pkl", {})
        format_type = VectorStoreLoader.detect_format(temp_path / "old_format")
        print(f"   ✓ Detected format: {format_type}")

    print("\n=== All examples completed ===")
