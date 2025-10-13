"""
Common Utilities for SUJBOT2

This module provides shared utility functions used across the system including:
- Text processing and normalization
- Token counting and chunking helpers
- File and path utilities
- Data structure conversions
- Performance helpers
"""

import hashlib
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from datetime import datetime
import tiktoken

logger = logging.getLogger(__name__)


# ============================================================================
# Text Processing Utilities
# ============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for processing.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove zero-width characters
    text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def clean_legal_text(text: str) -> str:
    """
    Clean legal text while preserving structure markers.

    Args:
        text: Legal text

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Preserve legal references and structure
    text = normalize_text(text)

    # Remove page numbers (common artifacts)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

    # Remove excessive line breaks (more than 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text


def truncate_text(text: str, max_chars: int = 200, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Input text
        max_chars: Maximum characters
        suffix: Suffix for truncated text

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_chars:
        return text

    return text[:max_chars - len(suffix)] + suffix


# ============================================================================
# Token Counting & Estimation
# ============================================================================

_tokenizer_cache: Optional[tiktoken.Encoding] = None


def get_tokenizer(model: str = "cl100k_base") -> tiktoken.Encoding:
    """
    Get tiktoken tokenizer (cached).

    Args:
        model: Tokenizer model name

    Returns:
        Tokenizer instance
    """
    global _tokenizer_cache

    if _tokenizer_cache is None:
        try:
            _tokenizer_cache = tiktoken.get_encoding(model)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer {model}: {e}")
            _tokenizer_cache = tiktoken.get_encoding("cl100k_base")

    return _tokenizer_cache


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """
    Count tokens in text.

    Args:
        text: Input text
        model: Tokenizer model

    Returns:
        Token count
    """
    if not text:
        return 0

    try:
        tokenizer = get_tokenizer(model)
        return len(tokenizer.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens: {e}")
        # Fallback: rough estimate (1 token ≈ 4 chars)
        return len(text) // 4


def estimate_tokens(text: str) -> int:
    """
    Fast token estimation without tokenizer.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    # Rough estimate: 1 token ≈ 4 characters for English/Czech
    return len(text) // 4


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences (Czech-aware).

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    if not text:
        return []

    # Split on common sentence endings
    # Handle abbreviations like "odst.", "písm.", "čl."
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+', text)

    return [s.strip() for s in sentences if s.strip()]


# ============================================================================
# Legal Reference Parsing
# ============================================================================

def extract_legal_references(text: str) -> List[str]:
    """
    Extract legal references from text.

    Args:
        text: Input text

    Returns:
        List of references
    """
    patterns = [
        r'§\s*\d+(?:\s+odst\.\s*\d+)?(?:\s+písm\.\s*[a-z])?',  # §89 odst. 2 písm. a
        r'[Čč]l(?:ánek|\.)\s*\d+(?:\.\s*\d+)?',  # Článek 5.2
        r'[Pp]aragra(?:f|ph)\s*\d+',  # Paragraph 89
        r'[Oo]dst(?:avec|\.)\s*\d+',  # Odstavec 2
        r'[Pp]ísm(?:ena|\.)\s*[a-z]',  # Písmena a
    ]

    references = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        references.extend(matches)

    return list(set(references))  # Remove duplicates


def normalize_reference(ref: str) -> str:
    """
    Normalize legal reference format.

    Args:
        ref: Legal reference

    Returns:
        Normalized reference
    """
    # Convert to standard format
    ref = ref.strip()

    # Normalize section symbols
    ref = re.sub(r'[Pp]aragra(?:f|ph)', '§', ref)

    # Normalize abbreviations
    ref = re.sub(r'[Oo]dstavec', 'odst.', ref)
    ref = re.sub(r'[Pp]ísmena', 'písm.', ref)
    ref = re.sub(r'[Čč]lánek', 'čl.', ref)

    # Remove excessive whitespace
    ref = re.sub(r'\s+', ' ', ref)

    return ref


# ============================================================================
# File & Path Utilities
# ============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if needed.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """
    Calculate file hash.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha256, etc.)

    Returns:
        Hex digest of hash
    """
    hash_func = hashlib.new(algorithm)

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get file size in MB.

    Args:
        file_path: Path to file

    Returns:
        File size in MB
    """
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)


def safe_filename(filename: str, max_length: int = 255) -> str:
    """
    Create safe filename by removing invalid characters.

    Args:
        filename: Original filename
        max_length: Maximum filename length

    Returns:
        Safe filename
    """
    # Remove invalid characters
    safe = re.sub(r'[<>:"/\\|?*]', '_', filename)

    # Truncate if too long
    if len(safe) > max_length:
        name, ext = os.path.splitext(safe)
        safe = name[:max_length - len(ext)] + ext

    return safe


# ============================================================================
# Data Structure Utilities
# ============================================================================

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary.

    Args:
        d: Nested dictionary
        parent_key: Parent key prefix
        sep: Separator for keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Unflatten dictionary with dot-notation keys.

    Args:
        d: Flattened dictionary
        sep: Key separator

    Returns:
        Nested dictionary
    """
    result = {}

    for key, value in d.items():
        parts = key.split(sep)
        target = result

        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]

        target[parts[-1]] = value

    return result


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge multiple dictionaries.

    Args:
        *dicts: Dictionaries to merge

    Returns:
        Merged dictionary
    """
    result = {}

    for d in dicts:
        for key, value in d.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value)
            else:
                result[key] = value

    return result


# ============================================================================
# Performance & Timing Utilities
# ============================================================================

def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split items into batches.

    Args:
        items: List of items
        batch_size: Size of each batch

    Returns:
        List of batches
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


class Timer:
    """Simple context manager for timing operations."""

    def __init__(self, name: str = "Operation", logger_func=None):
        self.name = name
        self.logger_func = logger_func or logger.info
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        self.elapsed = (self.end_time - self.start_time).total_seconds()
        self.logger_func(f"{self.name} took {self.elapsed:.2f}s")


# ============================================================================
# Validation Utilities
# ============================================================================

def is_valid_uuid(value: str) -> bool:
    """
    Check if string is valid UUID.

    Args:
        value: String to check

    Returns:
        True if valid UUID
    """
    import uuid
    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, AttributeError):
        return False


def is_valid_email(email: str) -> bool:
    """
    Check if string is valid email.

    Args:
        email: Email string

    Returns:
        True if valid email
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_config_value(value: Any, expected_type: type, min_val: Optional[float] = None,
                         max_val: Optional[float] = None) -> bool:
    """
    Validate configuration value.

    Args:
        value: Value to validate
        expected_type: Expected type
        min_val: Minimum value (for numbers)
        max_val: Maximum value (for numbers)

    Returns:
        True if valid
    """
    if not isinstance(value, expected_type):
        return False

    if min_val is not None and isinstance(value, (int, float)):
        if value < min_val:
            return False

    if max_val is not None and isinstance(value, (int, float)):
        if value > max_val:
            return False

    return True


# ============================================================================
# String Similarity
# ============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def jaccard_similarity(set1: set, set2: set) -> float:
    """
    Calculate Jaccard similarity between two sets.

    Args:
        set1: First set
        set2: Second set

    Returns:
        Similarity score [0, 1]
    """
    if not set1 and not set2:
        return 1.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0

    return intersection / union


# ============================================================================
# Export
# ============================================================================

__all__ = [
    # Text processing
    'normalize_text',
    'clean_legal_text',
    'truncate_text',

    # Token counting
    'get_tokenizer',
    'count_tokens',
    'estimate_tokens',
    'split_into_sentences',

    # Legal references
    'extract_legal_references',
    'normalize_reference',

    # File utilities
    'ensure_dir',
    'get_file_hash',
    'get_file_size_mb',
    'safe_filename',

    # Data structures
    'flatten_dict',
    'unflatten_dict',
    'merge_dicts',

    # Performance
    'batch_items',
    'Timer',

    # Validation
    'is_valid_uuid',
    'is_valid_email',
    'validate_config_value',

    # Similarity
    'levenshtein_distance',
    'jaccard_similarity',
]
