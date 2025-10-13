"""
Pre-filter module for intelligent chunk filtering before compliance analysis.

This module implements lightweight filtering to skip irrelevant chunks early,
reducing LLM calls and improving performance.

Filtering strategies:
1. Keyword-based: Skip chunks without compliance-related keywords
2. Size-based: Skip too small or too large chunks
3. Content type: Prioritize obligations, prohibitions, definitions
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FilterMetrics:
    """Metrics for pre-filtering"""
    total_chunks: int
    filtered_chunks: int
    skipped_chunks: int
    filter_reasons: Dict[str, int]

    @property
    def reduction_percentage(self) -> float:
        """Calculate reduction percentage"""
        if self.total_chunks == 0:
            return 0.0
        return (self.skipped_chunks / self.total_chunks) * 100


class ChunkPreFilter:
    """
    Pre-filter chunks before expensive compliance analysis.

    Implements multiple filtering strategies based on research findings
    that show ~30-50% of chunks can be safely skipped without
    impacting compliance detection quality.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize pre-filter.

        Args:
            config: Configuration dictionary with filtering thresholds
        """
        self.config = config or {}

        # Filtering thresholds
        self.min_chunk_size = self.config.get('min_chunk_size', 50)
        self.max_chunk_size = self.config.get('max_chunk_size', 5000)

        # Czech legal compliance keywords (obligations, prohibitions, rights)
        self.compliance_keywords = {
            'obligations': [
                'musí', 'je povinen', 'je povinnost', 'má povinnost',
                'zavazuje se', 'je nucen', 'je třeba', 'je nutné',
                'shall', 'must', 'required', 'mandatory', 'obliged'
            ],
            'prohibitions': [
                'nesmí', 'je zakázáno', 'zakazuje se', 'je nepřípustné',
                'není dovoleno', 'není oprávněn', 'nemá právo',
                'prohibited', 'forbidden', 'shall not', 'must not', 'may not'
            ],
            'rights': [
                'má právo', 'je oprávněn', 'může', 'smí', 'nárok',
                'entitled', 'may', 'authorized', 'right to', 'permitted'
            ],
            'conditions': [
                'pokud', 'jestliže', 'v případě', 'za podmínky', 'pouze když',
                'if', 'when', 'unless', 'provided that', 'subject to'
            ],
            'warranties': [
                'záruka', 'záruční', 'ručí', 'odpovídá za',
                'warranty', 'guarantee', 'warrants', 'guarantees'
            ],
            'liability': [
                'odpovědnost', 'náhrada škody', 'újma', 'ztráta',
                'liability', 'damages', 'indemnify', 'compensation'
            ],
            'deadlines': [
                'lhůta', 'termín', 'do', 'dní', 'měsíců', 'let',
                'deadline', 'term', 'within', 'days', 'months', 'period'
            ]
        }

        # Compile all keywords into set for fast lookup
        self.all_keywords = set()
        for keywords in self.compliance_keywords.values():
            self.all_keywords.update(k.lower() for k in keywords)

        # Non-compliance patterns (skip these)
        self.skip_patterns = [
            r'^(příloha|annex|appendix)\s+\d+',  # Appendices
            r'^(tabulka|table)\s+\d+',  # Tables
            r'^(obrázek|figure|graf)\s+\d+',  # Figures
            r'^(zdroj|source):',  # Source citations
            r'^(poznámka|note):',  # Notes
            r'^\d+\.\s*$',  # Empty numbered sections
        ]

        self.logger = logging.getLogger(__name__)

    def filter_chunks(
        self,
        chunks: List[Any],
        strategy: str = "balanced"
    ) -> tuple[List[Any], FilterMetrics]:
        """
        Filter chunks before compliance analysis.

        Args:
            chunks: List of LegalChunk objects
            strategy: Filtering strategy:
                - "aggressive": Max filtering (~50% reduction)
                - "balanced": Medium filtering (~30% reduction) [DEFAULT]
                - "conservative": Min filtering (~10% reduction)

        Returns:
            (filtered_chunks, metrics)
        """
        self.logger.info(f"Pre-filtering {len(chunks)} chunks with '{strategy}' strategy")

        filtered = []
        filter_reasons = {
            'too_small': 0,
            'too_large': 0,
            'no_keywords': 0,
            'skip_pattern': 0,
            'low_priority': 0
        }

        for chunk in chunks:
            # Check size
            chunk_size = len(chunk.content)
            if chunk_size < self.min_chunk_size:
                filter_reasons['too_small'] += 1
                continue
            if chunk_size > self.max_chunk_size:
                filter_reasons['too_large'] += 1
                continue

            # Check skip patterns (appendices, tables, etc.)
            if self._matches_skip_pattern(chunk.content):
                filter_reasons['skip_pattern'] += 1
                continue

            # Keyword-based filtering (strategy-dependent)
            if strategy == "aggressive":
                # Require at least 2 compliance keywords
                if not self._has_keywords(chunk.content, min_count=2):
                    filter_reasons['no_keywords'] += 1
                    continue
            elif strategy == "balanced":
                # Require at least 1 compliance keyword
                if not self._has_keywords(chunk.content, min_count=1):
                    filter_reasons['no_keywords'] += 1
                    continue
            # Conservative: skip keyword check

            # Priority filtering (aggressive only)
            if strategy == "aggressive":
                priority_score = self._calculate_priority_score(chunk)
                if priority_score < 0.3:  # Low priority
                    filter_reasons['low_priority'] += 1
                    continue

            # Chunk passed all filters
            filtered.append(chunk)

        # Build metrics
        metrics = FilterMetrics(
            total_chunks=len(chunks),
            filtered_chunks=len(filtered),
            skipped_chunks=len(chunks) - len(filtered),
            filter_reasons=filter_reasons
        )

        self.logger.info(
            f"Pre-filtering complete: {metrics.filtered_chunks}/{metrics.total_chunks} chunks kept "
            f"({metrics.reduction_percentage:.1f}% reduction)"
        )
        self.logger.debug(f"Filter reasons: {filter_reasons}")

        return filtered, metrics

    def _matches_skip_pattern(self, text: str) -> bool:
        """Check if text matches skip patterns"""
        text_lower = text.lower().strip()
        for pattern in self.skip_patterns:
            if re.match(pattern, text_lower):
                return True
        return False

    def _has_keywords(self, text: str, min_count: int = 1) -> bool:
        """Check if text contains minimum number of compliance keywords"""
        text_lower = text.lower()
        count = sum(1 for keyword in self.all_keywords if keyword in text_lower)
        return count >= min_count

    def _calculate_priority_score(self, chunk: Any) -> float:
        """
        Calculate priority score for a chunk (0.0-1.0).

        Higher score = more important for compliance checking.

        Factors:
        - Content type (obligation/prohibition = high priority)
        - Keyword density
        - Legal reference presence
        - Structural level
        """
        score = 0.0
        text_lower = chunk.content.lower()

        # Factor 1: Content type (from metadata)
        content_type = getattr(chunk, 'metadata', {}).get('content_type', 'general')
        type_scores = {
            'obligation': 1.0,
            'prohibition': 1.0,
            'condition': 0.8,
            'right': 0.7,
            'definition': 0.5,
            'general': 0.3
        }
        score += type_scores.get(content_type, 0.3) * 0.4  # 40% weight

        # Factor 2: Keyword density
        keyword_count = sum(1 for kw in self.all_keywords if kw in text_lower)
        keyword_density = min(keyword_count / 5.0, 1.0)  # Normalize to max 5 keywords
        score += keyword_density * 0.3  # 30% weight

        # Factor 3: Legal reference presence
        has_reference = bool(getattr(chunk, 'legal_reference', None))
        score += (1.0 if has_reference else 0.0) * 0.2  # 20% weight

        # Factor 4: Structural level (paragraph/article level = higher priority)
        structural_level = getattr(chunk, 'structural_level', 'general')
        level_scores = {
            'paragraph': 1.0,  # §89 level
            'article': 0.9,    # Článek level
            'subsection': 0.8,
            'section': 0.6,
            'chapter': 0.4,
            'part': 0.3,
            'general': 0.2
        }
        score += level_scores.get(structural_level, 0.2) * 0.1  # 10% weight

        return min(score, 1.0)  # Clamp to [0, 1]

    def get_filter_summary(self, metrics: FilterMetrics) -> str:
        """Generate human-readable filter summary"""
        summary = f"""
Pre-filtering Summary:
---------------------
Total chunks: {metrics.total_chunks}
Kept: {metrics.filtered_chunks} ({(metrics.filtered_chunks/metrics.total_chunks)*100:.1f}%)
Skipped: {metrics.skipped_chunks} ({metrics.reduction_percentage:.1f}%)

Reasons for skipping:
"""
        for reason, count in metrics.filter_reasons.items():
            if count > 0:
                summary += f"  - {reason}: {count}\n"

        return summary
