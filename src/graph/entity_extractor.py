"""
Entity extraction from legal document chunks using LLM.

Extracts structured entities (Standards, Organizations, Dates, etc.) from text chunks
using few-shot prompting and parallel batch processing.
"""

import json
import logging
import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import Entity, EntityType
from .config import EntityExtractionConfig

try:
    from ..cost_tracker import get_global_tracker
except ImportError:
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from cost_tracker import get_global_tracker


logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    LLM-based entity extractor for legal documents.

    Extracts entities from chunks using structured prompts and parallel processing.
    Supports entity normalization, confidence scoring, and provenance tracking.
    """

    def __init__(
        self,
        config: EntityExtractionConfig,
        api_key: Optional[str] = None,
    ):
        """
        Initialize entity extractor.

        Args:
            config: Entity extraction configuration
            api_key: API key for LLM provider (overrides config)
        """
        self.config = config
        self.api_key = api_key

        # Initialize cost tracker
        self.tracker = get_global_tracker()

        # Initialize LLM client based on provider
        self._initialize_llm_client()

        # Cache for extracted entities (chunk_id -> entities)
        self.cache: Dict[str, List[Entity]] = {}

    def _initialize_llm_client(self):
        """Initialize LLM client based on provider."""
        if self.config.llm_provider == "openai":
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")

        elif self.config.llm_provider == "anthropic":
            try:
                from anthropic import Anthropic

                self.client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Install with: pip install anthropic"
                )

        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")

    def extract_from_chunks(self, chunks: List[Dict[str, Any]]) -> List[Entity]:
        """
        Extract entities from multiple chunks in parallel.

        Args:
            chunks: List of chunk dictionaries with 'id', 'content', 'metadata'

        Returns:
            List of extracted Entity objects
        """
        logger.info(f"Extracting entities from {len(chunks)} chunks...")

        # Process chunks in batches
        all_entities = []
        batches = [
            chunks[i : i + self.config.batch_size]
            for i in range(0, len(chunks), self.config.batch_size)
        ]

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {executor.submit(self._extract_from_batch, batch): batch for batch in batches}

            for future in as_completed(futures):
                try:
                    batch_entities = future.result()
                    all_entities.extend(batch_entities)
                except Exception as e:
                    batch = futures[future]
                    logger.error(f"Failed to extract entities from batch: {e}")

        # Deduplicate entities across chunks
        if self.config.normalize_entities:
            all_entities = self._deduplicate_entities(all_entities)

        logger.info(f"Extracted {len(all_entities)} unique entities")
        return all_entities

    def _extract_from_batch(self, chunks: List[Dict[str, Any]]) -> List[Entity]:
        """Extract entities from a batch of chunks."""
        batch_entities = []

        for chunk in chunks:
            chunk_id = chunk.get("id", str(uuid.uuid4()))

            # Check cache
            if self.config.cache_results and chunk_id in self.cache:
                batch_entities.extend(self.cache[chunk_id])
                continue

            # Extract entities from chunk
            try:
                entities = self._extract_from_single_chunk(chunk)

                # Cache results
                if self.config.cache_results:
                    self.cache[chunk_id] = entities

                batch_entities.extend(entities)

            except Exception as e:
                logger.error(f"Failed to extract entities from chunk {chunk_id}: {e}")

        return batch_entities

    def _extract_from_single_chunk(self, chunk: Dict[str, Any]) -> List[Entity]:
        """Extract entities from a single chunk using LLM."""
        chunk_id = chunk.get("id", str(uuid.uuid4()))
        chunk_content = chunk.get("raw_content", chunk.get("content", ""))
        chunk_metadata = chunk.get("metadata", {})

        prefill_entities: List[Entity] = []
        if self.config.enable_regex_prefill:
            try:
                prefill_entities = self._regex_extract_entities(
                    chunk_content=chunk_content,
                    chunk_id=chunk_id,
                    chunk_metadata=chunk_metadata,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"Regex prefill failed for chunk {chunk_id}: {exc}")

        # Build extraction prompt
        prompt = self._build_extraction_prompt(chunk_content, chunk_metadata)

        # Call LLM
        response_text = self._call_llm(prompt)

        # Parse response
        entities = prefill_entities + self._parse_llm_response(
            response_text,
            chunk_id=chunk_id,
            document_id=chunk_metadata.get("document_id"),
            section_path=chunk_metadata.get("section_path"),
        )

        return entities

    def _build_extraction_prompt(self, chunk_content: str, chunk_metadata: Dict[str, Any]) -> str:
        """Build entity extraction prompt for LLM (loads from template file)."""
        from pathlib import Path

        # Load template from file
        prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        template_path = prompts_dir / "entity_extraction.txt"

        try:
            template = template_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.error(f"Entity extraction template not found: {template_path}")
            # Fallback to inline prompt if file missing
            logger.warning("Using fallback inline prompt")
            return self._build_extraction_prompt_fallback(chunk_content, chunk_metadata)

        # Prepare substitution variables
        entity_types_str = ", ".join([et.value for et in self.config.enabled_entity_types])

        # Few-shot examples
        few_shot = self._get_few_shot_examples() if self.config.include_examples else ""

        # Substitute placeholders
        prompt = template.format(
            entity_types_str=entity_types_str,
            chunk_content=chunk_content,
            section_path=chunk_metadata.get('section_path', 'N/A'),
            document_id=chunk_metadata.get('document_id', 'N/A'),
            min_confidence=self.config.min_confidence,
            max_entities_per_chunk=self.config.max_entities_per_chunk,
            few_shot_examples=few_shot
        )

        return prompt

    def _build_extraction_prompt_fallback(self, chunk_content: str, chunk_metadata: Dict[str, Any]) -> str:
        """Fallback inline prompt if template file missing."""
        entity_types_str = ", ".join([et.value for et in self.config.enabled_entity_types])

        prompt = f"""Extract structured entities from the following legal document text.

**Task**: Identify and extract all entities of the following types:
{entity_types_str}

**Document Text**:
{chunk_content}

**Metadata Context**:
- Section: {chunk_metadata.get('section_path', 'N/A')}
- Document ID: {chunk_metadata.get('document_id', 'N/A')}

**Instructions**:
1. Extract ALL entities of the specified types
2. Only include entities with confidence >= {self.config.min_confidence}
3. Limit to {self.config.max_entities_per_chunk} entities
4. Return valid JSON array

**Output** (JSON array only, no other text):
"""
        return prompt

    def _get_few_shot_examples(self) -> str:
        """Get few-shot examples for entity extraction."""
        return """
**Example Input**:
"GRI 306: Waste 2020 supersedes GRI 306: Effluents and Waste 2016, which was issued by the Global Reporting Initiative (GRI) and became effective on 1 July 2018. This standard covers waste management topics including hazardous waste disposal."

**Example Output**:
[
  {"type": "standard", "value": "GRI 306: Waste 2020", "normalized_value": "GRI 306", "confidence": 0.95, "context": "GRI 306: Waste 2020 supersedes GRI 306"},
  {"type": "standard", "value": "GRI 306: Effluents and Waste 2016", "normalized_value": "GRI 306", "confidence": 0.95, "context": "supersedes GRI 306: Effluents and Waste 2016"},
  {"type": "organization", "value": "Global Reporting Initiative", "normalized_value": "GRI", "confidence": 0.9, "context": "issued by the Global Reporting Initiative (GRI)"},
  {"type": "organization", "value": "GRI", "normalized_value": "GRI", "confidence": 0.95, "context": "Global Reporting Initiative (GRI)"},
  {"type": "date", "value": "1 July 2018", "normalized_value": "2018-07-01", "confidence": 0.9, "context": "became effective on 1 July 2018"},
  {"type": "topic", "value": "waste management", "normalized_value": "waste management", "confidence": 0.85, "context": "covers waste management topics including"},
  {"type": "topic", "value": "hazardous waste disposal", "normalized_value": "hazardous waste disposal", "confidence": 0.8, "context": "topics including hazardous waste disposal"}
]

"""

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with retry logic."""
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                if self.config.llm_provider == "openai":
                    # Prepare base parameters
                    messages = [
                        {
                            "role": "system",
                            "content": "You are an expert at extracting structured entities from legal documents. Always return valid JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ]

                    params = {
                        "model": self.config.llm_model,
                        "messages": messages,
                    }

                    # GPT-5/o-series models use different parameters
                    if self.config.llm_model.startswith(("gpt-5", "o1-", "o3-")):
                        params["max_completion_tokens"] = 4000
                        # Note: GPT-5 only supports temperature=1.0 (default), don't set it
                    else:
                        # GPT-4 and earlier
                        params["max_tokens"] = 4000
                        params["temperature"] = self.config.temperature

                    response = self.client.chat.completions.create(**params)

                    # Track cost
                    self.tracker.track_llm(
                        provider="openai",
                        model=self.config.llm_model,
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        operation="kg_extraction",
                    )

                    return response.choices[0].message.content.strip()

                elif self.config.llm_provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.config.llm_model,
                        max_tokens=4000,
                        temperature=self.config.temperature,
                        messages=[{"role": "user", "content": prompt}],
                        system="You are an expert at extracting structured entities from legal documents. Always return valid JSON.",
                    )

                    # Track cost
                    self.tracker.track_llm(
                        provider="anthropic",
                        model=self.config.llm_model,
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                        operation="kg_extraction",
                    )

                    return response.content[0].text.strip()

            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time

                    time.sleep(retry_delay * (attempt + 1))
                else:
                    raise

        raise Exception("Max retries exceeded for LLM call")

    def _parse_llm_response(
        self,
        response_text: str,
        chunk_id: str,
        document_id: Optional[str],
        section_path: Optional[str],
    ) -> List[Entity]:
        """Parse LLM response and create Entity objects."""
        entities = []

        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # Try to find JSON array directly
                json_text = response_text.strip()

            # Parse JSON
            entities_data = json.loads(json_text)

            if not isinstance(entities_data, list):
                logger.warning(f"Expected list of entities, got {type(entities_data)}")
                return []

            # Create Entity objects
            for entity_data in entities_data:
                try:
                    entity = self._create_entity_from_dict(
                        entity_data,
                        chunk_id=chunk_id,
                        document_id=document_id,
                        section_path=section_path,
                    )

                    # Filter by confidence
                    if entity.confidence >= self.config.min_confidence:
                        entities.append(entity)

                except Exception as e:
                    logger.warning(f"Failed to create entity from data {entity_data}: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response text: {response_text[:500]}")

        return entities

    def _create_entity_from_dict(
        self,
        entity_data: Dict[str, Any],
        chunk_id: str,
        document_id: Optional[str],
        section_path: Optional[str],
    ) -> Entity:
        """Create Entity object from dictionary."""
        # Parse entity type
        entity_type_str = entity_data.get("type", "").lower()
        try:
            entity_type = EntityType(entity_type_str)
        except ValueError:
            raise ValueError(f"Invalid entity type: {entity_type_str}")

        # Check if enabled
        if entity_type not in self.config.enabled_entity_types:
            raise ValueError(f"Entity type {entity_type} not enabled")

        # Extract fields
        value = entity_data.get("value", "")
        normalized_value = entity_data.get("normalized_value", value)
        confidence = float(entity_data.get("confidence", 0.0))

        # Create entity
        entity = Entity(
            id=str(uuid.uuid4()),
            type=entity_type,
            value=value,
            normalized_value=normalized_value,
            confidence=confidence,
            source_chunk_ids=[chunk_id],
            first_mention_chunk_id=chunk_id,
            document_id=document_id,
            section_path=section_path,
            metadata={
                "context": entity_data.get("context", ""),
                "extraction_confidence": confidence,
            },
            extraction_method="llm",
            extracted_at=datetime.now(),
        )

        return entity

    def _regex_extract_entities(
        self,
        chunk_content: str,
        chunk_id: str,
        chunk_metadata: Dict[str, Any],
    ) -> List[Entity]:
        """Extract simple entities via regex/patterns before calling the LLM."""
        entities: List[Entity] = []
        seen: Set[tuple] = set()
        document_id = chunk_metadata.get("document_id")
        section_path = chunk_metadata.get("section_path")

        def add_entity(
            entity_type: EntityType,
            value: str,
            normalized: Optional[str] = None,
            context: Optional[str] = None,
        ):
            key = (entity_type, (normalized or value).lower())
            if key in seen or entity_type not in self.config.enabled_entity_types:
                return
            seen.add(key)
            confidence = self.config.regex_prefill_confidence
            entities.append(
                Entity(
                    id=str(uuid.uuid4()),
                    type=entity_type,
                    value=value.strip(),
                    normalized_value=(normalized or value).strip(),
                    confidence=confidence,
                    source_chunk_ids=[chunk_id],
                    first_mention_chunk_id=chunk_id,
                    document_id=document_id,
                    section_path=section_path,
                    metadata={
                        "context": context or value.strip(),
                        "extraction_confidence": confidence,
                    },
                    extraction_method="regex",
                    extracted_at=datetime.now(),
                )
            )

        content = chunk_content or ""

        # Dates (textual month)
        month_map = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }

        for match in re.finditer(r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b", content):
            day, month_name, year = match.groups()
            month = month_map[month_name.lower()]
            try:
                normalized = datetime(int(year), month, int(day)).strftime("%Y-%m-%d")
            except ValueError:
                normalized = None
            add_entity(EntityType.DATE, match.group(0), normalized)

        # ISO style dates (YYYY-MM-DD)
        for match in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", content):
            year, month, day = match.groups()
            try:
                normalized = datetime(int(year), int(month), int(day)).strftime("%Y-%m-%d")
            except ValueError:
                normalized = None
            add_entity(EntityType.DATE, match.group(0), normalized)

        # European style dates (DD.MM.YYYY)
        for match in re.finditer(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", content):
            day, month, year = match.groups()
            try:
                normalized = datetime(int(year), int(month), int(day)).strftime("%Y-%m-%d")
            except ValueError:
                normalized = None
            add_entity(EntityType.DATE, match.group(0), normalized)

        # Legal citations and sections
        citation_pattern = re.compile(
            r"(Article|Art\.|Section|Sec\.|Clause|Paragraph|Para\.|§)\s+([\dA-Za-z\.\-\/\(\)]+)",
            re.IGNORECASE,
        )
        for match in citation_pattern.finditer(content):
            keyword = match.group(1).lower()
            label = match.group(2)
            value = match.group(0)
            if keyword in {"article", "art.", "paragraph", "para.", "§"}:
                entity_type = EntityType.LEGAL_PROVISION
            else:
                entity_type = EntityType.CLAUSE
            normalized = f"{keyword} {label}".lower()
            add_entity(entity_type, value, normalized)

        # Standards or technical identifiers (ISO/IEC/EN...)
        standard_pattern = re.compile(
            r"\b(?:ISO|IEC|EN|ASTM|IEEE|GRI)\s*\d{3,5}(?::\d{2,4})?\b",
            re.IGNORECASE,
        )
        for match in standard_pattern.finditer(content):
            value = match.group(0)
            normalized = value.upper().replace(" ", "")
            add_entity(EntityType.STANDARD, value, normalized)

        return entities

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Deduplicate entities based on (type, normalized_value).

        Merges duplicate entities by:
        - Keeping highest confidence
        - Merging source_chunk_ids
        - Keeping first_mention_chunk_id from earliest chunk
        """
        # Group by (type, normalized_value)
        entity_groups: Dict[tuple, List[Entity]] = {}

        for entity in entities:
            key = (entity.type, entity.normalized_value)
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)

        # Merge duplicates
        deduplicated = []
        for key, group in entity_groups.items():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Merge multiple entities
                merged = self._merge_entities(group)
                deduplicated.append(merged)

        logger.info(f"Deduplicated {len(entities)} entities to {len(deduplicated)}")
        return deduplicated

    def _merge_entities(self, entities: List[Entity]) -> Entity:
        """Merge multiple entities into one."""
        # Sort by confidence (descending)
        entities = sorted(entities, key=lambda e: e.confidence, reverse=True)

        # Use highest-confidence entity as base
        merged = entities[0]

        # Merge source_chunk_ids from all entities
        all_chunk_ids = []
        for entity in entities:
            all_chunk_ids.extend(entity.source_chunk_ids)
        merged.source_chunk_ids = list(set(all_chunk_ids))  # Deduplicate

        # Keep first_mention_chunk_id from earliest extraction
        first_mentions = [e.first_mention_chunk_id for e in entities if e.first_mention_chunk_id]
        if first_mentions:
            merged.first_mention_chunk_id = first_mentions[0]

        # Merge metadata
        merged.metadata["merged_from"] = len(entities)
        merged.metadata["all_confidences"] = [e.confidence for e in entities]

        return merged

    def extract_from_text(self, text: str, chunk_id: Optional[str] = None) -> List[Entity]:
        """
        Extract entities from raw text (convenience method for testing).

        Args:
            text: Text content to extract from
            chunk_id: Optional chunk ID (auto-generated if not provided)

        Returns:
            List of extracted entities
        """
        chunk_id = chunk_id or str(uuid.uuid4())

        chunk = {
            "id": chunk_id,
            "raw_content": text,
            "content": text,
            "metadata": {},
        }

        return self._extract_from_single_chunk(chunk)
