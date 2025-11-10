"""
Relationship extraction from legal document chunks using LLM.

Extracts semantic relationships between entities (SUPERSEDED_BY, REFERENCES, etc.)
using LLM-based inference and heuristic rules.
"""

import json
import logging
import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import Entity, Relationship, RelationshipType
from .config import RelationshipExtractionConfig

try:
    from ..cost_tracker import get_global_tracker
except ImportError:
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from cost_tracker import get_global_tracker


logger = logging.getLogger(__name__)


class RelationshipExtractor:
    """
    LLM-based relationship extractor for knowledge graphs.

    Extracts relationships between entities using:
    1. Within-chunk extraction: Relationships from single chunk context
    2. Cross-chunk extraction: Relationships across multiple chunks
    3. Metadata-based extraction: Relationships from document structure
    """

    def __init__(
        self,
        config: RelationshipExtractionConfig,
        api_key: Optional[str] = None,
    ):
        """
        Initialize relationship extractor.

        Args:
            config: Relationship extraction configuration
            api_key: API key for LLM provider
        """
        self.config = config
        self.api_key = api_key

        # Initialize cost tracker
        self.tracker = get_global_tracker()

        # Initialize LLM client
        self._initialize_llm_client()

        # Cache for extracted relationships
        self.cache: Dict[str, List[Relationship]] = {}

    def _initialize_llm_client(self):
        """Initialize LLM client based on provider."""
        if self.config.llm_provider == "openai":
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed")

        elif self.config.llm_provider == "anthropic":
            try:
                from anthropic import Anthropic

                self.client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed")

        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")

    def extract_relationships(
        self,
        entities: List[Entity],
        chunks: List[Dict[str, Any]],
    ) -> List[Relationship]:
        """
        Extract relationships between entities.

        Args:
            entities: List of extracted entities
            chunks: List of source chunks

        Returns:
            List of extracted Relationship objects
        """
        logger.info(
            f"Extracting relationships for {len(entities)} entities across {len(chunks)} chunks..."
        )

        all_relationships = []

        # 1. Within-chunk extraction
        if self.config.extract_within_chunk:
            logger.info("Extracting within-chunk relationships...")
            within_chunk_rels = self._extract_within_chunk_relationships(entities, chunks)
            all_relationships.extend(within_chunk_rels)

        # 2. Cross-chunk extraction (expensive)
        if self.config.extract_cross_chunk:
            logger.info("Extracting cross-chunk relationships...")
            cross_chunk_rels = self._extract_cross_chunk_relationships(entities, chunks)
            all_relationships.extend(cross_chunk_rels)

        # 3. Metadata-based extraction (fast heuristics)
        if self.config.extract_from_metadata:
            logger.info("Extracting metadata-based relationships...")
            metadata_rels = self._extract_metadata_relationships(entities, chunks)
            all_relationships.extend(metadata_rels)

        # Deduplicate relationships
        all_relationships = self._deduplicate_relationships(all_relationships)

        logger.info(f"Extracted {len(all_relationships)} unique relationships")
        return all_relationships

    def _extract_within_chunk_relationships(
        self,
        entities: List[Entity],
        chunks: List[Dict[str, Any]],
    ) -> List[Relationship]:
        """Extract relationships within single chunks."""
        # Build entity lookup by chunk
        chunk_entities: Dict[str, List[Entity]] = {}
        for entity in entities:
            for chunk_id in entity.source_chunk_ids:
                if chunk_id not in chunk_entities:
                    chunk_entities[chunk_id] = []
                chunk_entities[chunk_id].append(entity)

        # Build chunk lookup
        chunks_by_id = {chunk.get("id", str(i)): chunk for i, chunk in enumerate(chunks)}

        # Process chunks in parallel
        all_relationships = []
        chunk_tasks = []

        for chunk_id, chunk_ents in chunk_entities.items():
            if len(chunk_ents) < 2:
                continue  # Need at least 2 entities for relationships

            chunk = chunks_by_id.get(chunk_id)
            if not chunk:
                continue

            chunk_tasks.append((chunk, chunk_ents))

        # Batch processing
        batches = [
            chunk_tasks[i : i + self.config.batch_size]
            for i in range(0, len(chunk_tasks), self.config.batch_size)
        ]

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._extract_from_chunk_batch, batch): batch for batch in batches
            }

            for future in as_completed(futures):
                try:
                    batch_rels = future.result()
                    all_relationships.extend(batch_rels)
                except Exception as e:
                    logger.error(f"Failed to extract relationships from batch: {e}")

        return all_relationships

    def _extract_from_chunk_batch(
        self, batch: List[Tuple[Dict[str, Any], List[Entity]]]
    ) -> List[Relationship]:
        """Extract relationships from a batch of (chunk, entities) pairs."""
        batch_relationships = []

        for chunk, entities in batch:
            chunk_id = chunk.get("id", str(uuid.uuid4()))

            # Check cache
            if self.config.cache_results and chunk_id in self.cache:
                batch_relationships.extend(self.cache[chunk_id])
                continue

            # Extract relationships
            try:
                relationships = self._extract_from_single_chunk(chunk, entities)

                # Cache results
                if self.config.cache_results:
                    self.cache[chunk_id] = relationships

                batch_relationships.extend(relationships)

            except Exception as e:
                logger.error(f"Failed to extract relationships from chunk {chunk_id}: {e}")

        return batch_relationships

    def _extract_from_single_chunk(
        self, chunk: Dict[str, Any], entities: List[Entity]
    ) -> List[Relationship]:
        """Extract relationships from a single chunk using LLM."""
        chunk_id = chunk.get("id", str(uuid.uuid4()))
        chunk_content = chunk.get("raw_content", chunk.get("content", ""))

        # Build extraction prompt
        prompt = self._build_relationship_prompt(chunk_content, entities)

        # Call LLM
        response_text = self._call_llm(prompt)

        # Parse response
        relationships = self._parse_llm_response(
            response_text,
            entities=entities,
            chunk_id=chunk_id,
        )

        return relationships

    def _build_relationship_prompt(self, chunk_content: str, entities: List[Entity]) -> str:
        """Build relationship extraction prompt (loads from template file)."""
        from pathlib import Path

        # Load template from file
        prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        template_path = prompts_dir / "relationship_extraction.txt"

        try:
            template = template_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.error(f"Relationship extraction template not found: {template_path}")
            logger.warning("Using fallback inline prompt")
            return self._build_relationship_prompt_fallback(chunk_content, entities)

        # Format entity list
        entity_list = []
        for i, entity in enumerate(entities):
            entity_id = f"E{i+1}"
            entity_list.append(f"  {entity_id}: {entity.value} ({entity.type.value})")

        entities_str = "\n".join(entity_list)

        # Enabled relationship types
        rel_types_str = ", ".join([rt.value for rt in self.config.enabled_relationship_types])

        # Substitute placeholders
        prompt = template.format(
            entities_str=entities_str,
            chunk_content=chunk_content,
            rel_types_str=rel_types_str,
            min_confidence=self.config.min_confidence,
            max_evidence_length=self.config.max_evidence_length
        )

        return prompt

    def _build_relationship_prompt_fallback(self, chunk_content: str, entities: List[Entity]) -> str:
        """Fallback inline prompt if template file missing."""
        # Format entity list
        entity_list = []
        for i, entity in enumerate(entities):
            entity_id = f"E{i+1}"
            entity_list.append(f"  {entity_id}: {entity.value} ({entity.type.value})")

        entities_str = "\n".join(entity_list)
        rel_types_str = ", ".join([rt.value for rt in self.config.enabled_relationship_types])

        prompt = f"""Extract semantic relationships between entities in the following legal document text.

**Entities** (extracted from this text):
{entities_str}

**Document Text**:
{chunk_content}

**Instructions**:
1. Identify ALL relationships between entities
2. Only use relationship types: {rel_types_str}
3. Only include relationships with confidence >= {self.config.min_confidence}
4. Return valid JSON array

**Output** (JSON array only):
"""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with retry logic."""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                if self.config.llm_provider == "openai":
                    # Prepare base parameters
                    messages = [
                        {
                            "role": "system",
                            "content": "You are an expert at extracting semantic relationships from legal documents. Always return valid JSON.",
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
                        system="You are an expert at extracting semantic relationships from legal documents. Always return valid JSON.",
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

                    time.sleep(1.0 * (attempt + 1))
                else:
                    raise

        raise Exception("Max retries exceeded for LLM call")

    def _parse_llm_response(
        self,
        response_text: str,
        entities: List[Entity],
        chunk_id: str,
    ) -> List[Relationship]:
        """Parse LLM response and create Relationship objects."""
        relationships = []

        # Build entity ID lookup (E1 -> Entity)
        entity_lookup = {}
        for i, entity in enumerate(entities):
            entity_id = f"E{i+1}"
            entity_lookup[entity_id] = entity

        try:
            # Extract JSON from response
            json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                json_text = response_text.strip()

            # Parse JSON
            relationships_data = json.loads(json_text)

            if not isinstance(relationships_data, list):
                logger.warning(f"Expected list of relationships, got {type(relationships_data)}")
                return []

            # Create Relationship objects
            for rel_data in relationships_data:
                try:
                    relationship = self._create_relationship_from_dict(
                        rel_data,
                        entity_lookup=entity_lookup,
                        chunk_id=chunk_id,
                    )

                    # Filter by confidence
                    if relationship.confidence >= self.config.min_confidence:
                        relationships.append(relationship)

                except Exception as e:
                    logger.warning(f"Failed to create relationship from data {rel_data}: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response text: {response_text[:500]}")

        return relationships

    def _create_relationship_from_dict(
        self,
        rel_data: Dict[str, Any],
        entity_lookup: Dict[str, Entity],
        chunk_id: str,
    ) -> Relationship:
        """Create Relationship object from dictionary."""
        # Get source and target entities
        source_entity_id_str = rel_data.get("source_entity_id", "")
        target_entity_id_str = rel_data.get("target_entity_id", "")

        source_entity = entity_lookup.get(source_entity_id_str)
        target_entity = entity_lookup.get(target_entity_id_str)

        if not source_entity or not target_entity:
            raise ValueError(f"Invalid entity IDs: {source_entity_id_str}, {target_entity_id_str}")

        # Parse relationship type
        rel_type_str = rel_data.get("relationship_type", "").lower()
        try:
            rel_type = RelationshipType(rel_type_str)
        except ValueError:
            raise ValueError(f"Invalid relationship type: {rel_type_str}")

        # Check if enabled
        if rel_type not in self.config.enabled_relationship_types:
            raise ValueError(f"Relationship type {rel_type} not enabled")

        # Extract fields
        confidence = float(rel_data.get("confidence", 0.0))
        evidence_text = rel_data.get("evidence", "")

        # Truncate evidence if too long
        if len(evidence_text) > self.config.max_evidence_length:
            evidence_text = evidence_text[: self.config.max_evidence_length] + "..."

        # Create relationship
        relationship = Relationship(
            id=str(uuid.uuid4()),
            type=rel_type,
            source_entity_id=source_entity.id,
            target_entity_id=target_entity.id,
            confidence=confidence,
            source_chunk_id=chunk_id,
            evidence_text=evidence_text,
            properties={
                "source_entity_value": source_entity.value,
                "target_entity_value": target_entity.value,
            },
            extraction_method="llm",
            extracted_at=datetime.now(),
        )

        return relationship

    def _extract_cross_chunk_relationships(
        self,
        entities: List[Entity],
        chunks: List[Dict[str, Any]],
    ) -> List[Relationship]:
        """
        Extract relationships that span multiple chunks.

        This is more expensive as it requires comparing entities across chunks.
        Focus on specific relationship types that are likely to be cross-chunk:
        - REFERENCES (cross-document references)
        - SUPERSEDED_BY (standard evolution)
        """
        logger.info("Cross-chunk extraction not yet implemented (expensive operation)")
        return []

    def _extract_metadata_relationships(
        self,
        entities: List[Entity],
        chunks: List[Dict[str, Any]],
    ) -> List[Relationship]:
        """
        Extract relationships from document metadata using heuristics.

        Examples:
        - PART_OF relationships from section_path
        - MENTIONED_IN relationships (entity -> chunk)
        - Document structure relationships
        """
        relationships = []

        # Extract MENTIONED_IN relationships (provenance)
        for entity in entities:
            for chunk_id in entity.source_chunk_ids:
                relationship = Relationship(
                    id=str(uuid.uuid4()),
                    type=RelationshipType.MENTIONED_IN,
                    source_entity_id=entity.id,
                    target_entity_id=chunk_id,  # Chunk as target
                    confidence=1.0,  # High confidence (direct provenance)
                    source_chunk_id=chunk_id,
                    evidence_text=f"Entity '{entity.value}' mentioned in chunk",
                    properties={
                        "provenance": True,
                        "entity_type": entity.type.value,
                    },
                    extraction_method="heuristic",
                    extracted_at=datetime.now(),
                )
                relationships.append(relationship)

        logger.info(f"Extracted {len(relationships)} metadata-based relationships")
        return relationships

    def _deduplicate_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """
        Deduplicate relationships based on (type, source_entity_id, target_entity_id).

        Keeps highest confidence relationship.
        """
        # Group by (type, source, target)
        rel_groups: Dict[tuple, List[Relationship]] = {}

        for rel in relationships:
            key = (rel.type, rel.source_entity_id, rel.target_entity_id)
            if key not in rel_groups:
                rel_groups[key] = []
            rel_groups[key].append(rel)

        # Keep highest confidence
        deduplicated = []
        for key, group in rel_groups.items():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Sort by confidence (descending)
                group = sorted(group, key=lambda r: r.confidence, reverse=True)
                deduplicated.append(group[0])

        logger.info(f"Deduplicated {len(relationships)} relationships to {len(deduplicated)}")
        return deduplicated

    def extract_from_entity_pairs(
        self,
        entity_pairs: List[Tuple[Entity, Entity]],
        chunk_content: str,
        chunk_id: str,
    ) -> List[Relationship]:
        """
        Extract relationships for specific entity pairs (convenience method).

        Args:
            entity_pairs: List of (source_entity, target_entity) tuples
            chunk_content: Text content containing entities
            chunk_id: Chunk ID

        Returns:
            List of extracted relationships
        """
        # Flatten entity pairs to unique entities
        entities = []
        seen_ids = set()
        for source, target in entity_pairs:
            if source.id not in seen_ids:
                entities.append(source)
                seen_ids.add(source.id)
            if target.id not in seen_ids:
                entities.append(target)
                seen_ids.add(target.id)

        chunk = {
            "id": chunk_id,
            "raw_content": chunk_content,
            "content": chunk_content,
        }

        return self._extract_from_single_chunk(chunk, entities)
