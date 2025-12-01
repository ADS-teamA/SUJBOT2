"""
Phase Loaders for Resume Functionality

Loads completed phases from JSON files back into Python objects.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class PhaseLoaders:
    """
    Loads phase outputs from JSON files for resume functionality.

    Reconstructs Python objects (ExtractedDocument, Chunks) from saved JSON.
    """

    @staticmethod
    def load_phase1(json_path: Path):
        """
        Load PHASE 1 extraction result from JSON.

        Args:
            json_path: Path to phase1_extraction.json

        Returns:
            Partial ExtractedDocument (without full text/content)

        Raises:
            FileNotFoundError: If phase1 JSON file doesn't exist
            ValueError: If JSON is corrupted or missing required fields
            UnicodeDecodeError: If file encoding is invalid

        Note:
            Returns minimal object - enough for phase detection.
            Full content not needed since we skip to later phases.
        """
        from src.unstructured_extractor import ExtractedDocument, DocumentSection

        logger.info(f"Loading Phase 1 from {json_path}")

        # Read and parse JSON with specific error handling
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Phase 1 file not found: {json_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Phase 1 JSON corrupted in {json_path}: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Phase 1 file encoding error in {json_path}: {e}")

        # Validate required fields
        required_fields = ["document_id", "source_path", "sections"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(
                f"Phase 1 JSON missing required fields in {json_path}: {missing}"
            )

        # Reconstruct sections with validation
        sections = []
        section_required = ["section_id", "title", "level", "depth", "path", "page_number"]

        for idx, s in enumerate(data["sections"]):
            try:
                # Validate section fields
                missing_section = [f for f in section_required if f not in s]
                if missing_section:
                    raise KeyError(
                        f"Section {idx} missing fields: {missing_section}"
                    )

                sections.append(DocumentSection(
                    section_id=s["section_id"],
                    title=s["title"],
                    level=s["level"],
                    depth=s["depth"],
                    path=s["path"],
                    page_number=s["page_number"],
                    content="",  # Not saved in phase1, will be loaded if needed
                    parent_id=None,  # Not needed for resume
                    children_ids=[],  # Not needed for resume
                    ancestors=[],  # Not needed for resume
                    char_start=0,  # Not saved in phase1
                    char_end=s.get("content_length", 0),  # Use content_length as char_end
                    content_length=s.get("content_length", 0),
                    summary=None  # Loaded in phase2
                ))
            except (KeyError, TypeError) as e:
                raise ValueError(
                    f"Phase 1 section {idx} reconstruction failed in {json_path}: {e}"
                )

        # Create partial ExtractedDocument
        result = ExtractedDocument(
            document_id=data["document_id"],
            source_path=data["source_path"],
            sections=sections,
            hierarchy_depth=data["hierarchy_depth"],
            num_roots=data["num_roots"],
            num_sections=data["num_sections"],
            num_tables=data.get("num_tables", 0),
            # Fields not saved/needed:
            extraction_time=0.0,
            full_text="",
            markdown="",
            json_content={},
            tables=[],
            num_pages=0,
            total_chars=0,
            document_summary=None  # Added in phase2
        )

        logger.info(f"Loaded Phase 1: {result.document_id}, {result.num_sections} sections")
        return result

    @staticmethod
    def load_phase2(json_path: Path, extraction_result):
        """
        Load PHASE 2 summaries and merge into ExtractedDocument.

        Args:
            json_path: Path to phase2_summaries.json
            extraction_result: ExtractedDocument from phase1

        Returns:
            ExtractedDocument with summaries added

        Raises:
            FileNotFoundError: If phase2 JSON file doesn't exist
            ValueError: If JSON is corrupted or missing required fields
        """
        logger.info(f"Loading Phase 2 from {json_path}")

        # Read and parse JSON with error handling
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Phase 2 file not found: {json_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Phase 2 JSON corrupted in {json_path}: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Phase 2 file encoding error in {json_path}: {e}")

        # Validate required fields
        if "document_summary" not in data:
            raise ValueError(
                f"Phase 2 JSON missing 'document_summary' field in {json_path}"
            )
        if "section_summaries" not in data:
            raise ValueError(
                f"Phase 2 JSON missing 'section_summaries' field in {json_path}"
            )

        # Add document summary
        extraction_result.document_summary = data["document_summary"]

        # Add section summaries with validation
        try:
            section_summaries = {s["section_id"]: s["summary"] for s in data["section_summaries"]}
        except KeyError as e:
            raise ValueError(
                f"Phase 2 section summary missing required field in {json_path}: {e}"
            )

        # Log warning if sections missing summaries
        missing_summaries = []
        for section in extraction_result.sections:
            section.summary = section_summaries.get(section.section_id)
            if section.summary is None:
                missing_summaries.append(section.section_id)

        if missing_summaries:
            logger.warning(
                f"Phase 2: {len(missing_summaries)} sections missing summaries: "
                f"{missing_summaries[:3]}..." if len(missing_summaries) > 3 else f"{missing_summaries}"
            )

        logger.info(
            f"Loaded Phase 2: document summary + {len(section_summaries)} section summaries"
        )

        return extraction_result

    @staticmethod
    def load_phase3(json_path: Path) -> Dict[str, List]:
        """
        Load PHASE 3 chunks from JSON.

        Args:
            json_path: Path to phase3_chunks.json

        Returns:
            Dict with keys 'layer1', 'layer2', 'layer3' containing Chunk objects

        Raises:
            FileNotFoundError: If phase3 JSON file doesn't exist
            ValueError: If JSON is corrupted or missing required fields
        """
        from src.multi_layer_chunker import Chunk, ChunkMetadata

        logger.info(f"Loading Phase 3 from {json_path}")

        # Read and parse JSON with error handling
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Phase 3 file not found: {json_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Phase 3 JSON corrupted in {json_path}: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Phase 3 file encoding error in {json_path}: {e}")

        # Validate required fields
        required = ["layer1", "layer2", "layer3"]
        missing = [f for f in required if f not in data]
        if missing:
            raise ValueError(
                f"Phase 3 JSON missing required layer fields in {json_path}: {missing}"
            )

        def reconstruct_chunk(chunk_data: Dict, layer_name: str, idx: int) -> Chunk:
            """Reconstruct Chunk object from dict with error handling."""
            try:
                if "metadata" not in chunk_data:
                    raise KeyError("missing 'metadata' field")

                meta = chunk_data["metadata"]

                # Validate chunk required fields
                # New format uses: raw_content, context, embedding_text (no 'content' field)
                if "chunk_id" not in chunk_data:
                    raise KeyError("missing 'chunk_id' field")
                if "raw_content" not in chunk_data:
                    raise KeyError("missing 'raw_content' field")

                # Validate metadata required fields
                meta_required = ["chunk_id", "layer", "document_id"]
                meta_missing = [f for f in meta_required if f not in meta]
                if meta_missing:
                    raise KeyError(f"metadata missing fields: {meta_missing}")

                # New format: use embedding_text as content (text used for retrieval)
                # embedding_text = context-enriched text for embedding/search
                content = chunk_data.get("embedding_text") or chunk_data["raw_content"]

                return Chunk(
                    chunk_id=chunk_data["chunk_id"],
                    content=content,
                    raw_content=chunk_data["raw_content"],
                    metadata=ChunkMetadata(
                        chunk_id=meta["chunk_id"],
                        layer=meta["layer"],
                        document_id=meta["document_id"],
                        title=meta.get("title"),
                        section_id=meta.get("section_id"),
                        parent_chunk_id=meta.get("parent_chunk_id"),
                        page_number=meta.get("page_number", 0),
                        char_start=meta.get("char_start", 0),
                        char_end=meta.get("char_end", 0),
                        section_title=meta.get("section_title"),
                        section_path=meta.get("section_path"),
                        section_level=meta.get("section_level", 0),
                        section_depth=meta.get("section_depth", 0),
                    )
                )
            except (KeyError, TypeError) as e:
                raise ValueError(
                    f"Phase 3 {layer_name} chunk {idx} reconstruction failed in {json_path}: {e}"
                )

        # Reconstruct chunks for all 3 layers with per-chunk error handling
        chunks = {}
        for layer_name in ["layer1", "layer2", "layer3"]:
            chunks[layer_name] = [
                reconstruct_chunk(c, layer_name, idx)
                for idx, c in enumerate(data[layer_name])
            ]

        # Calculate total_chunks (required by get_chunking_stats)
        chunks["total_chunks"] = len(chunks["layer1"]) + len(chunks["layer2"]) + len(chunks["layer3"])

        logger.info(
            f"Loaded Phase 3: "
            f"L1={len(chunks['layer1'])}, "
            f"L2={len(chunks['layer2'])}, "
            f"L3={len(chunks['layer3'])} chunks"
        )

        return chunks
