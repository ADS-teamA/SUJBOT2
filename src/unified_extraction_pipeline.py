import logging
import json
import time
import difflib
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# Import from your existing modules
from src.unstructured_extractor import (
    UnstructuredExtractor, 
    ExtractionConfig, 
    ExtractedDocument, 
    DocumentSection,
    TableData, 
    _normalize_text_diacritics
)
from src.ToC_retrieval import PDFParser, HierarchyNode
from unstructured.documents.elements import Element

logger = logging.getLogger(__name__)

class UnifiedDocumentPipeline:
    """
    Orchestrator pipeline that combines LLM-based ToC retrieval with 
    Unstructured.io text processing.
    
    Logic Flow:
    1. Try to retrieve structure via Metadata (Tier 1) or LLM Agent (Tier 2) using PDFParser.
    2. IF structure found:
       - Extract raw content elements using Unstructured.
       - Map raw elements to the retrieved structure using linear scanning and fuzzy matching.
    3. IF structure NOT found:
       - Fallback to standard Unstructured extraction (visual hierarchy inference).
    """

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig.from_env()
        # We instantiate the extractors
        self.unstructured_extractor = UnstructuredExtractor(self.config)

    def process_document(self, file_path: Path) -> ExtractedDocument:
        """
        Main entry point for document processing.
        """
        logger.info(f"--- START PIPELINE: Processing {file_path.name} ---")
        start_time = time.time()
        
        # 1. Attempt Structure Retrieval (Metadata or LLM)
        # Only applicable for PDFs currently
        structure_root: Optional[HierarchyNode] = None
        
        if file_path.suffix.lower() == ".pdf":
            try:
                logger.info("Attempting Tier 1/2 Structure Retrieval...")
                # We don't instantiate PDFParser in __init__ because it requires file_path
                toc_parser = PDFParser(str(file_path))
                
                # We get the tree, raw OCR text (unused here), and cost
                # Note: Ensure PDFParser.process_document returns 3 values as per previous updates
                structure_root, _, cost = self._try_retrieve_structure(toc_parser)
                
                if structure_root:
                    logger.info(f"✅ Structure retrieved successfully. Cost: ${cost:.6f}")
                else:
                    logger.warning("⚠️ Structure retrieval returned None (No outline, heuristic failed).")

            except Exception as e:
                logger.error(f"Structure retrieval failed with error: {e}")
                structure_root = None

        # 2. Decision Fork
        if structure_root:
            # PATH A: We have a High-Quality Structure
            # We need to fetch content using Unstructured and map it to this structure
            logger.info("🚀 PATH A: Using Retrieved ToC Structure + Unstructured Content")
            return self._process_with_external_structure(
                file_path, 
                structure_root, 
                start_time
            )
        else:
            # PATH B: Fallback to Standard Unstructured
            # This infers hierarchy based on font sizes, etc.
            logger.info("🛡️ PATH B: Fallback to Unstructured Native Hierarchy")
            return self.unstructured_extractor.extract(file_path)

    def _try_retrieve_structure(self, parser: PDFParser) -> tuple[Optional[HierarchyNode], Optional[str], float]:
        """
        Wrapper to call the PDFParser logic directly.
        """
        # 1. Extract headings directly from the parser
        # Returns: (headings_list, ocr_text, cost)
        structured_headings, ocr_text, total_cost = parser.extract_structured_headings()

        if not structured_headings:
            return None, ocr_text, total_cost

        # 2. Build the tree structure manually here
        # We need to import HierarchyBuilder locally or use the one from imports
        from src.ToC_retrieval import HierarchyBuilder
        
        builder = HierarchyBuilder()
        document_tree = builder.build_tree(structured_headings)
        
        return document_tree, ocr_text, total_cost

    def _process_with_external_structure(
        self, 
        file_path: Path, 
        root_node: HierarchyNode, 
        start_time: float
    ) -> ExtractedDocument:
        """
        Hybrid processing: 
        1. Use Unstructured to get raw text elements (partitioning).
        2. Use the HierarchyNode tree to define sections.
        3. Map elements to sections using linear scanning.
        """
        # 1. Get Raw Elements (using Unstructured partition logic directly)
        # We access the internal method to get raw elements without building default hierarchy
        elements: List[Element] = self.unstructured_extractor._partition_document(file_path)
        
        # Filter rotated text if configured
        if self.config.filter_rotated_text:
            from src.unstructured_extractor import filter_rotated_elements
            elements = filter_rotated_elements(
                elements, 
                self.config.rotation_min_angle, 
                self.config.rotation_max_angle
            )

        # 2. Flatten the Hierarchy Tree for sequential processing
        flat_structure = self._flatten_hierarchy(root_node)
        
        # 3. Map Elements to Sections (Linear Scan Logic)
        sections = self._map_elements_to_structure(elements, flat_structure)
        
        # 4. Extract Tables (standard unstructured logic)
        tables = self.unstructured_extractor._extract_tables(elements)

        # 5. Construct Final Object
        full_text = "\n\n".join(s.content for s in sections)
        
        return ExtractedDocument(
            document_id=file_path.stem,
            source_path=str(file_path),
            extraction_time=time.time() - start_time,
            full_text=full_text,
            markdown=self.unstructured_extractor._generate_markdown(sections),
            json_content="", # populated on serialization
            sections=sections,
            hierarchy_depth=max((s.depth for s in sections), default=0),
            num_roots=sum(1 for s in sections if s.level == 1),
            tables=tables,
            num_pages=self.unstructured_extractor._count_pages(elements),
            num_sections=len(sections),
            num_tables=len(tables),
            total_chars=len(full_text),
            title=root_node.title if root_node.title != "Document Root" else file_path.stem,
            extraction_method="hybrid_toc_pipeline",
            config=self.config.__dict__
        )

    def _flatten_hierarchy(self, root: HierarchyNode) -> List[Dict[str, Any]]:
        """
        Flattens the recursive HierarchyNode tree into a list.
        Calculates ancestors and paths.
        """
        flat_list = []

        def traverse(node: HierarchyNode, ancestors: List[str], depth: int):
            # Skip the virtual root node (level 0) unless it's the only thing
            is_virtual_root = (node.level == 0 and node.title == "Document Root")
            
            current_ancestors = ancestors[:]
            
            if not is_virtual_root:
                flat_node = {
                    "title": node.title,
                    "level": node.level,
                    "page_number": node.page_number,
                    "ancestors": current_ancestors,
                    "depth": depth,
                    # Generate IDs based on traversal order
                    "temp_id": f"sec_{len(flat_list) + 1}"
                }
                flat_list.append(flat_node)
                # Add self to ancestors for children
                current_ancestors.append(node.title)
                next_depth = depth + 1
            else:
                next_depth = depth # Don't increment depth for virtual root

            for child in node.children:
                traverse(child, current_ancestors, next_depth)

        traverse(root, [], 0)
        
        # Sort by page number to ensure logical flow
        flat_list.sort(key=lambda x: x["page_number"])
        
        return flat_list

    def _calculate_page_font_stats(self, elements: List[Element]) -> Dict[int, float]:
        """
        Calculates the median font size for each page to establish a baseline.
        """
        page_fonts: Dict[int, List[float]] = {}
        
        for elem in elements:
            page = getattr(elem.metadata, 'page_number', 1)
            
            # Collect font sizes from metadata
            font_sizes = []
            if hasattr(elem.metadata, 'font_size') and elem.metadata.font_size:
                font_sizes.append(elem.metadata.font_size)
            
            if font_sizes:
                if page not in page_fonts:
                    page_fonts[page] = []
                page_fonts[page].extend(font_sizes)
        
        # Calculate median
        page_medians = {}
        for page, sizes in page_fonts.items():
            if sizes:
                page_medians[page] = float(np.median(sizes))
            else:
                page_medians[page] = 10.0 # Default assumption
                
        return page_medians

    def _is_match(self, element: Element, target_title: str, page_median_size: float) -> bool:
        """
        Determines if an element corresponds to the target section title.
        Uses Text Similarity AND (Optional) Font Heuristics.
        """
        elem_text = _normalize_text_diacritics(str(element)).strip()
        target_title = _normalize_text_diacritics(target_title).strip()
        
        if not elem_text or not target_title:
            return False

        # 1. Text Similarity Check (Fuzzy Matching)
        # Ratio > 0.85 allows for minor OCR typos
        similarity = difflib.SequenceMatcher(None, elem_text.lower(), target_title.lower()).ratio()
        
        # Also check straightforward inclusion (title in ToC is often shorter/cleaner)
        is_substring = target_title.lower() in elem_text.lower() or elem_text.lower() in target_title.lower()
        
        # Require high similarity OR substring match with sufficient length
        text_match = similarity > 0.85 or (is_substring and len(elem_text) > 5)

        if not text_match:
            return False

        # 2. Font Size Heuristic (Secondary Confirmation)
        # If text matches, check if it looks like a header (larger than median or categorized as Title)
        is_visually_prominent = True # Default to true if we lack font info
        
        elem_size = getattr(element.metadata, 'font_size', None)
        elem_cat = getattr(element, 'category', '')

        if elem_cat == "Title":
             return True

        if elem_size and page_median_size > 0:
            # Check if it's not significantly smaller than median (allowing 5% buffer)
            if elem_size < (page_median_size * 0.95):
                 is_visually_prominent = False
        
        return is_visually_prominent

    def _map_elements_to_structure(
        self, 
        elements: List[Element], 
        structure: List[Dict[str, Any]]
    ) -> List[DocumentSection]:
        """
        Advanced Logic: Assigns unstructured text elements to the LLM-derived structure.
        
        Strategy:
        1. Treat 'elements' as a linear stream.
        2. For each Section in 'structure', find its START INDEX in 'elements'.
           - Look on the specific page predicted by the LLM.
           - Look for literal text match of the title using fuzzy logic.
        3. Section content is everything from [Start Index] up to [Next Section Start Index].
        """
        if not structure:
            return []

        # 1. Pre-calculate font statistics per page
        page_font_stats = self._calculate_page_font_stats(elements)

        # 2. Find Anchor Indices
        # anchors = list of (element_index, section_metadata)
        anchors: List[Tuple[int, Dict[str, Any]]] = []
        
        current_elem_idx = 0
        total_elements = len(elements)

        for i, sec_meta in enumerate(structure):
            target_page = sec_meta["page_number"]
            found_idx = -1
            
            # Search loop: Scan from current cursor forward
            for j in range(current_elem_idx, total_elements):
                elem = elements[j]
                elem_page = getattr(elem.metadata, 'page_number', 1)
                
                # Stop looking if we've gone past the target page by more than 2 pages
                if elem_page > target_page + 2:
                    break
                
                # Check match if we are on or past the target page
                if elem_page >= target_page:
                    if self._is_match(elem, sec_meta["title"], page_font_stats.get(elem_page, 0)):
                        found_idx = j
                        break
            
            if found_idx != -1:
                # Found a specific header match
                anchors.append((found_idx, sec_meta))
                current_elem_idx = found_idx 
                logger.debug(f"Mapped '{sec_meta['title']}' to element {found_idx} on page {target_page}")
            else:
                # Fallback: Default to the first element found on the target page
                fallback_idx = -1
                for j in range(current_elem_idx, total_elements):
                    elem = elements[j]
                    elem_page = getattr(elem.metadata, 'page_number', 1)
                    if elem_page == target_page:
                        fallback_idx = j
                        break
                
                if fallback_idx != -1:
                    anchors.append((fallback_idx, sec_meta))
                    current_elem_idx = fallback_idx
                    logger.warning(f"Title match failed for '{sec_meta['title']}'. Fallback to first element on page {target_page}.")
                else:
                    # Total failure: Append to current cursor (empty section)
                    logger.warning(f"Could not locate start of section '{sec_meta['title']}' on page {target_page}.")
                    anchors.append((current_elem_idx, sec_meta))

        # 3. Slice content based on Anchors
        sections_out: List[DocumentSection] = []
        char_offset = 0
        
        for i in range(len(anchors)):
            start_idx, sec_meta = anchors[i]
            
            # End index is the start of the next section, or end of document
            if i + 1 < len(anchors):
                end_idx, _ = anchors[i+1]
            else:
                end_idx = total_elements

            # Safety check
            if start_idx > end_idx:
                end_idx = start_idx 

            # Extract elements slice
            section_elements = elements[start_idx:end_idx]
            
            # Convert elements to text
            content_parts = []
            for elem in section_elements:
                txt = _normalize_text_diacritics(str(elem)).strip()
                if txt:
                    content_parts.append(txt)
            
            content_text = "\n\n".join(content_parts)

            # Resolve IDs and relationships
            sec_id = sec_meta["temp_id"]
            
            # Determine Parent ID (based on flattened structure)
            parent_id = None
            for prev_idx in range(i - 1, -1, -1):
                if structure[prev_idx]["level"] < sec_meta["level"]:
                    parent_id = structure[prev_idx]["temp_id"]
                    break
            
            # Children IDs
            children_ids = []
            for next_idx in range(i + 1, len(structure)):
                if structure[next_idx]["level"] <= sec_meta["level"]:
                    break 
                if structure[next_idx]["level"] == sec_meta["level"] + 1:
                    children_ids.append(structure[next_idx]["temp_id"])

            section = DocumentSection(
                section_id=sec_id,
                title=sec_meta["title"],
                content=content_text,
                level=sec_meta["level"],
                depth=sec_meta["depth"],
                parent_id=parent_id,
                children_ids=children_ids,
                ancestors=sec_meta["ancestors"],
                path=" > ".join(sec_meta["ancestors"] + [sec_meta["title"]]),
                page_number=sec_meta["page_number"],
                char_start=char_offset,
                char_end=char_offset + len(content_text),
                content_length=len(content_text),
                element_category="Composite"
            )
            
            sections_out.append(section)
            char_offset += len(content_text) + 2

        return sections_out

def test_single_document(file_path_str: str, output_dir: str = "test_results_pipeline"):
    """
    Runs the UnifiedDocumentPipeline on a single file and dumps the 
    full dictionary output to JSON for validation.
    """
    file_path = Path(file_path_str)
    
    if not file_path.exists():
        print(f"❌ Error: File not found at {file_path}")
        return

    # Ensure output directory exists
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)

    print(f"🚀 Starting Pipeline Test for: {file_path.name}")
    
    try:
        # Initialize Pipeline
        pipeline = UnifiedDocumentPipeline()
        
        start_time = time.time()
        
        # Run Processing
        extracted_doc = pipeline.process_document(file_path)
        
        duration = time.time() - start_time
        
        # Convert to Dictionary
        output_dict = extracted_doc.to_dict()
        
        # Construct Output Filename
        json_filename = f"{file_path.stem}_output.json"
        json_path = out_path / json_filename
        
        # Save to JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Success! Processed in {duration:.2f}s")
        print(f"📄 Extraction Method: {extracted_doc.extraction_method}")
        print(f"📊 Sections Found: {extracted_doc.num_sections}")
        print(f"💾 Output saved to: {json_path}")
        
        # Print a small preview of the structure for immediate feedback
        print("\n--- Structure Preview (First 5 Sections) ---")
        for sec in extracted_doc.sections[:5]:
            print(f"[{sec.level}] {sec.title} (Page {sec.page_number})")
            
    except Exception as e:
        print(f"🛑 Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()

test_single_document(file_path_str=r"C:\Users\Majitel\Desktop\VŠ\FJFI\NMS\ADS\testing\cez-zour-cz-2022.pdf",output_dir=r"C:\Users\Majitel\Desktop\VŠ\FJFI\NMS\ADS\vysledky")