"""
ToC Retrieval Pipeline - LLM-based Table of Contents Extraction

This module provides an ALTERNATIVE extraction method for documents where:
1. PDF embedded outline/bookmarks are missing (Tier 1)
2. Visual ToC pages exist but aren't in metadata (Tier 2)

WHEN TO USE:
- Use this for documents where unstructured_extractor.py fails to extract structure
- Cost: ~$0.003 per document (Gemini 2.5 Flash)
- Supports: PDF (with potential for .tex, .txt extension)

INTEGRATION POINT:
- This is NOT part of the main indexing pipeline (run_pipeline.py)
- Use as a pre-processing step or fallback for structure extraction
- Output HierarchyNode can be converted to DoclingDocument format

STATUS CODES:
- "TIER_1_SUCCESS": PDF has embedded outline/bookmarks
- "TIER_2_SUCCESS": TOC found via LLM heuristic analysis
- "TIER_2_FAILURE": Heuristic failed to find TOC header
- "ERROR_DOC_OPEN": Failed to open PDF document
- "ERROR_AGENT_INIT": LLM agent initialization failed
"""

# ==============================================================================
# 0. IMPORTS AND HELPERS
# ==============================================================================
import os
import re
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any, Type
from dotenv import load_dotenv

import fitz
import google.generativeai as genai
from pydantic import BaseModel, Field

from src.cost_tracker import PRICING

logger = logging.getLogger(__name__)






# ==============================================================================
# 1. DATA CLASSES AND ARCHITECTURE
# ==============================================================================

HeadingData = Tuple[str, int, int] # (title, level, page_number)

# --- Abstract Base Class (Contract) ---
class BaseDocumentParser(ABC):
    def __init__(self, file_path: str):
        self.file_path: str = file_path

    @abstractmethod
    def get_document_type(self) -> str:
        pass

    # Returns triple: (List of headings, Raw OCR text, total cost)
    @abstractmethod
    def extract_structured_headings(self) -> Tuple[List[HeadingData], Optional[str], float]:
        pass

class HierarchyNode:
    """Represents a node in hierarchical tree structure."""
    def __init__(self, title: str, level: int, page_number: Optional[int] = None):
        self.title: str = title
        self.level: int = level 
        self.page_number: Optional[int] = page_number
        self.children: List['HierarchyNode'] = []

    def add_child(self, child_node: 'HierarchyNode'):
        self.children.append(child_node)

    def __repr__(self) -> str:
        return f"Node(title='{self.title[:30]}...', level={self.level}, page={self.page_number}, children={len(self.children)})"


class HierarchyBuilder:
    """Builds hierarchical tree from flat list of headings using stack."""

    def __init__(self):
        self.ROOT_TITLE = "Document Root"
        self.ROOT_LEVEL = 0
        self.ROOT_PAGE = 1

    def build_tree(self, headings: List[HeadingData]) -> Optional['HierarchyNode']:
        """Constructs tree from flat heading list."""
        if not headings:
            return None

        root = HierarchyNode(self.ROOT_TITLE, self.ROOT_LEVEL, self.ROOT_PAGE)
        # Stack holds (level, node)
        node_stack: List[Tuple[int, 'HierarchyNode']] = [(self.ROOT_LEVEL, root)]

        for title, level, page_num in headings:
            new_node = HierarchyNode(title, level, page_num)
            
            # Pop zřetězení dokud nenajdeme správného rodiče
            while node_stack and level <= node_stack[-1][0]:
                node_stack.pop()
            
            if node_stack:
                node_stack[-1][1].add_child(new_node)
                node_stack.append((level, new_node))

        return root

# --- Manuální Definice Schémat pro Gemini API ---

# Schéma pro Fázi 1 (Hledání první kapitoly)
FIRST_CHAPTER_SCHEMA_DICT = {
    "type": "object",
    "properties": {
        "first_chapter_page": {
            "type": "integer",
            "description": "Číslo stránky (1-based), kde začíná první hlavní kapitola/sekce (např. '1. Úvod' nebo 'Kapitola I')."
        }
    },
    "required": ["first_chapter_page"]
}

# Schéma pro Fázi 2 (Kompletní struktura)
# Všimněte si, jak je 'HeadingItem' definován přímo uvnitř 'items'
FULL_STRUCTURE_SCHEMA_DICT = {
    "type": "object",
    "properties": {
        "headings": {
            "type": "array",
            "description": "Kompletní seznam všech hierarchických položek z obsahu.",
            "items": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Čistý název kapitoly nebo sekce."
                    },
                    "level": {
                        "type": "integer",
                        "description": "Odvozená hierarchická úroveň (1 pro nejvyšší, 2 pro podsekci atd.)."
                    },
                    "page_number": {
                        "type": "integer",
                        "description": "Číslo stránky, kde tato sekce začíná."
                    }
                },
                "required": ["title", "level", "page_number"]
            }
        }
    },
    "required": ["headings"]
}

class LLMAgent:
    """
    Zapouzdřuje volání LLM a nyní také sleduje náklady na tokeny.

    Pricing is centralized in cost_tracker.py (SSOT).
    """

    # Default pricing fallback (used when model not found in PRICING)
    _DEFAULT_PRICING = {"input": 0.50, "output": 1.50}

    def __init__(self, api_key: Optional[str] = None, model_name: str = "models/gemini-2.5-flash"):
        # Accept API key as parameter (preferably from centralized config)
        # Fallback to environment variable for backward compatibility
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("GOOGLE_API_KEY not provided. Pass api_key parameter or set GOOGLE_API_KEY environment variable.")

        genai.configure(api_key=api_key)

        self.model_name = model_name
        self.model = genai.GenerativeModel(self.model_name)

        # Get pricing from centralized PRICING dict (SSOT: cost_tracker.py)
        # Model name format: "models/gemini-2.5-flash" -> "gemini-2.5-flash"
        short_model_name = model_name.replace("models/", "")
        google_pricing = PRICING.get("google", {})
        self.pricing = google_pricing.get(short_model_name, self._DEFAULT_PRICING)

        logger.info(f"LLM Agent (Gemini) initialized with model: {self.model_name}")

    def _execute_json_prompt(self, prompt: str, schema_dict: Dict[str, Any]) -> Tuple[Optional[dict], float]:
        """Returns (json_result, calculated_cost)."""
        try:
            config = genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=schema_dict 
            )
            response = self.model.generate_content(prompt, generation_config=config)
            
            cost = 0.0
            if response.usage_metadata:
                usage = response.usage_metadata
                in_tokens = usage.prompt_token_count
                out_tokens = usage.candidates_token_count
                
                cost = ((in_tokens / 1_000_000) * self.pricing["input"]) + \
                       ((out_tokens / 1_000_000) * self.pricing["output"])

                logger.debug(f"LLM usage: Input {in_tokens} tokens, Output {out_tokens} tokens, Cost: ${cost:.6f}")

            return json.loads(response.text), cost

        except json.JSONDecodeError as e:
            logger.error(f"LLM returned invalid JSON: {e}", exc_info=True)
            raise RuntimeError("ToC extraction failed: LLM returned malformed response") from e
        except Exception as e:
            logger.error(f"Unexpected error in LLM execution: {e}", exc_info=True)
            raise RuntimeError(f"ToC extraction failed unexpectedly: {str(e)}") from e

    def find_first_chapter_page(self, toc_page_text: str) -> Tuple[Optional[int], float]:
        """Phase 1: Returns (page_number, cost)."""
        prompt = f"""
        Analyze the following text from the first page of the table of contents.
        Identify the first entry that appears in the contents. This marks the beginning of the first section of the document's actual content. Return the page number (1-based) where this section begins.
        The level of this section (chapter, subchapter, etc.) doesn't matter. Focus on finding the first content entry. Ignore entries like 'Contents', 'Obsah', or 'Table of Contents'.
        Return ONLY JSON object according to schema.

        TEXT FROM FIRST TOC PAGE:
        {toc_page_text[:4000]} 
        """ 
        
        result, cost = self._execute_json_prompt(prompt, FIRST_CHAPTER_SCHEMA_DICT)

        if result and 'first_chapter_page' in result:
            return int(result['first_chapter_page']), cost

        logger.warning("LLM Phase 1 failed to find 'first_chapter_page'")
        return None, cost

    def extract_full_structure(self, full_toc_text: str) -> Tuple[List[HeadingData], float]:
        """Phase 2: Returns (list_of_headings, cost)."""
        prompt = f"""
        Analyze complete table of contents (TOC) text from document.
        Ignore meta-entries like 'Contents' or 'List of Figures'.
        Extract complete hierarchical structure (chapters, sections, subsections).
        Infer level from numbering (e.g., 1.1 = level 2, A. = level 2) and indentation.
        Return ONLY JSON object according to schema.

        COMPLETE TOC TEXT:
        {full_toc_text}
        """
        
        result, cost = self._execute_json_prompt(prompt, FULL_STRUCTURE_SCHEMA_DICT)
        headings_list: List[HeadingData] = []
        
        if result and 'headings' in result:
            for item in result['headings']:
                headings_list.append(
                    (item['title'], item['level'], item['page_number'])
                )
        else:
            logger.warning("LLM Phase 2 failed to extract 'headings'")

        return headings_list, cost
# ==============================================================================
# 2. KONKRÉTNÍ PARSERY
# ==============================================================================

class PDFParser(BaseDocumentParser):
    
    def __init__(self, file_path: str, max_toc_pages: int = 10):
        super().__init__(file_path)
        # max_toc_pages nyní slouží jako limit pro hledání *začátku* TOC
        self.max_toc_pages_search: int = max_toc_pages
        try:
            self.llm_agent = LLMAgent()
        except ValueError as e:
            logger.error(f"Failed to initialize LLM agent: {e}", exc_info=True)
            logger.warning("ToC extraction via LLM will be unavailable. Set GOOGLE_API_KEY in .env to enable.")
            self.llm_agent = None

    def get_document_type(self) -> str:
        return "PDF"
        
    def parse_document(self) -> Dict[str, Any]:
        """Open PDF document and return metadata."""
        try:
            doc = fitz.open(self.file_path)
            return {"doc_object": doc, "page_count": doc.page_count}
        except fitz.FileDataError as e:
            logger.error(f"PDF file is corrupted or malformed: {self.file_path}", exc_info=True)
            raise RuntimeError(f"Cannot open PDF: file is corrupted or password-protected") from e
        except PermissionError as e:
            logger.error(f"Permission denied reading PDF: {self.file_path}", exc_info=True)
            raise RuntimeError(f"Cannot open PDF: permission denied") from e
        except Exception as e:
            logger.error(f"Unexpected error opening PDF {self.file_path}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to open PDF: {str(e)}") from e

    def find_toc_scope(self) -> Tuple[Optional[int], Optional[int], Optional[str], float, str]:
        """
        PHASE 1: Find start and end page indices of table of contents.
        Returns (start_index, end_index, first_page_text, cost, STATUS)
        """
        logger.info("PDFParser: Starting Phase 1 (ToC Scope Detection)")
        total_cost = 0.0
        data = self.parse_document()
        doc: fitz.Document = data.get("doc_object")

        if not doc:
            return None, None, None, 0.0, "ERROR_DOC_OPEN"
        if not self.llm_agent:
            doc.close()
            return None, None, None, 0.0, "ERROR_AGENT_INIT"

        # Tier 1 (Outline) has priority
        if doc.get_toc():
            logger.info("Document has Tier 1 Outline, Phase 1 skipped")
            doc.close()
            return None, None, None, 0.0, "TIER_1_SUCCESS"

        # Phase 1A: Detect ToC *start* (Heuristic)
        toc_start_page_index = -1
        first_page_text = ""
        for i in range(min(doc.page_count, self.max_toc_pages_search)):
            text = doc[i].get_text("text")
            if re.search(r'(table of contents|contents|obsah|seznam|content)', text[:500], re.IGNORECASE):
                toc_start_page_index = i
                first_page_text = text
                break

        if toc_start_page_index == -1:
            logger.warning("Phase 1: ToC start not found (Tier 2 Heuristic failed)")
            doc.close()
            return None, None, None, 0.0, "TIER_2_FAILURE"

        # Phase 1B: Detect ToC *end* (LLM call #1)
        logger.info("LLM Agent Phase 1: Finding ToC end page...")
        first_chapter_page, cost1 = self.llm_agent.find_first_chapter_page(first_page_text)
        total_cost += cost1

        toc_end_page_index: int
        if not first_chapter_page:
            logger.warning("LLM Phase 1 failed. Using fallback (single-page ToC)")
            toc_end_page_index = toc_start_page_index
        else:
            toc_end_page_index = first_chapter_page - 2  # 1-based page to 0-based index
            if toc_end_page_index < toc_start_page_index:
                toc_end_page_index = toc_start_page_index

        doc.close()
        logger.info(f"Phase 1: ToC scope defined - Pages {toc_start_page_index + 1} to {toc_end_page_index + 1}")
        return toc_start_page_index, toc_end_page_index, first_page_text, total_cost, "TIER_2_SUCCESS"

    def extract_structure_from_scope(self, toc_start_page_index: int, toc_end_page_index: int) -> Tuple[List[HeadingData], Optional[str], float]:
        """
        PHASE 2: Extract complete structure from given page range.
        Returns (headings, raw_text, cost).
        """
        logger.info("PDFParser: Starting Phase 2 (Structure Extraction)")
        data = self.parse_document()
        doc: fitz.Document = data.get("doc_object")
        if not doc or not self.llm_agent:
            if doc: doc.close()
            return [], None, 0.0

        # Phase 2A: Extract complete ToC text
        full_toc_text = ""
        for i in range(toc_start_page_index, min(toc_end_page_index + 1, doc.page_count)):
            full_toc_text += doc[i].get_text("text") + "\n--- Page Break ---\n"

        doc.close()

        # Phase 2B: Extract structure (LLM call #2)
        logger.info("LLM Agent Phase 2: Extracting complete structure...")
        structured_headings, cost2 = self.llm_agent.extract_full_structure(full_toc_text)

        return structured_headings, full_toc_text, cost2

    def extract_structured_headings(self) -> Tuple[List[HeadingData], Optional[str], float]:
        """
        Orchestration method (Phase 0) that calls Phase 1 and Phase 2.
        Returns (headings, raw_text, total_cost)
        """
        data = self.parse_document()
        doc: fitz.Document = data.get("doc_object")
        if not doc: 
            return [], None, 0.0

        # TIER 1 (Outline) has priority
        outline = doc.get_toc()
        if outline:
            logger.info("Structure extracted from PDF Outline/Bookmarks (Tier 1)")
            headings = [(title, level, page + 1) for level, title, page in outline]
            doc.close()
            return headings, None, 0.0  # Returns (data, text, cost)

        # Tier 1 failed, calling Phase 1
        doc.close()  # Close document, Phase 1 will reopen it
        toc_start, toc_end, _, cost1, status = self.find_toc_scope()

        # Check for explicit Phase 1 success
        if status != "TIER_2_SUCCESS":
            # This now covers TIER_1_SUCCESS (which shouldn't happen here)
            # and especially TIER_2_FAILURE
            return [], None, cost1

        # Call Phase 2
        headings, ocr_text, cost2 = self.extract_structure_from_scope(toc_start, toc_end)
        
        total_cost = cost1 + cost2
        return headings, ocr_text, total_cost
# ==============================================================================
# 3. ORCHESTRATOR MODULE AND TEST FRAMEWORK
# ==============================================================================

# Mapping for orchestrator module
PARSER_MAPPING: Dict[str, Type[BaseDocumentParser]] = {
    '.pdf': PDFParser
    # .tex and .txt would be here if implemented
}

class DocumentHierarchyTool:
    """
    Main orchestrator class (Facade/Factory). Properly propagates costs.
    """

    def __init__(self):
        self.builder = HierarchyBuilder()
        # Assumption: PARSER_MAPPING is defined globally or as attribute
        self.PARSER_MAPPING = PARSER_MAPPING

    def get_parser(self, file_path: str) -> Optional[BaseDocumentParser]:
        ext = os.path.splitext(file_path)[-1].lower()
        ParserClass = self.PARSER_MAPPING.get(ext)
        if ParserClass:
            return ParserClass(file_path)
        return None

    def process_document(self, file_path: str) -> Tuple[Optional['HierarchyNode'], Optional[str], float]:
        """
        Runs process and returns tree, raw OCR text, and TOTAL COST.
        """
        if not os.path.exists(file_path):
            return None, None, 0.0
            
        parser = self.get_parser(file_path)
        if not parser:
            return None, None, 0.0

        # Unpack 3 values: headings, OCR text, LLM costs
        structured_headings, ocr_text, total_cost = parser.extract_structured_headings()

        if not structured_headings:
            return None, ocr_text, total_cost

        document_tree = self.builder.build_tree(structured_headings)

        return document_tree, ocr_text, total_cost

# --- Helper function for visualization (should be defined globally elsewhere for context) ---
def visualize_tree_to_string(node: 'HierarchyNode', prefix: str = "", is_last: bool = True) -> List[str]:
    """Recursive function that visualizes tree into list of strings."""
    lines = []
    if node.level != 0:  # Don't display virtual root
        line = prefix + ("└── " if is_last else "├── ") + \
               f"[{node.level}] {node.title[:80]} (Page {node.page_number})"
        lines.append(line)
    
    child_count = len(node.children)
    next_prefix = prefix + ("    " if is_last else "│   ")
    
    for i, child in enumerate(node.children):
        is_last_child = i == child_count - 1
        lines.extend(visualize_tree_to_string(child, next_prefix, is_last_child))
            
    return lines

# --- Opravená Třída DocumentTestRunner ---

class DocumentTestRunner:
    
    def __init__(self, test_dir_path: str, output_dir_path: str = "test_results"):
        """
        Inicializuje Runner cestami a připraví výstupní složku.
        """
        self.test_dir = test_dir_path
        self.output_dir = output_dir_path
        self.tool = DocumentHierarchyTool() 
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"✅ TestRunner inicializován. Výstupy budou uloženy do: {self.output_dir}")

    def run_tests(self, phase: str = "full"):
        """
        Spouští testování v různých fázích.
        
        Args:
            phase (str): Mód testování: 'full', 'scope', 'structure'.
        """
        print(f"\n===== ZAHÁJENÍ TESTOVACÍHO BĚHU (FÁZE: {phase.upper()}) =====")
        
        if not os.path.isdir(self.test_dir):
            print(f"🛑 Chyba: Testovací složka nenalezena na cestě: {self.test_dir}")
            return
        
        supported_extensions = list(self.tool.PARSER_MAPPING.keys())

        for filename in sorted(os.listdir(self.test_dir)):
            
            file_path = os.path.join(self.test_dir, filename)

            if not os.path.isfile(file_path):
                continue # Přeskočíme složky

            ext = os.path.splitext(filename)[-1].lower()
            
            if ext not in supported_extensions:
                print(f"\n--- SKIPPING {filename}: Nepodporovaný typ ({ext}) ---")
                continue
            
            # --- ZÍSKÁNÍ PARSERU ---
            parser = self.tool.get_parser(file_path) 
            
            if not parser:
                print(f"--- SKIPPING {filename}: Nebyl nalezen parser ---")
                continue
                
            # Dvoufázové testování je relevantní pouze pro PDFParser
            if not isinstance(parser, PDFParser) and phase != "full":
                print(f"--- SKIPPING {filename}: Fázové testování je jen pro PDFParser ---")
                continue

            print(f"\n--- SPOUŠTÍM TEST: {filename} (Mód: {phase}) ---")

            # --- VÝBĚR FÁZE ---
            
            if phase == "scope":
                # FÁZE 1: POUZE HLEDÁNÍ ROZSAHU
                output_path = os.path.join(self.output_dir, f"{filename}_PHASE1_SCOPE.txt")
                
                # Rozbalíme 5 hodnot včetně 'status'
                toc_start, toc_end, first_page_text, cost1, status = parser.find_toc_scope()
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"VÝSLEDEK FÁZE 1 (HLEDÁNÍ ROZSAHU) PRO: {filename}\n")
                    f.write(f"ODHADOVANÁ CENA FÁZE 1: ${cost1:.8f} USD\n")
                    f.write("="*60 + "\n")

                    # --- NOVÁ LOGIKA PRO JASNÝ VÝSTUP ---
                    if status == "TIER_1_SUCCESS":
                        f.write("STATUS: ÚSPĚCH (TIER 1 - PDF OUTLINE)\n")
                        f.write("LLM Fáze 1 nebyla spuštěna, protože dokument obsahuje vestavěné PDF Záložky (Outline).\n")
                    
                    elif status == "TIER_2_FAILURE":
                        f.write("STATUS: SELHÁNÍ (TIER 2 - DETEKCE)\n")
                        f.write("LLM Fáze 1 nebyla spuštěna, protože heuristika nenašla titulek 'Obsah' na prvních stranách.\n")
                    
                    elif status == "TIER_2_SUCCESS":
                        f.write("STATUS: ÚSPĚCH (LLM FÁZE 1)\n")
                        f.write(f"Rozsah nalezen (0-based index): Strana {toc_start} až {toc_end}\n")
                        f.write("\n--- Text první strany (použitý pro LLM F1) ---\n")
                        f.write(first_page_text if first_page_text else "N/A")
                    
                    else: # např. ERROR_PARSER_INIT
                        f.write(f"STATUS: CHYBA ({status})\n")
                        f.write("Došlo k chybě při inicializaci parseru nebo agenta.\n")
                    # --- KONEC NOVÉ LOGIKY ---
                        
                print(f"✅ Fáze 1: Výsledek uložen do {output_path}")

            elif phase == "structure":
                # FÁZE 2: POUZE ANALÝZA STRUKTURY
                output_path = os.path.join(self.output_dir, f"{filename}_PHASE2_STRUCTURE.txt")
                
                print("   (Spouštím F1 pro získání rozsahu...)")
                toc_start, toc_end, _, cost1, status = parser.find_toc_scope()
                
                if status != "TIER_2_SUCCESS":
                    print(f"   F1 selhala (Status: {status}), F2 nelze spustit.")
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(f"FÁZE 2 PŘESKOČENA: Fáze 1 nenalezla rozsah TOC (Status: {status}).")
                    continue
                    
                document_tree, ocr_source_text, cost2 = parser.extract_structure_from_scope(toc_start, toc_end)
                total_cost = cost1 + cost2
                self._save_results(output_path, filename, "FÁZE 2 (STRUKTURA)", document_tree, ocr_source_text, total_cost)
            
            else: # "full" (default)
                # PLNÝ BĚH (přes DocumentHierarchyTool)
                output_path = os.path.join(self.output_dir, f"{filename}_FULL_RUN.txt")
                
                # Zde musíme aktualizovat DocumentHierarchyTool, aby vracel náklady
                # Prozatím předpokládáme, že self.tool.process_document() vrací 3 hodnoty
                document_tree, ocr_source_text, total_cost = self.tool.process_document(file_path) 
                self._save_results(output_path, filename, "PLNÝ BĚH (F1+F2)", document_tree, ocr_source_text, total_cost)

    def _save_results(self, output_path: str, filename: str, run_type: str, 
                      document_tree: Optional['HierarchyNode'], 
                      ocr_source_text: Optional[str], 
                      total_cost: float): # Přidán parametr total_cost
        """Pomocná metoda pro ukládání výsledků."""
        
        if document_tree:
            tree_lines = visualize_tree_to_string(document_tree, is_last=True)
            header = [
                "*" * 60,
                f"VÝSLEDEK BĚHU ({run_type}) PRO: {filename}",
                f"ODHADOVANÁ CELKOVÁ CENA: ${total_cost:.8f} USD", # Zobrazení ceny
                "*" * 60,
            ]
            
            if ocr_source_text:
                ocr_section = [
                    "\n" + "#" * 70,
                    "# SUROVÝ TEXT PŘEDANÝ LLM (FÁZE 2)",
                    "#" * 70,
                    ocr_source_text,
                    "\n" + "-" * 70,
                    "VÝSLEDEK PARSOVÁNÍ STRUKTURY:",
                ]
                header.extend(ocr_section)
            
            header.extend([f"KOŘEN STROMU: {document_tree.title}", "-" * 60])
            final_output = header + tree_lines
            
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(final_output))
                print(f"✅ {run_type}: Výsledek uložen do {output_path}")
            except Exception as e:
                print(f"❌ Chyba při ukládání souboru {output_path}: {e}")

        else:
            log_content = f"🛑 Zpracování ({run_type}) pro {filename} selhalo.\n"
            log_content += f"Celkové náklady (před selháním): ${total_cost:.8f} USD\n"
            if ocr_source_text:
                log_content += "\n--- SUROVÝ TEXT PŘEDANÝ LLM ---\n" + ocr_source_text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(log_content)
            print(f"🛑 {run_type}: Zpracování selhalo, uložen log do: {output_path}")


if __name__ == "__main__":
    """
    Example usage - customize paths for your environment.

    Usage:
        python src/ToC_retrieval.py [test_dir] [output_dir] [phase]

    Args:
        test_dir: Directory containing PDF files to test (default: "test_data/")
        output_dir: Directory for output files (default: "test_results/")
        phase: Test phase - "full", "scope", or "structure" (default: "full")
    """
    import sys

    # Parse command-line arguments or use defaults
    test_path = sys.argv[1] if len(sys.argv) > 1 else "test_data/"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "test_results/"
    phase = sys.argv[3] if len(sys.argv) > 3 else "full"

    # Run tests
    testing = DocumentTestRunner(test_dir_path=test_path, output_dir_path=output_path)
    testing.run_tests(phase=phase)
