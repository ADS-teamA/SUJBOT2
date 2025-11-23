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

logger = logging.getLogger(__name__)






# ==============================================================================
# 1. TŘÍDY DAT A ARCHITEKTURY
# ==============================================================================

HeadingData = Tuple[str, int, int] # (title, level, page_number)

# --- Abstraktní Třída (Kontrakt) ---
class BaseDocumentParser(ABC):
    def __init__(self, file_path: str):
        self.file_path: str = file_path
        
    @abstractmethod
    def get_document_type(self) -> str:
        pass

    # Všimněte si, že vracíme TROJICI: (List nadpisů, Surový OCR text, celková_cena)
    @abstractmethod
    def extract_structured_headings(self) -> Tuple[List[HeadingData], Optional[str], float]:
        pass

class HierarchyNode:
    """Reprezentuje jeden uzel v hierarchické stromové struktuře."""
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
    """Sestavuje hierarchický strom z plochého seznamu nadpisů pomocí zásobníku."""
    
    def __init__(self):
        self.ROOT_TITLE = "Document Root"
        self.ROOT_LEVEL = 0
        self.ROOT_PAGE = 1
        
    def build_tree(self, headings: List[HeadingData]) -> Optional['HierarchyNode']:
        """Konstruuje strom."""
        if not headings:
            return None

        root = HierarchyNode(self.ROOT_TITLE, self.ROOT_LEVEL, self.ROOT_PAGE)
        # Zásobník drží (úroveň, uzel)
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

    NOTE: Pricing accurate as of 2025-01. Update according to https://ai.google.dev/pricing
    """

    MODEL_PRICING = {
        "models/gemini-2.5-flash": {
            "input": 0.30,  # USD per 1M tokens
            "output": 0.60
        },
        "default": {
            "input": 0.50,
            "output": 1.50
        }
    }

    def __init__(self):
        # Load API key from .env file
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("Error: 'GOOGLE_API_KEY' not set in .env file. Set it to use ToC extraction via Gemini LLM.")

        genai.configure(api_key=api_key)

        self.model_name = "models/gemini-2.5-flash"
        self.model = genai.GenerativeModel(self.model_name)
        self.pricing = self.MODEL_PRICING.get(self.model_name, self.MODEL_PRICING["default"])

        logger.info(f"LLM Agent (Gemini) initialized with model: {self.model_name}")

    def _execute_json_prompt(self, prompt: str, schema_dict: Dict[str, Any]) -> Tuple[Optional[dict], float]:
        """Vrací (výsledek_json, vypočtená_cena)."""
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
        """Fáze 1: Nyní vrací (číslo_stránky, cena)."""
        prompt = f"""
        Analyzuj následující text první stránky obsahu.
        Identifikuj první záznam, který se v obsahu nachází. Ten označuje začátek první sekce skutečného obsahu dokumentu. Vrať číslo stránky (1-based), kde tato sekce začíná.
        Nezáleží na tom, jakou má tato sekce úroveň (kapitola, podkapitola atd.). Řiď se tím, že by to měl být první záznam. Ignoruj položku jako 'Obsah', 'Contents'nebo 'Table of Contents'. 
        Vrať POUZE JSON objekt podle schématu.

        TEXT PRVNÍ STRÁNKY OBSAHU:
        {toc_page_text[:4000]} 
        """ 
        
        result, cost = self._execute_json_prompt(prompt, FIRST_CHAPTER_SCHEMA_DICT)

        if result and 'first_chapter_page' in result:
            return int(result['first_chapter_page']), cost

        print("⚠️ LLM (Fáze 1) selhal při hledání 'first_chapter_page'.")
        return None, cost

    def extract_full_structure(self, full_toc_text: str) -> Tuple[List[HeadingData], float]:
        """Fáze 2: Nyní vrací (seznam_nadpisů, cena)."""
        prompt = f"""
        Analyzuj kompletní text obsahu (TOC) dokumentu. 
        Ignoruj položky jako 'Obsah' nebo 'Seznam obrázků'.
        Extrahuj kompletní hierarchickou strukturu (kapitoly, sekce, podsekce).
        Odvoď úroveň (level) z číslování (např. 1.1 = level 2, A. = level 2) a odsazení.
        Vrať POUZE JSON objekt podle schématu.

        KOMPLETNÍ TEXT OBSAHU:
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
             print("⚠️ LLM (Fáze 2) selhal při extrakci 'headings'.")

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
            print(f"🛑 {e}")
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
        FÁZE 1: Najde počáteční a koncový index stránky obsahu.
        Vrací (start_index, end_index, text_první_stránky, cena, STATUS)
        """
        print("--- PDFParser: Spouštím Fázi 1 (Hledání Rozsahu TOC) ---")
        total_cost = 0.0
        data = self.parse_document()
        doc: fitz.Document = data.get("doc_object")
        
        if not doc:
            return None, None, None, 0.0, "ERROR_DOC_OPEN"
        if not self.llm_agent:
            doc.close()
            return None, None, None, 0.0, "ERROR_AGENT_INIT"

        # Tier 1 (Outline) má přednost
        if doc.get_toc():
            print("INFO: Dokument má Tier 1 Outline, Fáze 1 se přeskakuje.")
            doc.close()
            # Vracíme nový stavový kód
            return None, None, None, 0.0, "TIER_1_SUCCESS" 

        # Fáze 1A: Detekce *začátku* TOC (Heuristika)
        toc_start_page_index = -1
        first_page_text = ""
        for i in range(min(doc.page_count, self.max_toc_pages_search)):
            text = doc[i].get_text("text")
            if re.search(r'(table of contents|contents|obsah|seznam|content)', text[:500], re.IGNORECASE):
                toc_start_page_index = i
                first_page_text = text
                break
        
        if toc_start_page_index == -1:
            print("⚠️ Fáze 1: Začátek TOC nenalezen (Tier 2 Heuristika selhala).")
            doc.close()
            # Vracíme nový stavový kód
            return None, None, None, 0.0, "TIER_2_FAILURE"

        # Fáze 1B: Detekce *konce* TOC (Volání LLM č. 1)
        print("🤖 LLM Agent (Fáze 1): Hledám konec TOC...")
        first_chapter_page, cost1 = self.llm_agent.find_first_chapter_page(first_page_text)
        total_cost += cost1
        
        toc_end_page_index: int
        if not first_chapter_page:
            print("⚠️ LLM (Fáze 1) selhal. Používám fallback (pouze 1 stránka TOC).")
            toc_end_page_index = toc_start_page_index
        else:
            toc_end_page_index = first_chapter_page - 2 # 1-based stranu na 0-based index
            if toc_end_page_index < toc_start_page_index:
                toc_end_page_index = toc_start_page_index
        
        doc.close()
        print(f"✅ Fáze 1: Rozsah TOC definován: Strany {toc_start_page_index + 1} až {toc_end_page_index + 1}.")
        # Vracíme nový stavový kód
        return toc_start_page_index, toc_end_page_index, first_page_text, total_cost, "TIER_2_SUCCESS"

    def extract_structure_from_scope(self, toc_start_page_index: int, toc_end_page_index: int) -> Tuple[List[HeadingData], Optional[str], float]:
        """
        FÁZE 2: Extrahuje kompletní strukturu z daného rozsahu stránek.
        Vrací (nadpisy, surový_text, cena).
        """
        print("--- PDFParser: Spouštím Fázi 2 (Extrakce Struktury) ---")
        data = self.parse_document()
        doc: fitz.Document = data.get("doc_object")
        if not doc or not self.llm_agent:
            if doc: doc.close()
            return [], None, 0.0
            
        # Fáze 2A: Extrakce kompletního textu TOC
        full_toc_text = ""
        for i in range(toc_start_page_index, min(toc_end_page_index + 1, doc.page_count)):
            full_toc_text += doc[i].get_text("text") + "\n--- Page Break ---\n"
        
        doc.close()

        # Fáze 2B: Extrakce struktury (Volání LLM č. 2)
        print("🤖 LLM Agent (Fáze 2): Extrahuje kompletní strukturu...")
        structured_headings, cost2 = self.llm_agent.extract_full_structure(full_toc_text)
             
        return structured_headings, full_toc_text, cost2

    def extract_structured_headings(self) -> Tuple[List[HeadingData], Optional[str], float]:
        """
        Orchestrační metoda (Fáze 0), která volá F1 i F2.
        Vrací (nadpisy, surový_text, celková_cena)
        """
        data = self.parse_document()
        doc: fitz.Document = data.get("doc_object")
        if not doc: 
            return [], None, 0.0

        # TIER 1 (Outline) má stále přednost
        outline = doc.get_toc()
        if outline:
            print("✅ Struktura Extrahována z PDF Outline/Bookmarks (Tier 1).")
            headings = [(title, level, page + 1) for level, title, page in outline]
            doc.close()
            return headings, None, 0.0 # Vracíme (data, text, cena)

        # Tier 1 selhal, voláme Fázi 1
        doc.close() # Zavřeme dokument, Fáze 1 si ho otevře znovu
        toc_start, toc_end, _, cost1, status = self.find_toc_scope()
        
        # Kontrolujeme explicitní úspěch Fáze 1
        if status != "TIER_2_SUCCESS":
            # Toto nyní pokryje TIER_1_SUCCESS (který by se zde neměl stát)
            # a hlavně TIER_2_FAILURE
            return [], None, cost1 

        # Voláme Fázi 2
        headings, ocr_text, cost2 = self.extract_structure_from_scope(toc_start, toc_end)
        
        total_cost = cost1 + cost2
        return headings, ocr_text, total_cost
# ==============================================================================
# 3. ŘÍDICÍ MODUL A TESTOVACÍ RÁMEC
# ==============================================================================

# Mapování pro řídicí modul
PARSER_MAPPING: Dict[str, Type[BaseDocumentParser]] = {
    '.pdf': PDFParser
    # .tex a .txt by zde byly, pokud by byly implementovány
}

class DocumentHierarchyTool:
    """
    Hlavní řídicí třída (Facade/Factory). Nyní správně propaguje náklady.
    """
    
    def __init__(self):
        self.builder = HierarchyBuilder()
        # Předpoklad: PARSER_MAPPING je definován globálně nebo jako atribut
        self.PARSER_MAPPING = PARSER_MAPPING 

    def get_parser(self, file_path: str) -> Optional[BaseDocumentParser]:
        ext = os.path.splitext(file_path)[-1].lower()
        ParserClass = self.PARSER_MAPPING.get(ext)
        if ParserClass:
            return ParserClass(file_path)
        return None

    def process_document(self, file_path: str) -> Tuple[Optional['HierarchyNode'], Optional[str], float]:
        """
        Spouští proces a vrací strom, surový OCR text a CELKOVOU CENU.
        """
        if not os.path.exists(file_path):
            return None, None, 0.0
            
        parser = self.get_parser(file_path)
        if not parser:
            return None, None, 0.0

        # Rozbalíme 3 hodnoty: nadpisy, OCR text, náklady na LLM
        structured_headings, ocr_text, total_cost = parser.extract_structured_headings()

        if not structured_headings:
            return None, ocr_text, total_cost

        document_tree = self.builder.build_tree(structured_headings)

        return document_tree, ocr_text, total_cost

# --- Pomocná funkce pro vizualizaci (pro kontext, měla by být definována globálně/jinde) ---
def visualize_tree_to_string(node: 'HierarchyNode', prefix: str = "", is_last: bool = True) -> List[str]:
    """Rekurzivní funkce, která vizualizuje strom do seznamu řetězců."""
    lines = []
    if node.level != 0: # Nezobrazujeme virtuální kořen
        line = prefix + ("└── " if is_last else "├── ") + \
               f"[{node.level}] {node.title[:80]} (Strana {node.page_number})"
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
