#!/usr/bin/env python3
"""
Document Analyzer - Hlavní aplikace pro analýzu dokumentů pomocí Claude Code SDK
Paralelní zpracování rozsáhlých dokumentů s využitím subagentů
"""

import asyncio
import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
import hashlib
from dotenv import load_dotenv

try:
    from claude_code_sdk import ClaudeSDKClient, ClaudeCodeOptions
except ImportError:
    # Použít wrapper pokud claude_code_sdk není dostupný
    try:
        from .claude_sdk_wrapper import ClaudeSDKClient, ClaudeCodeOptions
    except ImportError:
        from claude_sdk_wrapper import ClaudeSDKClient, ClaudeCodeOptions

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

# Importy - zkusí relativní, pak absolutní
try:
    from .document_reader import DocumentReader
    from .question_parser import QuestionParser
    from .result_aggregator import ResultAggregator
    from .prompt_manager import PromptManager
except ImportError:
    from document_reader import DocumentReader
    from question_parser import QuestionParser
    from result_aggregator import ResultAggregator
    from prompt_manager import PromptManager

# Načtení konfigurace z .env
load_dotenv()

console = Console()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Konfigurace aplikace z environment variables"""
    api_key: str = field(default_factory=lambda: os.getenv('CLAUDE_API_KEY', ''))
    main_agent_model: str = field(default_factory=lambda: os.getenv('MAIN_AGENT_MODEL', 'claude-3-haiku-20240307'))
    subagent_model: str = field(default_factory=lambda: os.getenv('SUBAGENT_MODEL', 'claude-3-haiku-20240307'))
    max_parallel_agents: int = field(default_factory=lambda: int(os.getenv('MAX_PARALLEL_AGENTS', '10')))
    chunk_size: int = field(default_factory=lambda: int(os.getenv('CHUNK_SIZE', '50000')))
    agent_timeout: int = field(default_factory=lambda: int(os.getenv('AGENT_TIMEOUT', '120')))
    debug_mode: bool = field(default_factory=lambda: os.getenv('DEBUG_MODE', 'false').lower() == 'true')
    verbose_logging: bool = field(default_factory=lambda: os.getenv('VERBOSE_LOGGING', 'false').lower() == 'true')
    enable_cache: bool = field(default_factory=lambda: os.getenv('ENABLE_CACHE', 'true').lower() == 'true')
    cache_dir: str = field(default_factory=lambda: os.getenv('CACHE_DIR', '.cache'))
    prompts_dir: str = field(default_factory=lambda: os.getenv('PROMPTS_DIR', 'prompts'))

    def validate(self) -> bool:
        """Validace konfigurace"""
        if not self.api_key:
            logger.warning("CLAUDE_API_KEY není nastavený v .env souboru")
            return False
        return True

    def log_config(self):
        """Vypíše konfiguraci (bez citlivých dat)"""
        logger.info(f"Konfigurace:")
        logger.info(f"  Main Agent Model: {self.main_agent_model}")
        logger.info(f"  Subagent Model: {self.subagent_model}")
        logger.info(f"  Max Parallel Agents: {self.max_parallel_agents}")
        logger.info(f"  Chunk Size: {self.chunk_size}")
        logger.info(f"  Cache Enabled: {self.enable_cache}")


@dataclass
class DocumentSection:
    """Reprezentace sekce dokumentu"""
    id: str
    title: str
    content: str
    page_start: int
    page_end: int
    subsections: List['DocumentSection'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_reference(self) -> str:
        """Vrací referenci na sekci"""
        return f"{self.title} (str. {self.page_start}-{self.page_end})"


@dataclass
class AnalysisResult:
    """Výsledek analýzy od subagenta"""
    question_id: str
    question_text: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    agent_id: str
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentAnalyzer:
    """Hlavní třída pro orchestraci analýzy dokumentů"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.max_parallel_agents = self.config.max_parallel_agents
        self.document_reader = DocumentReader()
        self.question_parser = QuestionParser()
        self.result_aggregator = ResultAggregator()
        self.prompt_manager = PromptManager(self.config.prompts_dir)
        self.console = console
        self.analysis_cache = {}
        self.document_content = ""  # Uchová původní obsah dokumentu

        # Validace konfigurace
        if not self.config.validate():
            logger.warning("Konfigurace není kompletní, některé funkce nemusí fungovat správně")

        if self.config.verbose_logging:
            self.config.log_config()

    async def analyze_document(self, document_path: str, questions_input: str) -> Dict[str, Any]:
        """
        Hlavní metoda pro analýzu dokumentu

        Args:
            document_path: Cesta k dokumentu
            questions_input: String s otázkami nebo cesta k markdown souboru

        Returns:
            Slovník s výsledky analýzy
        """
        start_time = datetime.now()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:

            # 1. Načtení dokumentu
            task = progress.add_task("[cyan]Načítání dokumentu...", total=None)
            document_content = await self.document_reader.read_document(document_path)
            self.document_content = document_content  # Uložení pro pozdější použití
            document_size = len(document_content)
            progress.update(task, description=f"[green]Dokument načten ({document_size:,} znaků)")

            # 2. Extrakce struktury dokumentu pomocí prvního vlny subagentů
            task = progress.add_task("[cyan]Analyzování struktury dokumentu...", total=None)
            document_structure = await self._analyze_structure(document_content)
            progress.update(task, description=f"[green]Struktura analyzována ({len(document_structure)} sekcí)")

            # 3. Parsování otázek
            task = progress.add_task("[cyan]Zpracování otázek...", total=None)
            questions = await self.question_parser.parse_questions(questions_input)
            progress.update(task, description=f"[green]Otázky zpracovány ({len(questions)} otázek)")

            # 4. Distribuce otázek na subagenty
            task = progress.add_task("[cyan]Paralelní analýza otázek...", total=len(questions))
            results = await self._process_questions_parallel(
                document_structure,
                questions,
                progress,
                task
            )
            progress.update(task, description="[green]Analýza dokončena")

        # 5. Agregace výsledků
        final_results = self.result_aggregator.aggregate_results(results)

        # 6. Příprava finální zprávy
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        report = {
            "document_path": document_path,
            "document_size": document_size,
            "structure": self._structure_to_dict(document_structure),
            "questions_count": len(questions),
            "results": final_results,
            "processing_time": processing_time,
            "timestamp": start_time.isoformat(),
            "metadata": {
                "max_parallel_agents": self.max_parallel_agents,
                "document_hash": hashlib.sha256(document_content.encode()).hexdigest()[:16]
            }
        }

        return report

    async def _analyze_structure(self, document_content: str) -> List[DocumentSection]:
        """
        Analyzuje strukturu dokumentu pomocí subagentů
        """
        self.console.print("[bold cyan]Spouštím analýzu struktury dokumentu...[/bold cyan]")

        # Rozdělení dokumentu na chunky pro paralelní zpracování
        chunks = self._split_document_to_chunks(document_content, chunk_size=self.config.chunk_size)

        # Paralelní analýza chunků
        structure_tasks = []
        async with ClaudeSDKClient() as client:
            for i, chunk in enumerate(chunks):
                task = self._analyze_chunk_structure(client, chunk, i, len(chunks))
                structure_tasks.append(task)

            # Počkání na dokončení všech úkolů
            chunk_structures = await asyncio.gather(*structure_tasks)

        # Sloučení struktur z jednotlivých chunků
        merged_structure = self._merge_chunk_structures(chunk_structures)

        return merged_structure

    async def _analyze_chunk_structure(self, client: ClaudeSDKClient, chunk: str, chunk_id: int, total_chunks: int) -> Dict:
        """Analyzuje strukturu jednoho chunku dokumentu"""
        try:
            # Použití prompt manageru pro získání promptu
            prompt = self.prompt_manager.get_prompt(
                'structure_analyzer',
                chunk_number=chunk_id + 1,
                total_chunks=total_chunks,
                chunk_id=chunk_id,
                document_chunk=chunk[:10000]
            )

            options = ClaudeCodeOptions(
                allowed_tools=["Read"],
                permission_mode='acceptOnly',
                system_prompt="Jsi expert na analýzu struktury dokumentů. Extrahuj hierarchickou strukturu.",
                model=self.config.subagent_model  # Použití subagent modelu
            )

            response = ""
            await client.query(prompt, options=options)
            async for message in client.receive_response():
                if message.get("type") == "text":
                    response += message.get("text", "")

            # Parsování JSON odpovědi
            try:
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return {"chunk_id": chunk_id, "sections": []}
            except:
                return {"chunk_id": chunk_id, "sections": []}

        except Exception as e:
            logger.error(f"Chyba při analýze chunku {chunk_id}: {e}")
            return {"chunk_id": chunk_id, "sections": [], "error": str(e)}

    async def _process_questions_parallel(
        self,
        document_structure: List[DocumentSection],
        questions: List[Dict[str, str]],
        progress: Progress,
        task_id: int
    ) -> List[AnalysisResult]:
        """
        Paralelně zpracovává otázky pomocí subagentů
        """
        results = []

        # Vytvoření poolů pro paralelní zpracování
        semaphore = asyncio.Semaphore(self.max_parallel_agents)

        async def process_question(question: Dict[str, str]) -> AnalysisResult:
            async with semaphore:
                # Nalezení relevantních sekcí pro otázku
                relevant_sections = self._find_relevant_sections(
                    document_structure,
                    question['text']
                )

                # Spuštění subagenta pro zpracování otázky
                result = await self._run_question_agent(
                    question,
                    relevant_sections
                )

                progress.advance(task_id)
                return result

        # Spuštění všech otázek paralelně
        tasks = [process_question(q) for q in questions]
        results = await asyncio.gather(*tasks)

        return results

    async def _run_question_agent(
        self,
        question: Dict[str, str],
        relevant_sections: List[DocumentSection]
    ) -> AnalysisResult:
        """
        Spustí subagenta pro zpracování jedné otázky
        """
        start_time = datetime.now()
        agent_id = f"agent_{question['id']}_{hashlib.md5(question['text'].encode()).hexdigest()[:8]}"

        try:
            async with ClaudeSDKClient() as client:
                # Příprava kontextu pro subagenta
                context = self._prepare_agent_context(relevant_sections)

                # Použití prompt manageru
                prompt = self.prompt_manager.get_prompt(
                    'question_analyzer',
                    question_text=question['text'],
                    document_sections=context
                )

                options = ClaudeCodeOptions(
                    allowed_tools=["Read"],
                    permission_mode='acceptOnly',
                    system_prompt="Jsi expert na analýzu právních a technických dokumentů. Poskytuj přesné odpovědi s uvedením zdrojů.",
                    model=self.config.subagent_model  # Použití subagent modelu pro odpovědi
                )

                response = ""
                await client.query(prompt, options=options)
                async for message in client.receive_response():
                    if message.get("type") == "text":
                        response += message.get("text", "")

                # Parsování odpovědi
                result_data = self._parse_agent_response(response)

                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()

                return AnalysisResult(
                    question_id=question['id'],
                    question_text=question['text'],
                    answer=result_data.get('answer', 'Nepodařilo se najít odpověď'),
                    sources=result_data.get('sources', []),
                    confidence=result_data.get('confidence', 0.0),
                    agent_id=agent_id,
                    processing_time=processing_time,
                    metadata={
                        'relevant_sections_count': len(relevant_sections),
                        'response_length': len(response)
                    }
                )

        except Exception as e:
            logger.error(f"Chyba v agentovi {agent_id}: {e}")
            return AnalysisResult(
                question_id=question['id'],
                question_text=question['text'],
                answer=f"Chyba při zpracování: {str(e)}",
                sources=[],
                confidence=0.0,
                agent_id=agent_id,
                processing_time=0.0,
                metadata={'error': str(e)}
            )

    def _split_document_to_chunks(self, document: str, chunk_size: int = 50000) -> List[str]:
        """Rozdělí dokument na zpracovatelné chunky"""
        chunks = []
        words = document.split()
        current_chunk = []
        current_size = 0

        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1

            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _merge_chunk_structures(self, chunk_structures: List[Dict]) -> List[DocumentSection]:
        """Sloučí struktury z jednotlivých chunků"""
        sections = []
        section_id = 0

        for chunk in chunk_structures:
            if 'sections' in chunk:
                for section_data in chunk.get('sections', []):
                    section = DocumentSection(
                        id=f"section_{section_id}",
                        title=section_data.get('title', f'Sekce {section_id}'),
                        content=section_data.get('content', ''),
                        page_start=section_data.get('page_start', 0),
                        page_end=section_data.get('page_end', 0),
                        metadata=section_data.get('metadata', {})
                    )
                    sections.append(section)
                    section_id += 1

        return sections

    def _find_relevant_sections(
        self,
        structure: List[DocumentSection],
        question: str
    ) -> List[DocumentSection]:
        """Najde relevantní sekce pro danou otázku"""
        # Jednoduchá implementace - v produkci by se použilo embeddings/semantic search
        relevant = []
        question_lower = question.lower()
        keywords = question_lower.split()

        for section in structure:
            section_text = (section.title + ' ' + section.content).lower()
            relevance_score = sum(1 for keyword in keywords if keyword in section_text)

            if relevance_score > 0:
                relevant.append((relevance_score, section))

        # Seřazení podle relevance a vrácení top N
        relevant.sort(key=lambda x: x[0], reverse=True)
        return [section for _, section in relevant[:5]]

    def _prepare_agent_context(self, sections: List[DocumentSection]) -> str:
        """Připraví kontext pro subagenta - vrací relevantní části původního dokumentu"""

        # Pokud nemáme sekce, vrátíme část celého dokumentu
        if not sections:
            # Vrátíme prvních 30000 znaků dokumentu jako fallback
            return self.document_content[:30000] if self.document_content else "Dokument není k dispozici"

        # Pokud máme sekce, ale dokument je krátký, vrátíme celý dokument
        if len(self.document_content) < 50000:
            return self.document_content

        # Pro delší dokumenty se pokusíme najít relevantní části
        context_parts = []

        # Přidáme strukturu sekcí pro orientaci
        context_parts.append("=== STRUKTURA DOKUMENTU ===")
        for section in sections[:3]:  # Max 3 nejrelevantnější sekce
            context_parts.append(f"- {section.title} (str. {section.page_start}-{section.page_end})")

        # Přidáme relevantní části textu dokumentu
        context_parts.append("\n=== OBSAH DOKUMENTU ===")

        # Rozdělíme dokument na řádky pro lepší práci
        lines = self.document_content.split('\n')

        # Zkusíme najít relevantní části podle klíčových slov ze sekcí
        relevant_parts = []
        for section in sections[:5]:  # Bereme max 5 nejrelevantnějších sekcí
            # Hledáme zmínky o tématu sekce v dokumentu
            section_keywords = section.title.lower().split()

            for i, line in enumerate(lines):
                line_lower = line.lower()
                # Pokud řádek obsahuje klíčová slova ze sekce
                if any(keyword in line_lower for keyword in section_keywords if len(keyword) > 3):
                    # Přidáme kontext - 5 řádků před a po
                    start = max(0, i - 5)
                    end = min(len(lines), i + 6)
                    context_snippet = '\n'.join(lines[start:end])
                    if context_snippet not in relevant_parts:
                        relevant_parts.append(context_snippet)
                        if len('\n'.join(relevant_parts)) > 20000:  # Limit na velikost kontextu
                            break

            if len('\n'.join(relevant_parts)) > 20000:
                break

        # Pokud jsme něco našli, použijeme to
        if relevant_parts:
            context_parts.extend(relevant_parts)
        else:
            # Jinak vrátíme prostě část dokumentu
            context_parts.append(self.document_content[:20000])

        return '\n'.join(context_parts)

    def _parse_agent_response(self, response: str) -> Dict[str, Any]:
        """Parsuje odpověď od agenta"""
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        # Fallback na jednoduchý parsing
        return {
            "answer": response,
            "sources": [],
            "confidence": 0.5
        }

    def _structure_to_dict(self, structure: List[DocumentSection]) -> List[Dict]:
        """Konvertuje strukturu dokumentu na serializovatelný slovník"""
        result = []
        for section in structure:
            result.append({
                "id": section.id,
                "title": section.title,
                "page_start": section.page_start,
                "page_end": section.page_end,
                "subsections": [self._structure_to_dict([sub])[0] for sub in section.subsections] if section.subsections else [],
                "metadata": section.metadata
            })
        return result


async def main():
    """Hlavní funkce CLI"""
    parser = argparse.ArgumentParser(
        description="Document Analyzer - Paralelní analýza dokumentů pomocí Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Příklady použití:
  python document_analyzer.py dokument.pdf "Jaké jsou hlavní podmínky smlouvy?"
  python document_analyzer.py zakon.pdf otazky.md
  python document_analyzer.py specifikace.docx otazky.md --parallel 20 --output vysledky.json
        """
    )

    parser.add_argument(
        "document",
        help="Cesta k dokumentu (PDF, DOCX, TXT, MD)"
    )

    parser.add_argument(
        "questions",
        help="Otázky (text nebo cesta k markdown souboru)"
    )

    parser.add_argument(
        "--parallel",
        type=int,
        help="Maximální počet paralelních subagentů (výchozí: z .env nebo 10)"
    )

    parser.add_argument(
        "--output",
        help="Cesta pro uložení výsledků (JSON)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Detailní výstup"
    )

    parser.add_argument(
        "--main-model",
        help="Model pro hlavního agenta (přepíše .env)"
    )

    parser.add_argument(
        "--sub-model",
        help="Model pro subagenty (přepíše .env)"
    )

    args = parser.parse_args()

    # Vytvoření konfigurace
    config = Config()

    # Přepsání konfigurace z CLI argumentů
    if args.parallel:
        config.max_parallel_agents = args.parallel
    if args.main_model:
        config.main_agent_model = args.main_model
    if args.sub_model:
        config.subagent_model = args.sub_model
    if args.verbose:
        config.verbose_logging = True
        logging.getLogger().setLevel(logging.DEBUG)

    # Kontrola existence dokumentu
    if not Path(args.document).exists():
        console.print(f"[bold red]Chyba: Dokument '{args.document}' neexistuje![/bold red]")
        sys.exit(1)

    # Kontrola API klíče
    if not config.api_key:
        console.print("[bold yellow]⚠️ CLAUDE_API_KEY není nastavený![/bold yellow]")
        console.print("Nastavte API klíč v .env souboru nebo jako environment variable:")
        console.print("  export CLAUDE_API_KEY=sk-ant-api03-xxxxx")
        console.print("\nPro testování bez API můžete použít:")
        console.print("  python examples/test_without_sdk.py")
        sys.exit(1)

    # Spuštění analýzy
    analyzer = DocumentAnalyzer(config=config)

    console.print(Panel.fit(
        f"[bold cyan]Document Analyzer[/bold cyan]\n"
        f"Dokument: {args.document}\n"
        f"Paralelních agentů: {config.max_parallel_agents}\n"
        f"Main Model: {config.main_agent_model}\n"
        f"Subagent Model: {config.subagent_model}",
        title="Spouštění analýzy"
    ))

    try:
        results = await analyzer.analyze_document(args.document, args.questions)

        # Zobrazení výsledků
        display_results(results)

        # Uložení výsledků
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            console.print(f"[green]Výsledky uloženy do: {args.output}[/green]")

    except Exception as e:
        console.print(f"[bold red]Chyba při analýze: {e}[/bold red]")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def display_results(results: Dict[str, Any]):
    """Zobrazí výsledky analýzy"""
    console.print("\n[bold cyan]═══ VÝSLEDKY ANALÝZY ═══[/bold cyan]\n")

    # Informace o dokumentu
    console.print(f"[bold]Dokument:[/bold] {results['document_path']}")
    console.print(f"[bold]Velikost:[/bold] {results['document_size']:,} znaků")
    console.print(f"[bold]Počet otázek:[/bold] {results['questions_count']}")
    console.print(f"[bold]Čas zpracování:[/bold] {results['processing_time']:.2f} sekund\n")

    # Tabulka s odpověďmi
    for i, result in enumerate(results['results'], 1):
        console.print(f"[bold yellow]Otázka {i}:[/bold yellow] {result['question']}")
        console.print(f"[green]Odpověď:[/green] {result['answer']}")

        if result['sources']:
            console.print("[dim]Zdroje:[/dim]")
            for source in result['sources']:
                console.print(f"  • {source['reference']}")

        console.print(f"[dim]Jistota: {result['confidence']:.0%}[/dim]\n")


if __name__ == "__main__":
    asyncio.run(main())