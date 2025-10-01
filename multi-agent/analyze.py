#!/usr/bin/env python3
"""
Enhanced Document Analyzer with Vector Search
Analyzuje dokumenty pomocí vektorového vyhledávání a hybridního retrieval
"""

import asyncio
import sys
import os

# Fix tokenizers parallelism warning
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from pathlib import Path
import argparse
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Přidání src adresáře do cesty
sys.path.insert(0, str(Path(__file__).parent / "src"))

from document_reader import DocumentReader
from indexing_pipeline import IndexingPipeline
from hybrid_retriever import HybridRetriever
from claude_sdk_wrapper import ClaudeSDKClient, ClaudeCodeOptions
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console = Console()


class VectorizedDocumentAnalyzer:
    """Enhanced document analyzer with vector search capabilities"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.document_reader = DocumentReader()
        self.indexing_pipeline = IndexingPipeline(config_path)
        self.hybrid_retriever = HybridRetriever(config_path)

        # Claude SDK pro analýzu
        api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key:
            raise ValueError("CLAUDE_API_KEY není nastavený v prostředí")

        # Načtení konfigurace z environment proměnných
        self.main_model = os.getenv("MAIN_AGENT_MODEL", "claude-3-5-sonnet-20241022")
        self.subagent_model = os.getenv("SUBAGENT_MODEL", "claude-3-5-haiku-20241022")
        self.max_parallel_agents = int(os.getenv("MAX_PARALLEL_AGENTS", "10"))
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "50000"))
        self.agent_timeout = int(os.getenv("AGENT_TIMEOUT", "120"))
        self.query_timeout = int(os.getenv("QUERY_TIMEOUT", "60"))
        self.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
        self.verbose_logging = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"
        self.enable_cache = os.getenv("ENABLE_CACHE", "true").lower() == "true"
        self.cache_dir = os.getenv("CACHE_DIR", ".cache")
        self.prompts_dir = os.getenv("PROMPTS_DIR", "prompts")
        self.custom_system_prompt = os.getenv("CUSTOM_SYSTEM_PROMPT", "")

        # Nastavení loggingu na základě environment proměnných
        if self.verbose_logging:
            logging.getLogger().setLevel(logging.DEBUG)
        elif self.debug_mode:
            logging.getLogger().setLevel(logging.DEBUG)

        self.claude_client = ClaudeSDKClient(api_key=api_key)
        self.indexed_documents = {}

        # Vytvoření cache adresáře pokud je povolena cache a adresář neexistuje
        if self.enable_cache:
            cache_path = Path(self.cache_dir)
            cache_path.mkdir(exist_ok=True)

        # Výpis aktuální konfigurace
        logger.info(f"Inicializace VectorizedDocumentAnalyzer:")
        logger.info(f"  • Hlavní model: {self.main_model}")
        logger.info(f"  • Subagent model: {self.subagent_model}")
        logger.info(f"  • Max paralelní agenti: {self.max_parallel_agents}")
        logger.info(f"  • Chunk size: {self.chunk_size}")
        logger.info(f"  • Agent timeout: {self.agent_timeout}s")
        logger.info(f"  • Query timeout: {self.query_timeout}s")
        logger.info(f"  • Debug mode: {self.debug_mode}")
        logger.info(f"  • Verbose logging: {self.verbose_logging}")
        logger.info(f"  • Cache povolena: {self.enable_cache}")
        logger.info(f"  • Cache adresář: {self.cache_dir}")
        logger.info(f"  • Prompts adresář: {self.prompts_dir}")

    async def index_document(self, document_path: str) -> Dict[str, Any]:
        """Index document for vector search"""
        console.print(f"\n[cyan]📚 Indexuji dokument: {document_path}[/cyan]")

        # Index using the pipeline
        result = await self.indexing_pipeline.index_document(document_path)

        if result.get('status') == 'success':
            self.indexed_documents[document_path] = result

            # FIX: No more duplicate indexing!
            # The chunks are already indexed in vector store by indexing_pipeline
            # We only need to index them in BM25 for hybrid retriever

            # Get the already-created chunks from indexing pipeline
            chunks = self.indexing_pipeline.chunker.chunk_document(
                result.get('document_content', ''),
                metadata={'document_id': result['document_id'], 'document_path': document_path}
            )

            # Only index in BM25 (not in vector store - already done)
            await self.hybrid_retriever.index_documents_bm25_only(chunks)

            console.print(f"[green]✓ Dokument úspěšně zaindexován[/green]")
            console.print(f"  • Počet chunků: {result['total_chunks']}")
            console.print(f"  • Čas zpracování: {result['processing_time']:.2f}s")
        else:
            console.print(f"[red]✗ Indexování selhalo: {result.get('message')}[/red]")

        return result

    async def answer_question(self, question: str, max_context_tokens: int = None) -> Dict[str, Any]:
        """Answer question using vector search and Claude"""
        console.print(f"\n[yellow]🔍 Hledám relevantní kontext pro: {question}[/yellow]")

        # Nastavení max_context_tokens na základě konfigurace
        if max_context_tokens is None:
            max_context_tokens = min(self.chunk_size, 4000)

        # Retrieve relevant context with source tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Vyhledávání v dokumentu...", total=None)

            # Get relevant chunks (not just text, but full SearchResult objects)
            retrieved_chunks = await self.hybrid_retriever.retrieve(question)

            progress.update(task, completed=True)

        if not retrieved_chunks:
            return {
                "question": question,
                "answer": "Nepodařilo se najít relevantní kontext pro zodpovězení otázky.",
                "confidence": 0.0,
                "sources": [],
                "context_used": 0
            }

        # Build context and citations from retrieved chunks
        context_parts = []
        sources = []

        for i, chunk in enumerate(retrieved_chunks, 1):
            # Add numbered context for citation reference
            context_parts.append(f"[{i}] {chunk.content}")

            # Extract source information
            source = {
                "chunk_id": chunk.chunk_id,
                "reference": f"[{i}]",
                "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                "score": float(chunk.score),
                "metadata": chunk.metadata or {}
            }

            # Add page number if available
            if chunk.metadata and 'page' in chunk.metadata:
                source['page'] = chunk.metadata['page']

            sources.append(source)

        context = "\n\n".join(context_parts)

        # Calculate dynamic confidence based on retrieval scores
        avg_score = sum(chunk.score for chunk in retrieved_chunks) / len(retrieved_chunks)

        # Normalize confidence score based on score type
        # If scores are negative (from cross-encoder reranking), normalize differently
        if avg_score < 0:
            # Cross-encoder scores typically range from -15 to +10
            # Map to 0-1 range with sigmoid-like function
            # score > 0 = high confidence (>0.6), score around -10 = medium (0.2-0.4)
            confidence = max(0.1, min(1.0, (avg_score + 15) / 25))
        else:
            # Semantic similarity scores are already 0-1
            confidence = min(max(avg_score, 0.0), 1.0)

        # Prepare prompt for Claude with citation instructions
        base_prompt = f"""Na základě následujícího kontextu z dokumentu odpověz na otázku.
Každá část kontextu je označena číslem v hranatých závorkách [1], [2], atd.
Pokud odkazuješ na informace z kontextu, uveď jejich zdroj pomocí tohoto čísla.

KONTEXT:
{context}

OTÁZKA: {question}

Odpověz přesně a uveď konkrétní informace z kontextu. Při citaci použij formát [číslo].
Pokud informace není v kontextu, řekni to."""

        # Přidání vlastního system promptu pokud je nastaven
        if self.custom_system_prompt:
            prompt = f"{self.custom_system_prompt}\n\n{base_prompt}"
        else:
            prompt = base_prompt

        console.print("[yellow]🤖 Analyzuji s Claude...[/yellow]")

        # Get answer from Claude
        try:
            options = ClaudeCodeOptions(
                model=self.main_model,
                max_tokens=2000,
                temperature=0.3
            )

            response = await self.claude_client.query(prompt, options)

            return {
                "question": question,
                "answer": response,
                "confidence": round(confidence, 3),  # Dynamic confidence from retrieval scores
                "sources": sources,  # Citation tracking!
                "context_used": len(context),
                "num_sources": len(sources),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return {
                "question": question,
                "answer": f"Chyba při zpracování: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "context_used": 0
            }

    async def analyze_document(self, document_path: str, questions: List[str]) -> List[Dict[str, Any]]:
        """Analyze document with multiple questions"""

        # First, index the document
        index_result = await self.index_document(document_path)

        if index_result.get('status') != 'success':
            console.print("[red]Nelze pokračovat bez úspěšného indexování[/red]")
            return []

        # Answer each question
        results = []
        for i, question in enumerate(questions, 1):
            console.print(f"\n[cyan]Otázka {i}/{len(questions)}:[/cyan] {question}")

            result = await self.answer_question(question)
            results.append(result)

            console.print(f"[green]Odpověď:[/green] {result['answer'][:200]}...")
            console.print(f"[dim]Confidence: {result['confidence']:.1%} | Zdroje: {result.get('num_sources', 0)}[/dim]")

            # Display sources if available
            if result.get('sources'):
                console.print(f"\n[dim]📚 Zdroje citací:[/dim]")
                for source in result['sources'][:3]:  # Show top 3 sources
                    ref = source['reference']
                    score = source['score']
                    preview = source['content_preview']
                    page = source.get('page', 'N/A')
                    console.print(f"  {ref} [Score: {score:.3f}] Page {page}: {preview[:100]}...")

        return results


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Analyzátor dokumentů s vektorovým vyhledáváním")
    parser.add_argument("document", help="Cesta k dokumentu (PDF, DOCX, TXT, MD)")
    parser.add_argument("questions", nargs='?', help="Otázky (text nebo soubor)")
    parser.add_argument("--config", default="config.yaml", help="Konfigurační soubor")
    parser.add_argument("--output", "-o", help="Výstupní soubor pro výsledky")
    parser.add_argument("--verbose", "-v", action="store_true", help="Detailní výstup")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check document exists
    doc_path = Path(args.document)
    if not doc_path.exists():
        console.print(f"[red]Dokument nenalezen: {args.document}[/red]")
        sys.exit(1)

    # Prepare questions
    questions = []

    if args.questions:
        # Check if it's a file
        question_path = Path(args.questions)
        if question_path.exists():
            with open(question_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Parse questions (one per line or numbered)
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Remove numbering if present
                        if line[0].isdigit() and '. ' in line:
                            line = line.split('. ', 1)[1]
                        questions.append(line)
        else:
            # Single question
            questions = [args.questions]
    else:
        # Interactive mode
        console.print("[cyan]Zadejte otázku:[/cyan]")
        question = input("> ")
        questions = [question]

    # Initialize analyzer
    try:
        analyzer = VectorizedDocumentAnalyzer(args.config)
    except Exception as e:
        console.print(f"[red]Chyba inicializace: {e}[/red]")
        sys.exit(1)

    # Analyze document
    console.print(f"\n[bold cyan]═══ Analýza dokumentu ═══[/bold cyan]")
    console.print(f"📄 Dokument: {doc_path.name}")
    console.print(f"❓ Počet otázek: {len(questions)}")

    try:
        results = await analyzer.analyze_document(str(doc_path), questions)

        # Save results if output specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            console.print(f"\n[green]✓ Výsledky uloženy do: {args.output}[/green]")

        # Print summary
        console.print(f"\n[bold green]═══ Souhrn ═══[/bold green]")
        console.print(f"✓ Zodpovězeno otázek: {len(results)}")
        avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
        console.print(f"📊 Průměrná confidence: {avg_confidence:.2%}")

    except Exception as e:
        console.print(f"[red]Chyba při analýze: {e}[/red]")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Analýza přerušena uživatelem[/yellow]")
        sys.exit(0)