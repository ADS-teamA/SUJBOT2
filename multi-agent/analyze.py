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

        self.claude_client = ClaudeSDKClient(api_key=api_key)
        self.indexed_documents = {}

    async def index_document(self, document_path: str) -> Dict[str, Any]:
        """Index document for vector search"""
        console.print(f"\n[cyan]📚 Indexuji dokument: {document_path}[/cyan]")

        # Index using the pipeline
        result = await self.indexing_pipeline.index_document(document_path)

        if result.get('status') == 'success':
            self.indexed_documents[document_path] = result

            # IMPORTANT: Also index chunks in the hybrid retriever
            # Get the chunks from the vector store
            chunks = self.indexing_pipeline.chunker.chunk_document(
                result.get('document_content', ''),
                metadata={'document_id': result['document_id'], 'document_path': document_path}
            )

            # Index chunks in hybrid retriever for BM25 search
            await self.hybrid_retriever.index_documents(chunks)

            console.print(f"[green]✓ Dokument úspěšně zaindexován[/green]")
            console.print(f"  • Počet chunků: {result['total_chunks']}")
            console.print(f"  • Čas zpracování: {result['processing_time']:.2f}s")
        else:
            console.print(f"[red]✗ Indexování selhalo: {result.get('message')}[/red]")

        return result

    async def answer_question(self, question: str, max_context_tokens: int = 4000) -> Dict[str, Any]:
        """Answer question using vector search and Claude"""
        console.print(f"\n[yellow]🔍 Hledám relevantní kontext pro: {question}[/yellow]")

        # Retrieve relevant context
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Vyhledávání v dokumentu...", total=None)

            # Get relevant context using hybrid retrieval
            context = await self.hybrid_retriever.get_relevant_context(
                question,
                max_tokens=max_context_tokens
            )

            progress.update(task, completed=True)

        if not context:
            return {
                "question": question,
                "answer": "Nepodařilo se najít relevantní kontext pro zodpovězení otázky.",
                "confidence": 0.0
            }

        # Prepare prompt for Claude
        prompt = f"""Na základě následujícího kontextu z dokumentu odpověz na otázku.

KONTEXT:
{context}

OTÁZKA: {question}

Odpověz přesně a uveď konkrétní informace z kontextu. Pokud informace není v kontextu, řekni to."""

        console.print("[yellow]🤖 Analyzuji s Claude...[/yellow]")

        # Get answer from Claude
        try:
            options = ClaudeCodeOptions(
                model="claude-3-5-sonnet-latest",
                max_tokens=2000,
                temperature=0.3
            )

            response = await self.claude_client.query(prompt, options)

            return {
                "question": question,
                "answer": response,
                "confidence": 0.85,  # Can be calculated based on retrieval scores
                "context_used": len(context),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return {
                "question": question,
                "answer": f"Chyba při zpracování: {str(e)}",
                "confidence": 0.0
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
            console.print(f"[dim]Confidence: {result['confidence']:.2%}[/dim]")

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