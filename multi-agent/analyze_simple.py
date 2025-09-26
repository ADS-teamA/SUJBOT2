#!/usr/bin/env python3
"""
Simplified Document Analyzer - basic functionality without vector dependencies
"""

import asyncio
import sys
import os
from pathlib import Path
import argparse
import json
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from document_reader import DocumentReader
from claude_sdk_wrapper import ClaudeSDKClient, ClaudeCodeOptions
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rank_bm25 import BM25Okapi
import tiktoken

console = Console()

class SimpleDocumentAnalyzer:
    """Simplified document analyzer with BM25 search only"""

    def __init__(self):
        self.document_reader = DocumentReader()

        # Claude SDK
        api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key:
            raise ValueError("CLAUDE_API_KEY není nastaven v prostředí")

        self.claude_client = ClaudeSDKClient(api_key=api_key)
        self.document_chunks = []
        self.bm25_index = None

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Simple text chunking"""
        chunks = []
        words = text.split()

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())

        return chunks

    async def index_document(self, document_path: str) -> bool:
        """Index document with BM25"""
        console.print(f"\n[cyan]📚 Indexuji dokument: {document_path}[/cyan]")

        try:
            # Read document
            content = await self.document_reader.read_document(document_path)

            if not content or len(content) < 100:
                console.print("[red]✗ Dokument je příliš krátký nebo prázdný[/red]")
                return False

            # Chunk document
            self.document_chunks = self.chunk_text(content)
            console.print(f"[green]✓ Vytvořeno {len(self.document_chunks)} chunků[/green]")

            # Build BM25 index
            tokenized_chunks = [chunk.lower().split() for chunk in self.document_chunks]
            self.bm25_index = BM25Okapi(tokenized_chunks)

            console.print(f"[green]✓ BM25 index vytvořen[/green]")
            return True

        except Exception as e:
            console.print(f"[red]✗ Chyba při indexování: {e}[/red]")
            return False

    def search_relevant_chunks(self, question: str, top_k: int = 5) -> List[str]:
        """Search relevant chunks using BM25"""
        if not self.bm25_index:
            return []

        # Tokenize query
        query_tokens = question.lower().split()

        # Get scores and find top chunks
        scores = self.bm25_index.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        relevant_chunks = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include chunks with positive scores
                relevant_chunks.append(self.document_chunks[idx])

        return relevant_chunks

    async def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer question using BM25 search and Claude"""
        console.print(f"\n[yellow]🔍 Hledám relevantní kontext pro: {question}[/yellow]")

        # Search relevant chunks
        relevant_chunks = self.search_relevant_chunks(question, top_k=3)

        if not relevant_chunks:
            return {
                "question": question,
                "answer": "Nepodařilo se najít relevantní kontext pro zodpovězení otázky.",
                "confidence": 0.0,
                "chunks_found": 0
            }

        # Prepare context
        context = "\n\n".join(f"Část {i+1}:\n{chunk}" for i, chunk in enumerate(relevant_chunks))

        console.print(f"[green]✓ Nalezeno {len(relevant_chunks)} relevantních částí[/green]")

        # Prepare prompt for Claude
        prompt = f"""Na základě následujícího kontextu z dokumentu odpověz na otázku přesně a stručně.

KONTEXT:
{context}

OTÁZKA: {question}

Odpověz přímo na otázku na základě informací v kontextu. Pokud informace není v kontextu dostupná, řekni to."""

        console.print("[yellow]🤖 Analyzuji s Claude...[/yellow]")

        try:
            options = ClaudeCodeOptions(
                model="claude-3-5-haiku-20241022",
                max_tokens=1000,
                temperature=0.2
            )

            response = await self.claude_client.query(prompt, options)

            return {
                "question": question,
                "answer": response,
                "confidence": 0.8,  # Basic confidence
                "chunks_found": len(relevant_chunks),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            console.print(f"[red]✗ Chyba při dotazu na Claude: {e}[/red]")
            return {
                "question": question,
                "answer": f"Chyba při zpracování: {str(e)}",
                "confidence": 0.0,
                "chunks_found": len(relevant_chunks)
            }

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Jednoduchý analyzátor dokumentů")
    parser.add_argument("document", help="Cesta k dokumentu")
    parser.add_argument("question", help="Otázka k dokumentu")
    parser.add_argument("--output", "-o", help="Výstupní soubor")

    args = parser.parse_args()

    # Check document exists
    doc_path = Path(args.document)
    if not doc_path.exists():
        console.print(f"[red]Dokument nenalezen: {args.document}[/red]")
        sys.exit(1)

    # Initialize analyzer
    try:
        analyzer = SimpleDocumentAnalyzer()
    except Exception as e:
        console.print(f"[red]Chyba inicializace: {e}[/red]")
        sys.exit(1)

    # Analyze document
    console.print(f"\n[bold cyan]═══ Jednoduchá analýza dokumentu ═══[/bold cyan]")
    console.print(f"📄 Dokument: {doc_path.name}")
    console.print(f"❓ Otázka: {args.question}")

    try:
        # Index document
        if not await analyzer.index_document(str(doc_path)):
            sys.exit(1)

        # Answer question
        result = await analyzer.answer_question(args.question)

        # Display result
        console.print(f"\n[bold green]═══ Výsledek ═══[/bold green]")
        console.print(f"[green]Otázka:[/green] {result['question']}")
        console.print(f"[green]Odpověď:[/green] {result['answer']}")
        console.print(f"[dim]Confidence: {result['confidence']:.0%}[/dim]")
        console.print(f"[dim]Nalezené části: {result['chunks_found']}[/dim]")

        # Save result if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            console.print(f"\n[green]✓ Výsledek uložen do: {args.output}[/green]")

    except Exception as e:
        console.print(f"[red]Chyba při analýze: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Analýza přerušena uživatelem[/yellow]")
        sys.exit(0)