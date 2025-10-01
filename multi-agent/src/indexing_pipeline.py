"""
Indexing Pipeline for Document Analyzer
Handles document chunking, embedding generation, and indexing for large-scale processing
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Generator, Tuple
from dataclasses import dataclass
from pathlib import Path
import hashlib
from datetime import datetime
import pickle
import yaml
import re

import tiktoken
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.asyncio import tqdm
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from document_reader import DocumentReader
from vector_store_faiss import DocumentChunk, create_vector_store

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class ChunkingStrategy:
    """Configuration for document chunking"""
    strategy: str = "semantic"  # semantic, fixed, sliding_window
    chunk_size: int = 512
    chunk_overlap: float = 0.15
    min_chunk_size: int = 128
    max_chunk_size: int = 1024
    separator: str = "\n\n"


class DocumentChunker:
    """Handles document chunking with various strategies"""

    def __init__(self, strategy: ChunkingStrategy, encoding_name: str = "cl100k_base"):
        self.strategy = strategy
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk_document(self, text: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Chunk document based on configured strategy"""
        if self.strategy.strategy == "semantic":
            return self._semantic_chunking(text, metadata)
        elif self.strategy.strategy == "fixed":
            return self._fixed_chunking(text, metadata)
        elif self.strategy.strategy == "sliding_window":
            return self._sliding_window_chunking(text, metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy.strategy}")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

    def _semantic_chunking(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Semantic chunking based on sentence embeddings similarity
        Groups similar sentences together to maintain context
        """
        chunks = []

        # Split into sentences
        sentences = self._split_into_sentences(text)
        if not sentences:
            return chunks

        # Group sentences into semantic chunks
        current_chunk = []
        current_tokens = 0

        for i, sentence in enumerate(sentences):
            sentence_tokens = self._count_tokens(sentence)

            # Check if adding this sentence exceeds max size
            if current_tokens + sentence_tokens > self.strategy.max_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()

                chunk_metadata = {**(metadata or {}), "chunk_index": len(chunks)}
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    metadata=chunk_metadata
                ))

                # Start new chunk with overlap
                overlap_size = int(len(current_chunk) * self.strategy.chunk_overlap)
                current_chunk = current_chunk[-overlap_size:] if overlap_size > 0 else []
                current_tokens = sum(self._count_tokens(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

            # Check if we have minimum chunk size
            if current_tokens >= self.strategy.chunk_size:
                # Look ahead for natural break point
                if i + 1 < len(sentences):
                    next_sentence = sentences[i + 1]
                    # Check for paragraph break or topic change
                    if next_sentence.startswith("\n") or self._is_topic_boundary(current_chunk, next_sentence):
                        # Save chunk at natural boundary
                        chunk_text = " ".join(current_chunk)
                        chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()

                        chunk_metadata = {**(metadata or {}), "chunk_index": len(chunks)}
                        chunks.append(DocumentChunk(
                            chunk_id=chunk_id,
                            content=chunk_text,
                            metadata=chunk_metadata
                        ))

                        # Start new chunk
                        overlap_size = int(len(current_chunk) * self.strategy.chunk_overlap)
                        current_chunk = current_chunk[-overlap_size:] if overlap_size > 0 else []
                        current_tokens = sum(self._count_tokens(s) for s in current_chunk)

        # Add remaining chunk
        if current_chunk and current_tokens >= self.strategy.min_chunk_size:
            chunk_text = " ".join(current_chunk)
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()

            chunk_metadata = {**(metadata or {}), "chunk_index": len(chunks)}
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                metadata=chunk_metadata
            ))

        return chunks

    def _fixed_chunking(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Fixed-size chunking with overlap"""
        chunks = []
        tokens = self.encoding.encode(text)

        chunk_size = self.strategy.chunk_size
        overlap_size = int(chunk_size * self.strategy.chunk_overlap)
        stride = chunk_size - overlap_size

        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i:i + chunk_size]

            if len(chunk_tokens) < self.strategy.min_chunk_size:
                continue

            chunk_text = self.encoding.decode(chunk_tokens)
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()

            chunk_metadata = {
                **(metadata or {}),
                "chunk_index": len(chunks),
                "token_start": i,
                "token_end": i + len(chunk_tokens)
            }

            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                metadata=chunk_metadata
            ))

        return chunks

    def _sliding_window_chunking(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Sliding window chunking for maximum context preservation"""
        chunks = []
        sentences = self._split_into_sentences(text)

        window_size = self.strategy.chunk_size
        stride = int(window_size * (1 - self.strategy.chunk_overlap))

        for i in range(0, len(sentences), stride):
            window_sentences = sentences[i:i + window_size]
            chunk_text = " ".join(window_sentences)

            if self._count_tokens(chunk_text) < self.strategy.min_chunk_size:
                continue

            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()

            chunk_metadata = {
                **(metadata or {}),
                "chunk_index": len(chunks),
                "window_start": i,
                "window_end": i + len(window_sentences)
            }

            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                metadata=chunk_metadata
            ))

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - can be improved with spaCy or NLTK
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _is_topic_boundary(self, current_chunk: List[str], next_sentence: str) -> bool:
        """Detect if there's a topic change between current chunk and next sentence"""
        # Simple heuristic - can be improved with embeddings
        # Check for heading patterns
        if re.match(r'^[A-Z][A-Z\s]+$', next_sentence.strip()):
            return True
        if re.match(r'^\d+\.', next_sentence.strip()):
            return True
        if next_sentence.strip().startswith('#'):
            return True

        return False


class IndexingPipeline:
    """Main pipeline for document indexing"""

    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.document_reader = DocumentReader()
        self.vector_store = create_vector_store(self.config)

        # Setup chunking strategy
        chunking_config = self.config.get('indexing', {})
        self.chunking_strategy = ChunkingStrategy(
            strategy="semantic" if chunking_config.get('semantic_chunking', True) else "fixed",
            chunk_size=chunking_config.get('chunk_size', 512),
            chunk_overlap=chunking_config.get('chunk_overlap', 0.15),
            min_chunk_size=chunking_config.get('min_chunk_size', 128),
            max_chunk_size=chunking_config.get('max_chunk_size', 1024)
        )
        self.chunker = DocumentChunker(self.chunking_strategy)

        # Processing settings
        self.batch_size = chunking_config.get('batch_size', 1000)
        self.max_workers = chunking_config.get('max_workers', 8)

        # Create checkpoint directory
        self.checkpoint_dir = Path(self.config.get('paths', {}).get('temp_dir', '.tmp')) / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    async def index_document(self, document_path: str, document_id: str = None) -> Dict[str, Any]:
        """Index a single document"""
        start_time = datetime.now()

        # Generate document ID if not provided
        if not document_id:
            document_id = hashlib.md5(document_path.encode()).hexdigest()[:16]

        console.print(f"[cyan]Indexing document: {document_path}[/cyan]")

        # Read document
        try:
            content = await self.document_reader.read_document(document_path)
            doc_metadata = await self.document_reader.get_document_metadata(document_path)
        except Exception as e:
            logger.error(f"Failed to read document {document_path}: {e}")
            return {"status": "error", "message": str(e)}

        # Chunk document
        console.print("[yellow]Chunking document...[/yellow]")
        chunks = self.chunker.chunk_document(content, metadata={
            "document_id": document_id,
            "document_path": document_path,
            **doc_metadata
        })

        console.print(f"[green]Created {len(chunks)} chunks[/green]")

        # Index chunks in batches
        console.print("[yellow]Indexing chunks...[/yellow]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Indexing...", total=len(chunks))

            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                await self.vector_store.add_documents(batch)
                progress.update(task, advance=len(batch))

        # Calculate statistics
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        stats = {
            "status": "success",
            "document_id": document_id,
            "document_path": document_path,
            "document_content": content,  # Include content for hybrid retriever
            "chunks": chunks,  # Return actual chunks for BM25 indexing
            "total_chunks": len(chunks),
            "processing_time": processing_time,
            "avg_chunk_size": sum(len(c.content) for c in chunks) / len(chunks) if chunks else 0,
            "indexed_at": end_time.isoformat()
        }

        console.print(f"[green]✓ Document indexed successfully in {processing_time:.2f}s[/green]")
        return stats

    async def index_directory(self, directory_path: str, pattern: str = "*.pdf") -> List[Dict[str, Any]]:
        """Index all documents in a directory"""
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory_path}")

        # Find all matching files
        files = list(directory.glob(pattern))
        console.print(f"[cyan]Found {len(files)} files matching pattern: {pattern}[/cyan]")

        results = []
        for file_path in files:
            result = await self.index_document(str(file_path))
            results.append(result)

            # Save checkpoint after each document
            self._save_checkpoint(results)

        return results

    async def index_batch(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Index multiple documents in parallel"""
        console.print(f"[cyan]Indexing {len(file_paths)} documents in parallel[/cyan]")

        # Use asyncio for parallel processing
        semaphore = asyncio.Semaphore(self.max_workers)

        async def index_with_semaphore(path):
            async with semaphore:
                return await self.index_document(path)

        # Process all documents in parallel
        tasks = [index_with_semaphore(path) for path in file_paths]
        results = await asyncio.gather(*tasks)

        # Save final checkpoint
        self._save_checkpoint(results)

        return results

    def _save_checkpoint(self, results: List[Dict[str, Any]]):
        """Save indexing checkpoint for recovery"""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{datetime.now():%Y%m%d_%H%M%S}.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({
                "results": results,
                "timestamp": datetime.now().isoformat(),
                "config": self.config
            }, f)

        # Keep only last 5 checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        if len(checkpoints) > 5:
            for old_checkpoint in checkpoints[:-5]:
                old_checkpoint.unlink()

    async def reindex_failed(self, checkpoint_file: str) -> List[Dict[str, Any]]:
        """Reindex documents that failed in previous run"""
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)

        failed_docs = [r for r in checkpoint['results'] if r.get('status') != 'success']
        console.print(f"[yellow]Found {len(failed_docs)} failed documents to reindex[/yellow]")

        results = []
        for doc in failed_docs:
            if 'document_path' in doc:
                result = await self.index_document(doc['document_path'])
                results.append(result)

        return results

    async def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed documents"""
        vector_stats = await self.vector_store.get_stats()

        return {
            "vector_store": vector_stats,
            "chunking_strategy": {
                "strategy": self.chunking_strategy.strategy,
                "chunk_size": self.chunking_strategy.chunk_size,
                "overlap": self.chunking_strategy.chunk_overlap
            },
            "config": {
                "batch_size": self.batch_size,
                "max_workers": self.max_workers
            }
        }


# CLI interface
async def main():
    """CLI for indexing pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="Document Indexing Pipeline")
    parser.add_argument("action", choices=["index", "batch", "directory", "stats", "reindex"],
                        help="Action to perform")
    parser.add_argument("--path", help="Path to document or directory")
    parser.add_argument("--paths", nargs="+", help="Multiple paths for batch processing")
    parser.add_argument("--pattern", default="*.pdf", help="File pattern for directory indexing")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--checkpoint", help="Checkpoint file for reindexing")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = IndexingPipeline(args.config)

    if args.action == "index":
        if not args.path:
            console.print("[red]Error: --path required for indexing[/red]")
            return

        result = await pipeline.index_document(args.path)
        console.print(result)

    elif args.action == "batch":
        if not args.paths:
            console.print("[red]Error: --paths required for batch indexing[/red]")
            return

        results = await pipeline.index_batch(args.paths)
        console.print(f"Indexed {len(results)} documents")

    elif args.action == "directory":
        if not args.path:
            console.print("[red]Error: --path required for directory indexing[/red]")
            return

        results = await pipeline.index_directory(args.path, args.pattern)
        console.print(f"Indexed {len(results)} documents from directory")

    elif args.action == "stats":
        stats = await pipeline.get_index_stats()
        console.print(stats)

    elif args.action == "reindex":
        if not args.checkpoint:
            console.print("[red]Error: --checkpoint required for reindexing[/red]")
            return

        results = await pipeline.reindex_failed(args.checkpoint)
        console.print(f"Reindexed {len(results)} documents")


if __name__ == "__main__":
    asyncio.run(main())