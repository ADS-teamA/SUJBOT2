#!/usr/bin/env python3
"""
Performance Benchmark for PostgreSQL + pgvector

Measures performance metrics for various operations:
- Batch insert throughput
- Vector search latency
- Hybrid search performance
- Concurrent query handling

Usage:
    python tests/benchmark_postgresql.py
    python tests/benchmark_postgresql.py --chunks 10000 --queries 100
"""

import asyncio
import argparse
import time
import os
import sys
from pathlib import Path
from typing import List
import statistics

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pg_vector_store import PostgreSQLVectorStore, PostgreSQLConfig
from src.embeddings import LegalEmbedder, EmbeddingConfig
from src.chunker import LegalChunk


class PerformanceBenchmark:
    """Performance benchmark suite"""

    def __init__(self, num_chunks: int = 1000, num_queries: int = 50):
        self.num_chunks = num_chunks
        self.num_queries = num_queries

        # Initialize components
        self.embedder = LegalEmbedder(EmbeddingConfig(
            model_name="BAAI/bge-m3",
            device="cpu",
            batch_size=32
        ))

        self.config = PostgreSQLConfig(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "sujbot2"),
            user=os.getenv("POSTGRES_USER", "sujbot_app"),
            password=os.getenv("POSTGRES_PASSWORD", "sujbot2_dev_password"),
        )

        self.store = PostgreSQLVectorStore(self.embedder, self.config)

    def generate_test_chunks(self, num_chunks: int, doc_id: str) -> List[LegalChunk]:
        """Generate test chunks with varied content"""
        chunks = []
        sample_contents = [
            "Dodavatel odpovídá za vady výrobku podle občanského zákoníku.",
            "Kupující má právo odstoupit od smlouvy do 14 dnů.",
            "Záruční doba činí 24 měsíců od převzetí věci.",
            "Při porušení smlouvy vzniká nárok na náhradu škody.",
            "Smluvní strany se dohodly na následujících podmínkách.",
        ]

        for i in range(num_chunks):
            content = f"§{i} {sample_contents[i % len(sample_contents)]} Dodatečný text číslo {i}."

            chunk = LegalChunk(
                chunk_id=f"bench_chunk_{doc_id}_{i}",
                chunk_index=i,
                content=content,
                document_id=doc_id,
                document_type="law_code",
                hierarchy_path=f"Část I > §{i}",
                legal_reference=f"§{i}",
                structural_level="paragraph",
                metadata={
                    "paragraph": i,
                    "content_type": ["obligation", "definition", "general"][i % 3],
                    "token_count": len(content.split())
                }
            )
            chunks.append(chunk)

        return chunks

    async def benchmark_batch_insert(self):
        """Benchmark: Batch insert performance"""
        print("\n" + "=" * 70)
        print("BENCHMARK 1: Batch Insert Performance")
        print("=" * 70)

        chunks = self.generate_test_chunks(self.num_chunks, "bench_insert")

        print(f"Inserting {self.num_chunks} chunks...")

        start = time.time()
        await self.store.add_document(
            chunks=chunks,
            document_id="bench_insert",
            document_type="law_code",
            metadata={"benchmark": "insert"}
        )
        elapsed = time.time() - start

        chunks_per_sec = self.num_chunks / elapsed
        time_per_chunk = (elapsed / self.num_chunks) * 1000  # ms

        print(f"\n📊 Results:")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Throughput: {chunks_per_sec:.1f} chunks/sec")
        print(f"  Avg time per chunk: {time_per_chunk:.2f}ms")

        return {
            "total_time": elapsed,
            "chunks_per_sec": chunks_per_sec,
            "time_per_chunk": time_per_chunk
        }

    async def benchmark_vector_search(self):
        """Benchmark: Vector search latency"""
        print("\n" + "=" * 70)
        print("BENCHMARK 2: Vector Search Latency")
        print("=" * 70)

        # Ensure data exists
        chunks = self.generate_test_chunks(min(self.num_chunks, 1000), "bench_search")
        await self.store.add_document(
            chunks=chunks,
            document_id="bench_search",
            document_type="law_code",
            metadata={"benchmark": "search"}
        )

        queries = [
            "odpovědnost za vady",
            "záruční doba",
            "náhrada škody",
            "odstoupení od smlouvy",
            "práva kupujícího"
        ]

        print(f"Running {self.num_queries} search queries...")

        latencies = []
        for i in range(self.num_queries):
            query = queries[i % len(queries)]

            start = time.time()
            results = await self.store.search(
                query=query,
                document_ids=["bench_search"],
                top_k=20
            )
            elapsed = (time.time() - start) * 1000  # ms

            latencies.append(elapsed)

        avg_latency = statistics.mean(latencies)
        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99 = statistics.quantiles(latencies, n=100)[98]  # 99th percentile

        print(f"\n📊 Results ({self.num_queries} queries):")
        print(f"  Avg latency: {avg_latency:.1f}ms")
        print(f"  P50: {p50:.1f}ms")
        print(f"  P95: {p95:.1f}ms")
        print(f"  P99: {p99:.1f}ms")
        print(f"  Min: {min(latencies):.1f}ms")
        print(f"  Max: {max(latencies):.1f}ms")

        return {
            "avg_latency": avg_latency,
            "p50": p50,
            "p95": p95,
            "p99": p99
        }

    async def benchmark_concurrent_queries(self):
        """Benchmark: Concurrent query handling"""
        print("\n" + "=" * 70)
        print("BENCHMARK 3: Concurrent Query Performance")
        print("=" * 70)

        # Ensure data exists
        chunks = self.generate_test_chunks(min(self.num_chunks, 1000), "bench_concurrent")
        await self.store.add_document(
            chunks=chunks,
            document_id="bench_concurrent",
            document_type="law_code",
            metadata={"benchmark": "concurrent"}
        )

        queries = [
            "odpovědnost za vady",
            "záruční doba",
            "náhrada škody",
        ]

        num_concurrent = 10
        print(f"Running {num_concurrent} concurrent queries...")

        async def run_query(query: str):
            start = time.time()
            await self.store.search(
                query=query,
                document_ids=["bench_concurrent"],
                top_k=20
            )
            return (time.time() - start) * 1000

        start = time.time()
        tasks = [run_query(queries[i % len(queries)]) for i in range(num_concurrent)]
        latencies = await asyncio.gather(*tasks)
        total_time = time.time() - start

        avg_latency = statistics.mean(latencies)
        qps = num_concurrent / total_time

        print(f"\n📊 Results:")
        print(f"  Concurrent queries: {num_concurrent}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Queries per second: {qps:.1f} QPS")
        print(f"  Avg query latency: {avg_latency:.1f}ms")

        return {
            "qps": qps,
            "avg_latency": avg_latency,
            "total_time": total_time
        }

    async def benchmark_reference_lookup(self):
        """Benchmark: Direct reference lookup"""
        print("\n" + "=" * 70)
        print("BENCHMARK 4: Reference Lookup Performance")
        print("=" * 70)

        chunks = self.generate_test_chunks(min(self.num_chunks, 1000), "bench_ref")
        await self.store.add_document(
            chunks=chunks,
            document_id="bench_ref",
            document_type="law_code",
            metadata={"benchmark": "reference"}
        )

        num_lookups = 100
        print(f"Running {num_lookups} reference lookups...")

        latencies = []
        for i in range(num_lookups):
            ref = f"§{i % 100}"

            start = time.time()
            chunk = await self.store.search_by_reference(ref, "bench_ref")
            elapsed = (time.time() - start) * 1000

            latencies.append(elapsed)

        avg_latency = statistics.mean(latencies)

        print(f"\n📊 Results ({num_lookups} lookups):")
        print(f"  Avg latency: {avg_latency:.2f}ms")
        print(f"  Min: {min(latencies):.2f}ms")
        print(f"  Max: {max(latencies):.2f}ms")

        return {
            "avg_latency": avg_latency
        }

    async def cleanup(self):
        """Clean up benchmark data"""
        print("\n🧹 Cleaning up benchmark data...")

        docs_to_delete = [
            "bench_insert",
            "bench_search",
            "bench_concurrent",
            "bench_ref"
        ]

        for doc_id in docs_to_delete:
            try:
                await self.store.delete_document(doc_id)
            except:
                pass

        await self.store.close()
        print("✓ Cleanup complete")

    async def run_all_benchmarks(self):
        """Run all benchmarks and print summary"""
        print("\n" + "=" * 70)
        print("PostgreSQL + pgvector Performance Benchmark")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Database: {self.config.host}:{self.config.port}/{self.config.database}")
        print(f"  Test chunks: {self.num_chunks}")
        print(f"  Test queries: {self.num_queries}")

        results = {}

        try:
            # Run benchmarks
            results['insert'] = await self.benchmark_batch_insert()
            results['search'] = await self.benchmark_vector_search()
            results['concurrent'] = await self.benchmark_concurrent_queries()
            results['reference'] = await self.benchmark_reference_lookup()

            # Print summary
            print("\n" + "=" * 70)
            print("SUMMARY")
            print("=" * 70)

            print("\n📈 Insert Performance:")
            print(f"  • {results['insert']['chunks_per_sec']:.1f} chunks/sec")

            print("\n🔍 Search Performance:")
            print(f"  • Avg latency: {results['search']['avg_latency']:.1f}ms")
            print(f"  • P95 latency: {results['search']['p95']:.1f}ms")

            print("\n⚡ Concurrent Performance:")
            print(f"  • {results['concurrent']['qps']:.1f} queries/sec")

            print("\n🎯 Reference Lookup:")
            print(f"  • {results['reference']['avg_latency']:.2f}ms avg")

            # Evaluate performance
            print("\n📊 Performance Evaluation:")
            if results['search']['p95'] < 50:
                print("  ✅ Vector search: EXCELLENT (<50ms)")
            elif results['search']['p95'] < 100:
                print("  ✓ Vector search: GOOD (<100ms)")
            else:
                print("  ⚠️ Vector search: NEEDS TUNING (>100ms)")

            if results['insert']['chunks_per_sec'] > 50:
                print("  ✅ Insert throughput: EXCELLENT (>50 chunks/sec)")
            elif results['insert']['chunks_per_sec'] > 20:
                print("  ✓ Insert throughput: GOOD (>20 chunks/sec)")
            else:
                print("  ⚠️ Insert throughput: NEEDS TUNING (<20 chunks/sec)")

        finally:
            await self.cleanup()

        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="PostgreSQL + pgvector Performance Benchmark")
    parser.add_argument("--chunks", type=int, default=1000, help="Number of chunks to test with")
    parser.add_argument("--queries", type=int, default=50, help="Number of queries to run")

    args = parser.parse_args()

    benchmark = PerformanceBenchmark(
        num_chunks=args.chunks,
        num_queries=args.queries
    )

    asyncio.run(benchmark.run_all_benchmarks())


if __name__ == "__main__":
    main()
