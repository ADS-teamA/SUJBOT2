"""
Report generation (Markdown + JSON).

Generates human-readable and machine-readable benchmark reports.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from .runner import BenchmarkResult

logger = logging.getLogger(__name__)


def save_json_report(result: BenchmarkResult, output_dir: str) -> Path:
    """
    Save benchmark results as JSON.

    Args:
        result: Benchmark results
        output_dir: Directory to save report

    Returns:
        Path to saved JSON file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_{timestamp}.json"
    filepath = output_path / filename

    # Serialize to JSON
    data = result.to_dict()

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ JSON report saved: {filepath}")
    except (IOError, OSError) as e:
        import time

        error_id = f"ERR_JSON_WRITE_{int(time.time())}"
        logger.error(f"[{error_id}] Failed to write JSON report to {filepath}: {e}", exc_info=True)
        raise RuntimeError(
            f"[{error_id}] JSON report generation failed. "
            f"Check disk space and permissions. "
            f"Benchmark results are in memory but not saved to {filepath}"
        ) from e

    return filepath


def save_markdown_report(result: BenchmarkResult, output_dir: str) -> Path:
    """
    Save benchmark results as Markdown.

    Args:
        result: Benchmark results
        output_dir: Directory to save report

    Returns:
        Path to saved Markdown file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_report_{timestamp}.md"
    filepath = output_path / filename

    # Generate Markdown content
    md_content = _generate_markdown(result)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info(f"✓ Markdown report saved: {filepath}")
    except (IOError, OSError) as e:
        import time

        error_id = f"ERR_MD_WRITE_{int(time.time())}"
        logger.error(
            f"[{error_id}] Failed to write Markdown report to {filepath}: {e}", exc_info=True
        )
        raise RuntimeError(
            f"[{error_id}] Markdown report generation failed. "
            f"Check disk space and permissions. "
            f"Benchmark results are in memory but not saved to {filepath}"
        ) from e

    return filepath


def _generate_markdown(result: BenchmarkResult) -> str:
    """
    Generate Markdown report content.

    Args:
        result: Benchmark results

    Returns:
        Markdown string
    """
    lines = []

    # Header
    lines.append(f"# Benchmark Report: {result.dataset_name}")
    lines.append("")
    lines.append(f"**Timestamp:** {result.timestamp}")
    lines.append(f"**Total Queries:** {result.total_queries}")
    lines.append(f"**Total Time:** {result.total_time_seconds:.1f}s")
    lines.append(f"**Total Cost:** ${result.total_cost_usd:.4f}")
    lines.append("")

    # Aggregate Metrics
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")

    metric_labels = {
        "exact_match": "Exact Match (EM)",
        "f1_score": "F1 Score",
        "precision": "Precision",
        "recall": "Recall",
        "rag_confidence": "RAG Confidence (Avg)",
    }

    for metric_name in sorted(result.aggregate_metrics.keys()):
        score = result.aggregate_metrics[metric_name]
        label = metric_labels.get(metric_name, metric_name.replace("_", " ").title())
        lines.append(f"| {label} | {score:.4f} |")

    lines.append("")

    # Performance Metrics
    lines.append("## Performance Metrics")
    lines.append("")

    # Avoid division by zero if no queries were evaluated
    if result.total_queries > 0:
        avg_time_ms = (result.total_time_seconds * 1000) / result.total_queries
        cost_per_query = result.total_cost_usd / result.total_queries
    else:
        avg_time_ms = 0.0
        cost_per_query = 0.0

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Avg Time per Query | {avg_time_ms:.0f}ms |")
    lines.append(f"| Cost per Query | ${cost_per_query:.6f} |")
    lines.append("")

    # Configuration
    lines.append("## Configuration")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(result.config, indent=2))
    lines.append("```")
    lines.append("")

    # Per-Query Results (top 10 + bottom 10 by F1)
    lines.append("## Per-Query Results")
    lines.append("")

    # Sort by F1 score
    sorted_results = sorted(
        result.query_results,
        key=lambda qr: qr.metrics.get("f1_score", 0),
        reverse=True,
    )

    # Top N best (dynamic based on actual count)
    num_top = min(10, len(sorted_results))
    lines.append(f"### Top {num_top} Best Results (by F1 Score)")
    lines.append("")
    lines.append(
        "| Query ID | F1 | EM | Precision | Recall | RAG Conf | Time (ms) | Query Preview |"
    )
    lines.append(
        "|----------|----|----|-----------|--------|----------|-----------|---------------|"
    )

    for qr in sorted_results[:num_top]:
        query_preview = qr.query[:60] + "..." if len(qr.query) > 60 else qr.query
        conf_str = f"{qr.rag_confidence:.3f}" if qr.rag_confidence is not None else "N/A"
        lines.append(
            f"| {qr.query_id} | "
            f"{qr.metrics.get('f1_score', 0):.3f} | "
            f"{qr.metrics.get('exact_match', 0):.0f} | "
            f"{qr.metrics.get('precision', 0):.3f} | "
            f"{qr.metrics.get('recall', 0):.3f} | "
            f"{conf_str} | "
            f"{qr.retrieval_time_ms:.0f} | "
            f"{query_preview} |"
        )

    lines.append("")

    # Bottom N worst (dynamic based on actual count)
    num_bottom = min(10, len(sorted_results))
    lines.append(f"### Bottom {num_bottom} Worst Results (by F1 Score)")
    lines.append("")
    lines.append(
        "| Query ID | F1 | EM | Precision | Recall | RAG Conf | Time (ms) | Query Preview |"
    )
    lines.append(
        "|----------|----|----|-----------|--------|----------|-----------|---------------|"
    )

    for qr in sorted_results[-num_bottom:]:
        query_preview = qr.query[:60] + "..." if len(qr.query) > 60 else qr.query
        conf_str = f"{qr.rag_confidence:.3f}" if qr.rag_confidence is not None else "N/A"
        lines.append(
            f"| {qr.query_id} | "
            f"{qr.metrics.get('f1_score', 0):.3f} | "
            f"{qr.metrics.get('exact_match', 0):.0f} | "
            f"{qr.metrics.get('precision', 0):.3f} | "
            f"{qr.metrics.get('recall', 0):.3f} | "
            f"{conf_str} | "
            f"{qr.retrieval_time_ms:.0f} | "
            f"{query_preview} |"
        )

    lines.append("")

    # Interpretation Guide
    lines.append("## Metric Interpretation")
    lines.append("")
    lines.append(
        "- **Exact Match (EM)**: 1.0 if prediction exactly matches any ground truth, else 0.0"
    )
    lines.append("- **F1 Score**: Harmonic mean of precision and recall (token-level)")
    lines.append("- **Precision**: Fraction of predicted tokens that are correct")
    lines.append("- **Recall**: Fraction of ground truth tokens that were predicted")
    lines.append(
        "- **RAG Confidence**: Retrieval quality score (0-1). "
        "≥0.85: High (automated), 0.70-0.84: Medium (review recommended), "
        "0.50-0.69: Low (mandatory review), <0.50: Very low (expert required)"
    )
    lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    return "\n".join(lines)


def save_per_query_reports(result: BenchmarkResult, output_dir: str) -> None:
    """
    Save individual JSON files for each query (debug mode).

    Args:
        result: Benchmark results
        output_dir: Base output directory
    """
    per_query_dir = Path(output_dir) / "per_query"
    per_query_dir.mkdir(parents=True, exist_ok=True)

    for qr in result.query_results:
        filename = f"query_{qr.query_id:03d}.json"
        filepath = per_query_dir / filename

        data = qr.to_dict()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Per-query reports saved: {per_query_dir} ({len(result.query_results)} files)")


def generate_reports(
    result: BenchmarkResult,
    output_dir: str,
    save_markdown: bool = True,
    save_json: bool = True,
    save_per_query: bool = False,
) -> dict:
    """
    Generate all requested reports.

    Args:
        result: Benchmark results
        output_dir: Output directory
        save_markdown: Generate Markdown report
        save_json: Generate JSON report
        save_per_query: Generate per-query JSON files

    Returns:
        Dict with paths to generated reports
    """
    paths = {}

    if save_json:
        paths["json"] = save_json_report(result, output_dir)

    if save_markdown:
        paths["markdown"] = save_markdown_report(result, output_dir)

    if save_per_query:
        save_per_query_reports(result, output_dir)
        paths["per_query_dir"] = Path(output_dir) / "per_query"

    return paths
