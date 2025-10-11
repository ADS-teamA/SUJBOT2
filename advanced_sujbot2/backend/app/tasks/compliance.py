"""Celery tasks for compliance checking."""
import logging
import json
import os
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List

from app.core.celery_app import celery_app
from app.tasks.indexing import CallbackTask
from app.core.config import settings

# Import RAG components
from app.rag.indexing import MultiDocumentVectorStore
from app.rag.embeddings import LegalEmbedder, EmbeddingConfig
from app.rag.cross_doc_retrieval import ComparativeRetriever

# Import compliance analyzer from app.rag
from app.rag.compliance_analyzer import ComplianceAnalyzer
from app.rag.models import ComplianceReport

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, base=CallbackTask)
def compliance_check_task(
    self,
    contract_id: str,
    law_ids: list,
    mode: str = "exhaustive"
):
    """
    Run compliance check asynchronously using real ComplianceAnalyzer.

    Args:
        contract_id: Contract document ID
        law_ids: List of law document IDs
        mode: 'exhaustive' | 'sample'

    Returns:
        Compliance report
    """
    logger.info(f"Starting compliance check for contract {contract_id} against laws {law_ids}")

    try:
        # Run async analysis in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        report = loop.run_until_complete(
            _run_compliance_analysis(
                self,
                contract_id,
                law_ids,
                mode
            )
        )
        loop.close()

        logger.info(f"Compliance check complete for contract {contract_id}")

        return {
            "report_id": report["report_id"],
            "status": "completed",
            "report": report
        }

    except Exception as e:
        logger.error(f"Compliance check failed: {e}", exc_info=True)

        # Save error report
        reports_dir = os.path.join(settings.UPLOAD_DIR, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        report_path = os.path.join(reports_dir, f"{self.request.id}.json")
        with open(report_path, 'w') as f:
            json.dump({
                "status": "failed",
                "progress": 0,
                "report": None,
                "error_message": str(e)
            }, f, indent=2)

        raise


async def _run_compliance_analysis(
    task,
    contract_id: str,
    law_ids: List[str],
    mode: str
) -> dict:
    """
    Run compliance analysis (async helper).

    Args:
        task: Celery task instance
        contract_id: Contract document ID
        law_ids: Law document IDs
        mode: Analysis mode

    Returns:
        Report dictionary
    """
    # Step 1: Load vector store and embedder
    task.update_progress(5, 100, "Initializing RAG components...")

    indexes_dir = Path(settings.UPLOAD_DIR).parent / "indexes"

    embedding_config = EmbeddingConfig(
        model_name="BAAI/bge-m3",
        device="cpu",  # Use CPU for Celery workers
        batch_size=32
    )
    embedder = LegalEmbedder(embedding_config)

    vector_store = MultiDocumentVectorStore(embedder=embedder)

    # Step 2: Load contract chunks
    task.update_progress(10, 100, f"Loading contract {contract_id}...")

    contract_index_dir = indexes_dir / contract_id
    if not contract_index_dir.exists():
        raise ValueError(f"Contract index not found: {contract_id}")

    await vector_store.load_document(str(contract_index_dir))

    if contract_id not in vector_store.metadata_stores:
        raise ValueError(f"Contract {contract_id} not properly loaded")

    contract_chunks = list(vector_store.metadata_stores[contract_id].values())
    logger.info(f"Loaded {len(contract_chunks)} contract chunks")

    # Step 3: Load law chunks
    task.update_progress(20, 100, f"Loading {len(law_ids)} law documents...")

    all_law_chunks = []
    for law_id in law_ids:
        law_index_dir = indexes_dir / law_id
        if not law_index_dir.exists():
            logger.warning(f"Law index not found: {law_id}, skipping")
            continue

        await vector_store.load_document(str(law_index_dir))

        if law_id in vector_store.metadata_stores:
            law_chunks = list(vector_store.metadata_stores[law_id].values())
            all_law_chunks.extend(law_chunks)
            logger.info(f"Loaded {len(law_chunks)} chunks from law {law_id}")

    if not all_law_chunks:
        raise ValueError("No law chunks loaded")

    logger.info(f"Total law chunks: {len(all_law_chunks)}")

    # Step 4: Initialize cross-document retriever
    task.update_progress(30, 100, "Initializing cross-document retrieval...")

    cross_doc_retriever = ComparativeRetriever(
        vector_store=vector_store,
        config={}
    )

    # Step 5: Initialize ComplianceAnalyzer
    task.update_progress(35, 100, "Initializing compliance analyzer...")

    config = {
        "compliance": {
            "generate_recommendations": True,
            "confidence_threshold": 0.7
        }
    }

    analyzer = ComplianceAnalyzer(
        config=config,
        cross_doc_retriever=cross_doc_retriever
    )

    # Step 6: Run compliance analysis
    task.update_progress(40, 100, "Running compliance analysis...")

    # Define progress callback
    def analysis_progress(step: int, total: int, message: str):
        # Map analysis steps (0-6) to progress range (40-95)
        progress = 40 + int((step / total) * 55)
        task.update_progress(progress, 100, message)

    # Monkey-patch logger to capture progress
    original_info = logger.info
    def info_with_progress(msg):
        original_info(msg)
        if "Step" in msg:
            # Parse "Step X/Y: message"
            try:
                parts = msg.split(":")
                step_part = parts[0].split("/")
                if len(step_part) == 2:
                    current = int(step_part[0].replace("Step", "").strip())
                    total = int(step_part[1].strip())
                    message = ":".join(parts[1:]).strip()
                    analysis_progress(current, total, message)
            except:
                pass

    logger.info = info_with_progress

    try:
        report: ComplianceReport = await analyzer.analyze_compliance(
            contract_chunks=contract_chunks,
            law_chunks=all_law_chunks,
            contract_id=contract_id,
            law_ids=law_ids,
            mode=mode
        )
    finally:
        logger.info = original_info

    # Step 7: Convert report to dictionary
    task.update_progress(95, 100, "Generating report...")

    report_dict = report.to_dict()

    # Step 8: Save report
    reports_dir = os.path.join(settings.UPLOAD_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    report_path = os.path.join(reports_dir, f"{task.request.id}.json")
    with open(report_path, 'w') as f:
        json.dump({
            "status": "completed",
            "progress": 100,
            "report": report_dict
        }, f, indent=2)

    task.update_progress(100, 100, "Report complete")

    return report_dict
