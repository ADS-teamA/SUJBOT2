"""
Document Analyzer Package
Pokročilý nástroj pro analýzu dokumentů s hybridním vektorovým vyhledáváním
"""

from .document_reader import DocumentReader
from .indexing_pipeline import IndexingPipeline
from .hybrid_retriever import HybridRetriever
from .question_decomposer import QuestionDecomposer
from .answer_synthesizer import AnswerSynthesizer
from .claude_sdk_wrapper import ClaudeSDKClient, ClaudeCodeOptions
from .vector_store_faiss import FAISSVectorStore

__version__ = "2.0.0"
__author__ = "Document Analyzer Team"

__all__ = [
    "DocumentReader",
    "IndexingPipeline",
    "HybridRetriever",
    "QuestionDecomposer",
    "AnswerSynthesizer",
    "ClaudeSDKClient",
    "ClaudeCodeOptions",
    "FAISSVectorStore",
]