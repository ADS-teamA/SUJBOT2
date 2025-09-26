"""
Document Analyzer Package
Pokročilý nástroj pro analýzu dokumentů s využitím Claude Code SDK
"""

from .document_analyzer import DocumentAnalyzer, DocumentSection, AnalysisResult
from .document_reader import DocumentReader
from .question_parser import QuestionParser
from .result_aggregator import ResultAggregator
from .prompt_manager import PromptManager

__version__ = "1.0.0"
__author__ = "Document Analyzer Team"

__all__ = [
    "DocumentAnalyzer",
    "DocumentSection",
    "AnalysisResult",
    "DocumentReader",
    "QuestionParser",
    "ResultAggregator",
    "PromptManager",
]