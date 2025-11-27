"""
Unit tests for document labeling pipeline.

Tests the labeling components:
- MetadataFilter (SQL generation)
- DocumentTaxonomy (data structures)
- SectionKeywords (data structures)
- ChunkQuestions (data structures)
- LabelingConfig (configuration)

Integration tests require API keys and are marked with @pytest.mark.integration.
"""

import pytest
from typing import List

from src.config_schema import LabelingConfig
from src.storage.postgres_adapter import MetadataFilter
from src.indexing.transforms.category_extractor import DocumentTaxonomy
from src.indexing.transforms.keyword_extractor import SectionKeywords
from src.indexing.transforms.question_generator import ChunkQuestions


class TestMetadataFilter:
    """Tests for MetadataFilter SQL generation."""

    def test_empty_filter(self):
        """Empty filter should return empty conditions."""
        f = MetadataFilter()
        assert f.is_empty()
        conditions, params = f.to_sql_conditions()
        assert conditions == ""
        assert params == []

    def test_category_filter(self):
        """Category filter generates correct SQL."""
        f = MetadataFilter(category="nuclear_safety")
        assert not f.is_empty()
        conditions, params = f.to_sql_conditions()
        assert "metadata->>'category' = $1" in conditions
        assert params == ["nuclear_safety"]

    def test_categories_any_filter(self):
        """Categories ANY filter generates IN clause."""
        f = MetadataFilter(categories=["safety", "regulations", "compliance"])
        conditions, params = f.to_sql_conditions()
        assert "IN" in conditions
        assert len(params) == 3

    def test_keywords_all_filter(self):
        """Keywords ALL filter uses ?& operator."""
        f = MetadataFilter(keywords=["dozimetrie", "radiace"])
        conditions, params = f.to_sql_conditions()
        assert "?&" in conditions
        assert params == [["dozimetrie", "radiace"]]

    def test_keywords_any_filter(self):
        """Keywords ANY filter uses ?| operator."""
        f = MetadataFilter(keywords_any=["dozimetrie", "radiace"])
        conditions, params = f.to_sql_conditions()
        assert "?|" in conditions
        assert params == [["dozimetrie", "radiace"]]

    def test_entities_filter(self):
        """Entities filter uses ?| operator."""
        f = MetadataFilter(entities=["SÚJB", "ČEZ"])
        conditions, params = f.to_sql_conditions()
        assert "metadata->'entities' ?|" in conditions
        assert params == [["SÚJB", "ČEZ"]]

    def test_entity_types_filter(self):
        """Entity types filter uses ?| operator."""
        f = MetadataFilter(entity_types=["ORGANIZATION", "LEGISLATION"])
        conditions, params = f.to_sql_conditions()
        assert "metadata->'entity_types' ?|" in conditions
        assert params == [["ORGANIZATION", "LEGISLATION"]]

    def test_min_confidence_filter(self):
        """Min confidence filter casts to float."""
        f = MetadataFilter(min_confidence=0.8)
        conditions, params = f.to_sql_conditions()
        assert "category_confidence" in conditions
        assert "::float" in conditions
        assert params == [0.8]

    def test_combined_filters(self):
        """Multiple filters combine with AND."""
        f = MetadataFilter(
            category="nuclear_safety",
            keywords=["dozimetrie"],
            min_confidence=0.7
        )
        conditions, params = f.to_sql_conditions()
        assert " AND " in conditions
        assert len(params) == 3

    def test_param_offset(self):
        """Param offset adjusts parameter numbers."""
        f = MetadataFilter(category="test")
        # With offset=2, first param should be $3
        conditions, params = f.to_sql_conditions(param_offset=2)
        assert "$3" in conditions
        assert "$1" not in conditions


class TestDocumentTaxonomy:
    """Tests for DocumentTaxonomy data structure."""

    def test_default_taxonomy(self):
        """Default taxonomy has expected values."""
        t = DocumentTaxonomy.default()
        assert t.primary_category == "general"
        assert t.primary_subcategory == "uncategorized"
        assert t.secondary_categories == []
        assert t.confidence == 0.0
        assert "general" in t.taxonomy

    def test_from_dict(self):
        """Taxonomy can be created from dict (LLM response format)."""
        data = {
            "taxonomy": {
                "nuclear_safety": ["radiation_protection", "dosimetry"],
                "regulations": ["compliance"]
            },
            "document_classification": {
                "primary_category": "nuclear_safety",
                "primary_subcategory": "radiation_protection",
                "secondary_categories": ["regulations"],
                "confidence": 0.95
            }
        }
        t = DocumentTaxonomy.from_dict(data)
        assert t.primary_category == "nuclear_safety"
        assert t.primary_subcategory == "radiation_protection"
        assert "regulations" in t.secondary_categories
        assert t.confidence == 0.95

    def test_to_dict(self):
        """Taxonomy can be serialized to dict."""
        t = DocumentTaxonomy(
            taxonomy={"test": ["sub1"]},
            primary_category="test",
            primary_subcategory="sub1",
            secondary_categories=["sec1"],
            confidence=0.8
        )
        d = t.to_dict()
        assert d["primary_category"] == "test"
        assert "taxonomy" in d
        assert "primary_subcategory" in d


class TestSectionKeywords:
    """Tests for SectionKeywords data structure."""

    def test_default_keywords(self):
        """Default keywords are empty."""
        kw = SectionKeywords.default("section_1")
        assert kw.section_id == "section_1"
        assert kw.keywords == []
        assert kw.key_phrases == []

    def test_from_dict(self):
        """Keywords can be created from dict."""
        data = {
            "keywords": ["dozimetrie", "radiace", "ochrana"],
            "key_phrases": ["radiační ochrana", "dozimetrická měření"]
        }
        kw = SectionKeywords.from_dict(data, "section_1")
        assert len(kw.keywords) == 3
        assert len(kw.key_phrases) == 2

    def test_to_dict(self):
        """Keywords can be serialized to dict."""
        kw = SectionKeywords(
            section_id="s1",
            keywords=["a", "b"],
            key_phrases=["phrase 1"]
        )
        d = kw.to_dict()
        assert "keywords" in d
        assert "key_phrases" in d


class TestChunkQuestions:
    """Tests for ChunkQuestions data structure."""

    def test_default_questions(self):
        """Default questions are empty."""
        q = ChunkQuestions.default("chunk_1")
        assert q.chunk_id == "chunk_1"
        assert q.questions == []
        assert q.hyde_text == ""

    def test_from_dict(self):
        """Questions can be created from dict."""
        data = {
            "questions": [
                "Jaká je limita dávky?",
                "Kdo kontroluje radiační ochranu?"
            ]
        }
        q = ChunkQuestions.from_dict(data, "chunk_1")
        assert len(q.questions) == 2
        assert "limita" in q.hyde_text  # Questions joined

    def test_to_dict(self):
        """Questions can be serialized to dict."""
        q = ChunkQuestions(
            chunk_id="c1",
            questions=["Q1?", "Q2?"],
            hyde_text="Q1? Q2?"
        )
        d = q.to_dict()
        assert "questions" in d
        assert "hyde_text" in d


class TestLabelingConfig:
    """Tests for LabelingConfig."""

    def test_default_config(self):
        """Default config has expected values."""
        config = LabelingConfig()
        assert config.enabled is True
        assert config.model == "gpt-4o-mini"
        assert config.use_batch_api is True
        assert config.category_generation_level == "document"
        assert config.keyword_generation_level == "section"
        assert config.use_dynamic_categories is True

    def test_feature_toggles(self):
        """Feature toggles can disable components."""
        config = LabelingConfig(
            enable_categories=False,
            enable_keywords=False,
            enable_questions=True
        )
        assert config.enable_categories is False
        assert config.enable_keywords is False
        assert config.enable_questions is True

    def test_batch_api_config(self):
        """Batch API settings can be configured."""
        config = LabelingConfig(
            batch_api_poll_interval=60,
            batch_api_timeout_hours=24
        )
        assert config.batch_api_poll_interval == 60
        assert config.batch_api_timeout_hours == 24

    def test_limits(self):
        """Max limits can be configured."""
        config = LabelingConfig(
            max_keywords_per_chunk=15,
            max_questions_per_chunk=8
        )
        assert config.max_keywords_per_chunk == 15
        assert config.max_questions_per_chunk == 8


class TestLabelingPipelineImports:
    """Test that all labeling modules import correctly."""

    def test_labeling_pipeline_import(self):
        """LabelingPipeline imports correctly."""
        from src.indexing.transforms import LabelingPipeline, LabelingResult
        assert LabelingPipeline is not None
        assert LabelingResult is not None

    def test_category_extractor_import(self):
        """DocumentCategoryExtractor imports correctly."""
        from src.indexing.transforms import DocumentCategoryExtractor, DocumentTaxonomy
        assert DocumentCategoryExtractor is not None
        assert DocumentTaxonomy is not None

    def test_keyword_extractor_import(self):
        """SectionKeywordExtractor imports correctly."""
        from src.indexing.transforms import SectionKeywordExtractor, SectionKeywords
        assert SectionKeywordExtractor is not None
        assert SectionKeywords is not None

    def test_question_generator_import(self):
        """ChunkQuestionGenerator imports correctly."""
        from src.indexing.transforms import ChunkQuestionGenerator, ChunkQuestions
        assert ChunkQuestionGenerator is not None
        assert ChunkQuestions is not None

    def test_batch_processor_import(self):
        """LabelingBatchProcessor imports correctly."""
        from src.indexing.transforms import LabelingBatchProcessor
        assert LabelingBatchProcessor is not None


# Integration tests (require API keys)
@pytest.mark.integration
class TestLabelingPipelineIntegration:
    """Integration tests requiring API keys."""

    def test_pipeline_initialization(self):
        """Pipeline initializes with config."""
        from src.indexing.transforms import LabelingPipeline
        config = LabelingConfig(enabled=True)
        pipeline = LabelingPipeline(config)
        assert pipeline is not None
        assert pipeline.config.enabled is True
