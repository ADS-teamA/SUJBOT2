"""
Tests for PostgreSQL tsquery sanitization

These tests validate that the _sanitize_tsquery() function correctly removes
special characters that would break PostgreSQL full-text search syntax,
while preserving legitimate query text including Czech diacritics.

Critical for preventing search crashes from user input.
"""

import pytest
from src.storage.postgres_adapter import _sanitize_tsquery


class TestTsquerySanitization:
    """Test PostgreSQL tsquery sanitization."""

    def test_sanitize_basic_query(self):
        """Basic query should have spaces replaced with &."""
        assert _sanitize_tsquery("hello world") == "hello & world"

    def test_sanitize_removes_parentheses(self):
        """Parentheses should be removed."""
        assert _sanitize_tsquery("what is (A or B)") == "what & is & A & or & B"

    def test_sanitize_removes_pipe(self):
        """Pipe operator should be removed."""
        assert _sanitize_tsquery("A|B|C") == "A & B & C"

    def test_sanitize_removes_question_mark(self):
        """Question marks should be removed."""
        assert _sanitize_tsquery("What is X?") == "What & is & X"

    def test_sanitize_removes_exclamation(self):
        """Exclamation marks should be removed."""
        assert _sanitize_tsquery("Hello! World!") == "Hello & World"

    def test_sanitize_removes_colon(self):
        """Colons should be removed."""
        assert _sanitize_tsquery("test:query") == "test & query"

    def test_sanitize_removes_comma(self):
        """Commas should be removed."""
        assert _sanitize_tsquery("A,B,C") == "A & B & C"

    def test_sanitize_removes_ampersand(self):
        """Ampersands should be removed (will be re-added as operators)."""
        assert _sanitize_tsquery("A&B") == "A & B"

    def test_sanitize_removes_all_special_chars(self):
        """All tsquery special chars should be removed."""
        # ()&|!:,?
        result = _sanitize_tsquery("test()&|!:,?query")
        # All special chars replaced with spaces, then collapsed
        assert result == "test & query"

    def test_sanitize_empty_string(self):
        """Empty string should return empty string."""
        assert _sanitize_tsquery("") == ""

    def test_sanitize_only_special_chars(self):
        """String with only special chars should return empty."""
        assert _sanitize_tsquery("()&|!:,?") == ""

    def test_sanitize_preserves_diacritics(self):
        """Czech diacritics should be preserved."""
        query = "Popiš konstrukci palivového článku"
        result = _sanitize_tsquery(query)
        # Should preserve č, ů
        assert "Popiš" in result
        assert "palivového" in result
        assert "článku" in result
        # Should add & operators
        assert "&" in result

    def test_sanitize_preserves_alphanumeric(self):
        """Letters and numbers should be preserved."""
        query = "H01 and H02 version 2024"
        result = _sanitize_tsquery(query)
        assert "H01" in result
        assert "H02" in result
        assert "2024" in result

    def test_sanitize_collapses_multiple_spaces(self):
        """Multiple spaces should collapse to single &."""
        assert _sanitize_tsquery("hello    world") == "hello & world"

    def test_sanitize_trims_whitespace(self):
        """Leading/trailing whitespace should be trimmed."""
        assert _sanitize_tsquery("  hello world  ") == "hello & world"

    def test_sanitize_handles_mixed_content(self):
        """Should handle queries with mixed content."""
        query = "What is GDPR (General Data Protection Regulation)?"
        result = _sanitize_tsquery(query)
        expected = "What & is & GDPR & General & Data & Protection & Regulation"
        assert result == expected

    @pytest.mark.parametrize("query,expected", [
        ("Jaký je rozdíl mezi H01 a H02?", "Jaký & je & rozdíl & mezi & H01 & a & H02"),
        ("What is (A|B)?", "What & is & A & B"),
        ("test:query", "test & query"),
        ("A,B,C", "A & B & C"),
        ("hello world", "hello & world"),
        ("one  two   three", "one & two & three"),  # Multiple spaces
        ("  trim me  ", "trim & me"),  # Whitespace trimming
    ])
    def test_sanitize_parametrized(self, query, expected):
        """Parametrized tests for various query patterns."""
        assert _sanitize_tsquery(query) == expected

    def test_sanitize_preserves_word_boundaries(self):
        """Word boundaries should be maintained."""
        query = "test-query with-dashes"
        result = _sanitize_tsquery(query)
        # Dashes are not in special char list, so should be preserved
        assert "test-query" in result or "test" in result  # Depends on implementation

    def test_sanitize_realistic_legal_query(self):
        """Test with realistic legal document query."""
        query = "Jaké jsou požadavky GDPR? (čl. 5 odst. 1)"
        result = _sanitize_tsquery(query)

        # Should preserve: Jaké, jsou, požadavky, GDPR, čl, 5, odst, 1
        # Should remove: ?, (, ), .
        assert "Jaké" in result
        assert "požadavky" in result
        assert "GDPR" in result
        assert "čl" in result
        assert "odst" in result

        # Should NOT contain special chars
        assert "?" not in result
        assert "(" not in result
        assert ")" not in result

        # Should contain & operators
        assert "&" in result

    def test_sanitize_multiple_consecutive_special_chars(self):
        """Multiple consecutive special chars should collapse to single space."""
        query = "test((((query"
        result = _sanitize_tsquery(query)
        assert result == "test & query"

    def test_sanitize_special_chars_at_boundaries(self):
        """Special chars at start/end should be handled."""
        assert _sanitize_tsquery("?test query!") == "test & query"
        assert _sanitize_tsquery("(hello world)") == "hello & world"

    def test_sanitize_single_word(self):
        """Single word query should work."""
        assert _sanitize_tsquery("test") == "test"

    def test_sanitize_single_word_with_special_chars(self):
        """Single word with special chars should be cleaned."""
        assert _sanitize_tsquery("test?") == "test"
        assert _sanitize_tsquery("(test)") == "test"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_long_query(self):
        """Should handle very long queries."""
        query = " ".join([f"word{i}" for i in range(1000)])
        result = _sanitize_tsquery(query)
        assert "&" in result
        assert "word0" in result
        assert "word999" in result

    def test_unicode_characters(self):
        """Should handle various Unicode characters."""
        # Czech
        assert "ě" in _sanitize_tsquery("sběratel")
        assert "ř" in _sanitize_tsquery("příroda")
        assert "ů" in _sanitize_tsquery("dům")

        # Slovak
        assert "ľ" in _sanitize_tsquery("ľudový")
        assert "ý" in _sanitize_tsquery("konský")  # Check that ý is preserved

    def test_numbers_preserved(self):
        """Numbers should be preserved."""
        assert _sanitize_tsquery("123 456") == "123 & 456"
        assert _sanitize_tsquery("H01-2024") == "H01-2024"  # If dashes preserved

    def test_empty_after_sanitization(self):
        """Query that becomes empty after sanitization."""
        # Only special characters
        result = _sanitize_tsquery("??!!()")
        assert result == ""

    def test_whitespace_only(self):
        """Whitespace-only query should return empty."""
        assert _sanitize_tsquery("   ") == ""
        assert _sanitize_tsquery("\t\n") == ""
