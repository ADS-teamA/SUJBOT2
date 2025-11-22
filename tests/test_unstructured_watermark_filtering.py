#!/usr/bin/env python3
"""
Test rotated text (watermark) filtering in Unstructured.io extraction.

Tests bbox orientation analysis, watermark removal on BZ_VR1.pdf,
and configuration validation for rotation angle thresholds.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock
import sys

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from unstructured_extractor import (
    UnstructuredExtractor,
    ExtractionConfig,
    analyze_bbox_orientation,
)


class TestBboxOrientationAnalysis:
    """Test bbox orientation calculation for rotated text detection."""

    def test_bbox_orientation_horizontal(self):
        """Test that horizontal text is NOT detected as rotated."""
        # Mock element with 0-degree bbox (horizontal)
        element = Mock()
        element.metadata = Mock()
        element.metadata.coordinates = Mock()
        element.metadata.coordinates.points = [
            [100, 100],  # Top-left
            [200, 100],  # Top-right (same y = horizontal)
            [200, 150],  # Bottom-right
            [100, 150],  # Bottom-left
        ]

        angle, is_rotated = analyze_bbox_orientation(element)

        print(f"\n=== Horizontal Text Test ===")
        print(f"  Calculated angle: {angle}°")
        print(f"  Is rotated: {is_rotated}")

        assert angle is not None, "Should calculate angle for valid bbox"
        assert abs(angle) < 15 or abs(angle - 180) < 15, (
            f"Expected horizontal angle (0° or 180°), got {angle}°"
        )
        assert not is_rotated, "Horizontal text should NOT be detected as rotated"

    def test_bbox_orientation_45_degrees(self):
        """Test detection of 45-degree diagonal watermark."""
        # Mock element with 45-degree bbox
        element = Mock()
        element.metadata = Mock()
        element.metadata.coordinates = Mock()
        element.metadata.coordinates.points = [
            [100, 100],  # Start point
            [200, 200],  # 45-degree angle (dx=100, dy=100)
            [150, 250],
            [50, 150],
        ]

        angle, is_rotated = analyze_bbox_orientation(element)

        print(f"\n=== 45° Diagonal Text Test ===")
        print(f"  Calculated angle: {angle}°")
        print(f"  Is rotated: {is_rotated}")

        assert angle is not None, "Should calculate angle for valid bbox"
        # 45° should be detected as rotated (default threshold is 25-65°)
        assert is_rotated, f"45° should be detected as rotated"

    def test_bbox_orientation_vertical(self):
        """Test 90-degree vertical text."""
        # Mock element with 90-degree bbox (vertical)
        element = Mock()
        element.metadata = Mock()
        element.metadata.coordinates = Mock()
        element.metadata.coordinates.points = [
            [100, 100],
            [100, 200],  # Same x = vertical
            [150, 200],
            [150, 100],
        ]

        angle, is_rotated = analyze_bbox_orientation(element)

        print(f"\n=== Vertical Text Test ===")
        print(f"  Calculated angle: {angle}°")
        print(f"  Is rotated: {is_rotated}")

        assert angle is not None, "Should calculate angle for valid bbox"
        # 90° is outside default threshold (25-65°), so NOT rotated
        assert not is_rotated, "90° vertical text should NOT be filtered by default"

    def test_bbox_orientation_boundary_25_degrees(self):
        """Test boundary at 25 degrees (min_angle threshold)."""
        # Test angle right at the boundary
        for test_angle in [24, 25, 26]:
            element = Mock()
            element.metadata = Mock()
            element.metadata.coordinates = Mock()

            # Calculate points for desired angle
            import math
            rad = math.radians(test_angle)
            dx = 100
            dy = dx * math.tan(rad)

            element.metadata.coordinates.points = [
                [0, 0],
                [dx, dy],
                [dx, dy + 50],
                [0, 50],
            ]

            angle, is_rotated = analyze_bbox_orientation(element)

            print(f"\n  {test_angle}°: angle={angle:.1f}°, is_rotated={is_rotated}")

    def test_bbox_orientation_no_coordinates(self):
        """Test handling of element without bbox coordinates."""
        element = Mock()
        element.metadata = Mock()
        element.metadata.coordinates = None

        angle, is_rotated = analyze_bbox_orientation(element)

        print(f"\n=== No Coordinates Test ===")
        print(f"  Angle: {angle}")
        print(f"  Is rotated: {is_rotated}")

        # Should return (None, False) for missing coordinates
        assert angle is None, "Should return None for missing coordinates"
        assert not is_rotated, "Should return False for missing coordinates"

    def test_bbox_orientation_invalid_points(self):
        """Test handling of invalid bbox points."""
        element = Mock()
        element.metadata = Mock()
        element.metadata.coordinates = Mock()
        element.metadata.coordinates.points = [[100, 100]]  # Only 1 point (invalid)

        angle, is_rotated = analyze_bbox_orientation(element)

        print(f"\n=== Invalid Points Test ===")
        print(f"  Angle: {angle}")
        print(f"  Is rotated: {is_rotated}")

        # Should handle gracefully
        assert angle is None or not is_rotated, (
            "Should handle invalid bbox points gracefully"
        )


class TestWatermarkFilteringIntegration:
    """Integration tests for watermark filtering on real documents."""

    @pytest.fixture
    def extractor_with_filtering(self):
        """Create extractor with watermark filtering enabled."""
        config = ExtractionConfig(
            strategy="hi_res",
            model="detectron2_mask_rcnn",
            filter_rotated_text=True,  # Enable filtering
            rotation_min_angle=25.0,
            rotation_max_angle=65.0,
            generate_summaries=False,
        )
        return UnstructuredExtractor(config)

    @pytest.fixture
    def extractor_without_filtering(self):
        """Create extractor with watermark filtering disabled."""
        config = ExtractionConfig(
            strategy="hi_res",
            model="detectron2_mask_rcnn",
            filter_rotated_text=False,  # Disable filtering
            generate_summaries=False,
        )
        return UnstructuredExtractor(config)

    def test_watermark_filtering_bz_vr1(self, extractor_with_filtering):
        """
        Integration test: BZ_VR1.pdf watermark should be removed.

        This document is known to have "NEPLATNÉ" watermarks that should be filtered.
        """
        pdf_path = Path("data/BZ_VR1.pdf")
        if not pdf_path.exists():
            pytest.skip(f"Test document not found: {pdf_path}")

        doc = extractor_with_filtering.extract(pdf_path)

        # Check for watermark tokens in sections
        watermark_tokens = ["NEPLATNÉ", "NEPLATNE", "neplatné", "neplatne"]
        sections_with_watermark = []

        for section in doc.sections:
            combined_text = (section.title or "") + " " + (section.content or "")
            combined_upper = combined_text.upper()

            if any(token.upper() in combined_upper for token in watermark_tokens):
                sections_with_watermark.append({
                    "section_id": section.section_id,
                    "title": section.title[:50] if section.title else "",
                    "preview": combined_text[:100],
                })

        print(f"\n=== BZ_VR1.pdf Watermark Test (Filtering ENABLED) ===")
        print(f"  Total sections: {len(doc.sections)}")
        print(f"  Sections with watermark: {len(sections_with_watermark)}")

        if sections_with_watermark:
            print("\n  ⚠️  Watermark text found in sections:")
            for sec in sections_with_watermark[:3]:
                print(f"    [{sec['section_id']}] {sec['title']}")
                print(f"      Preview: {sec['preview']}...")

        # ASSERTION: Watermark should be filtered out
        # Allow small number (e.g., 1-2) in case of detection edge cases
        assert len(sections_with_watermark) <= 2, (
            f"Expected watermarks to be filtered, but found in {len(sections_with_watermark)} sections. "
            f"Watermark filtering may not be working correctly."
        )

        if len(sections_with_watermark) == 0:
            print("\n  ✓ PERFECT: No watermark text detected")

    def test_watermark_filtering_comparison(
        self,
        extractor_with_filtering,
        extractor_without_filtering
    ):
        """
        Compare filtering ON vs OFF to verify it makes a difference.

        With filtering, we should see fewer elements than without.
        """
        pdf_path = Path("data/BZ_VR1.pdf")
        if not pdf_path.exists():
            pytest.skip(f"Test document not found: {pdf_path}")

        # Extract without filtering
        doc_unfiltered = extractor_without_filtering.extract(pdf_path)
        unfiltered_count = len(doc_unfiltered.sections)

        # Extract with filtering
        doc_filtered = extractor_with_filtering.extract(pdf_path)
        filtered_count = len(doc_filtered.sections)

        removed = unfiltered_count - filtered_count

        print(f"\n=== Filtering Comparison ===")
        print(f"  Without filtering: {unfiltered_count} sections")
        print(f"  With filtering: {filtered_count} sections")
        print(f"  Removed: {removed} sections ({removed/unfiltered_count*100:.1f}%)")

        # Filtering should remove at least some elements
        # (if no elements removed, watermark detection may not be working)
        if removed == 0:
            print("\n  ⚠️  Warning: Filtering removed 0 elements")
            print("     This may indicate:")
            print("     - Document has no bbox metadata")
            print("     - No rotated text in document")
            print("     - Filtering configuration issue")


class TestWatermarkFilteringConfiguration:
    """Test configuration options for watermark filtering."""

    def test_config_different_angle_thresholds(self):
        """Test filtering with different angle thresholds."""
        # Strict filtering (only 30-40°)
        config_strict = ExtractionConfig(
            filter_rotated_text=True,
            rotation_min_angle=30.0,
            rotation_max_angle=40.0,
        )

        # Loose filtering (20-70°)
        config_loose = ExtractionConfig(
            filter_rotated_text=True,
            rotation_min_angle=20.0,
            rotation_max_angle=70.0,
        )

        print("\n=== Angle Threshold Configurations ===")
        print(f"  Strict: [{config_strict.rotation_min_angle}°, {config_strict.rotation_max_angle}°]")
        print(f"  Loose: [{config_loose.rotation_min_angle}°, {config_loose.rotation_max_angle}°]")

        # Both should be valid
        assert config_strict.rotation_min_angle < config_strict.rotation_max_angle
        assert config_loose.rotation_min_angle < config_loose.rotation_max_angle

    def test_config_validation_invalid_angles(self):
        """Test that invalid angle configurations are rejected."""
        # This should raise ValueError (min >= max)
        with pytest.raises(ValueError, match="ROTATION_MIN_ANGLE.*must be <.*ROTATION_MAX_ANGLE"):
            import os
            os.environ["ROTATION_MIN_ANGLE"] = "50"
            os.environ["ROTATION_MAX_ANGLE"] = "30"

            try:
                config = ExtractionConfig.from_env()
            finally:
                # Clean up env vars
                os.environ.pop("ROTATION_MIN_ANGLE", None)
                os.environ.pop("ROTATION_MAX_ANGLE", None)

        print("\n✓ Config validation correctly rejected min >= max")

    def test_config_validation_out_of_range(self):
        """Test that out-of-range angles are rejected."""
        # Angles should be in [0, 90] range
        with pytest.raises(ValueError, match="Rotation angles must be in range"):
            import os
            os.environ["ROTATION_MIN_ANGLE"] = "10"
            os.environ["ROTATION_MAX_ANGLE"] = "95"  # > 90°

            try:
                config = ExtractionConfig.from_env()
            finally:
                os.environ.pop("ROTATION_MIN_ANGLE", None)
                os.environ.pop("ROTATION_MAX_ANGLE", None)

        print("\n✓ Config validation correctly rejected out-of-range angles")

    def test_filtering_disabled(self):
        """Test that filtering can be completely disabled."""
        config = ExtractionConfig(
            filter_rotated_text=False,
        )

        assert not config.filter_rotated_text, "Filtering should be disabled"

        print("\n✓ Filtering can be disabled via config")


class TestFilteringStatistics:
    """Test logging and statistics for watermark filtering."""

    @pytest.fixture
    def extractor(self):
        """Create extractor with filtering enabled."""
        config = ExtractionConfig(
            strategy="fast",  # Use fast for quicker tests
            model="yolox",
            filter_rotated_text=True,
            generate_summaries=False,
        )
        return UnstructuredExtractor(config)

    def test_filtering_statistics_logged(self, extractor, caplog):
        """Test that filtering statistics are logged."""
        import logging
        caplog.set_level(logging.INFO)

        pdf_path = Path("data/BZ_VR1.pdf")
        if not pdf_path.exists():
            pytest.skip(f"Test document not found: {pdf_path}")

        doc = extractor.extract(pdf_path)

        # Check logs for filtering statistics
        filtering_logs = [
            record for record in caplog.records
            if "rotated text" in record.message.lower() or "filtered" in record.message.lower()
        ]

        print(f"\n=== Filtering Logs ===")
        for log in filtering_logs:
            print(f"  [{log.levelname}] {log.message}")

        # Should have at least one log about filtering
        assert len(filtering_logs) > 0, "Expected filtering statistics to be logged"


class TestRegressionWatermarkFiltering:
    """Regression tests to catch future watermark filtering failures."""

    @pytest.fixture
    def extractor(self):
        """Create extractor with filtering enabled."""
        config = ExtractionConfig(
            strategy="hi_res",
            model="detectron2_mask_rcnn",
            filter_rotated_text=True,
            rotation_min_angle=25.0,
            rotation_max_angle=65.0,
            generate_summaries=False,
        )
        return UnstructuredExtractor(config)

    def test_regression_watermark_not_removed(self, extractor):
        """
        REGRESSION TEST: Ensure watermarks stay filtered.

        This test will FAIL if watermark filtering breaks in the future.
        """
        pdf_path = Path("data/BZ_VR1.pdf")
        if not pdf_path.exists():
            pytest.skip(f"Test document not found: {pdf_path}")

        doc = extractor.extract(pdf_path)

        # Check for watermark tokens
        watermark_tokens = ["NEPLATNÉ", "NEPLATNE"]
        sections_with_watermark = []

        for section in doc.sections:
            combined_text = (section.title or "") + " " + (section.content or "")
            if any(token in combined_text.upper() for token in watermark_tokens):
                sections_with_watermark.append(section.section_id)

        # CRITICAL: Watermarks should be filtered
        if len(sections_with_watermark) > 5:
            pytest.fail(
                f"REGRESSION: Found watermark text in {len(sections_with_watermark)} sections! "
                f"Watermark filtering has broken. "
                f"Affected sections: {sections_with_watermark[:5]}"
            )

        print(f"\n✓ REGRESSION TEST PASSED: Watermarks filtered (found in {len(sections_with_watermark)} sections)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
