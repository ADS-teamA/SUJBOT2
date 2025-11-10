"""
Phase Detection for Resume Functionality

Detects highest completed phase in output directory to enable resume.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class PhaseStatus:
    """
    Status of completed phases in output directory.

    Attributes:
        completed_phase: Highest completed phase (0-5: 0=none, 1-3=extraction/summary/chunks, 4-5=vectors/KG)
        output_dir: Output directory path
        phase_files: Dict mapping phase number to file path
        is_valid: True if all detected phases have valid JSON
        error: Error message if validation failed
    """
    completed_phase: int
    output_dir: Path
    phase_files: Dict[int, Path]
    is_valid: bool = True
    error: Optional[str] = None

    def __post_init__(self):
        """Validate invariants at construction time."""
        # Validate phase range (0-5)
        if not (0 <= self.completed_phase <= 5):
            raise ValueError(
                f"completed_phase must be in range [0, 5], got {self.completed_phase}"
            )

        # Enforce is_valid/error correlation
        if self.is_valid and self.error is not None:
            raise ValueError(
                "Invalid state: is_valid=True but error is set"
            )
        if not self.is_valid and (self.error is None or not self.error.strip()):
            raise ValueError(
                "Invalid state: is_valid=False requires non-empty error message"
            )

        # Validate phase files completeness (no gaps in sequence)
        if self.completed_phase > 0:
            expected_phases = set(range(1, self.completed_phase + 1))
            actual_phases = set(self.phase_files.keys())
            if expected_phases != actual_phases:
                raise ValueError(
                    f"Incomplete phase sequence: expected phases {sorted(expected_phases)}, "
                    f"but got {sorted(actual_phases)}"
                )
        else:
            # If completed_phase=0, phase_files must be empty
            if self.phase_files:
                raise ValueError(
                    f"Invalid state: completed_phase=0 but phase_files is not empty: {list(self.phase_files.keys())}"
                )


class PhaseDetector:
    """
    Detects completed pipeline phases for resume functionality.

    Checks output directory for phase files and validates their integrity.
    Returns highest completed phase number with complete sequence validation.
    """

    # Phase file names
    PHASE_FILES = {
        1: "phase1_extraction.json",
        2: "phase2_summaries.json",
        3: "phase3_chunks.json",
        4: "phase4_vector_store",  # Directory
    }

    @staticmethod
    def detect(output_dir: Path) -> PhaseStatus:
        """
        Detect highest completed phase in output directory.

        Args:
            output_dir: Output directory to check

        Returns:
            PhaseStatus with completed_phase and validation info

        Example:
            >>> status = PhaseDetector.detect(Path("output/BZ_VR1"))
            >>> if status.completed_phase >= 3:
            >>>     print("Can resume from phase 4")
        """
        output_dir = Path(output_dir)

        # Check if directory exists
        if not output_dir.exists():
            logger.debug(f"Output directory does not exist: {output_dir}")
            return PhaseStatus(
                completed_phase=0,
                output_dir=output_dir,
                phase_files={},
                is_valid=True
            )

        # Scan for phase files
        phase_files = {}
        highest_phase = 0

        for phase_num, filename in PhaseDetector.PHASE_FILES.items():
            file_path = output_dir / filename

            if file_path.exists():
                # Validate file/directory
                if phase_num <= 3:  # JSON files
                    if PhaseDetector._validate_json_file(file_path, phase_num):
                        phase_files[phase_num] = file_path
                        highest_phase = phase_num
                    else:
                        # Corrupted file - stop here
                        logger.warning(f"Phase {phase_num} file corrupted: {file_path}")
                        return PhaseStatus(
                            completed_phase=phase_num - 1,
                            output_dir=output_dir,
                            phase_files=phase_files,
                            is_valid=False,
                            error=f"Phase {phase_num} file corrupted"
                        )
                else:  # Directories (phase 4+)
                    if file_path.is_dir():
                        phase_files[phase_num] = file_path
                        highest_phase = phase_num

        # Validate complete sequence (no gaps)
        if not PhaseDetector._validate_sequence(phase_files, highest_phase):
            logger.warning(f"Incomplete phase sequence detected (gaps found)")
            # Find highest complete sequence
            complete_phase = 0
            for p in range(1, highest_phase + 1):
                if p in phase_files:
                    complete_phase = p
                else:
                    break

            # Keep only phases up to the complete sequence (remove phases after gap)
            complete_phase_files = {
                phase: path for phase, path in phase_files.items()
                if phase <= complete_phase
            }

            return PhaseStatus(
                completed_phase=complete_phase,
                output_dir=output_dir,
                phase_files=complete_phase_files,
                is_valid=False,
                error="Incomplete phase sequence (missing intermediate phases)"
            )

        # Check for KG file (phase 5A)
        kg_files = list(output_dir.glob("*_kg.json"))
        if kg_files:
            phase_files[5] = kg_files[0]
            if highest_phase >= 4:  # Only count KG if phase 4 exists
                highest_phase = 5

        logger.info(f"Detected completed phases: {list(phase_files.keys())} (highest: {highest_phase})")

        return PhaseStatus(
            completed_phase=highest_phase,
            output_dir=output_dir,
            phase_files=phase_files,
            is_valid=True
        )

    @staticmethod
    def _validate_json_file(file_path: Path, phase_num: int) -> bool:
        """
        Validate that JSON file is parseable and has required fields.

        Args:
            file_path: Path to JSON file
            phase_num: Phase number (1-3)

        Returns:
            True if valid, False if corrupted
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check required fields based on phase
            required_fields = {
                1: ["document_id", "sections", "source_path"],
                2: ["document_id", "document_summary", "section_summaries"],
                3: ["document_id", "chunking_stats", "layer1", "layer2", "layer3"]
            }

            fields = required_fields.get(phase_num, [])
            missing = [f for f in fields if f not in data]

            if missing:
                logger.warning(f"Phase {phase_num} missing required fields: {missing}")
                return False

            return True

        except json.JSONDecodeError as e:
            logger.warning(f"Phase {phase_num} JSON decode error: {e}")
            return False
        except Exception as e:
            logger.warning(f"Phase {phase_num} validation error: {e}")
            return False

    @staticmethod
    def _validate_sequence(phase_files: Dict[int, Path], highest: int) -> bool:
        """
        Validate that phases form complete sequence (no gaps).

        Args:
            phase_files: Dict of detected phase files
            highest: Highest phase number

        Returns:
            True if sequence is complete (1→2→3→highest), False if gaps exist
        """
        if highest == 0:
            return True

        # Check all phases from 1 to highest exist
        for phase in range(1, highest + 1):
            if phase not in phase_files:
                logger.debug(f"Gap detected: phase {phase} missing in sequence up to {highest}")
                return False

        return True
