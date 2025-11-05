r"""
LaTeX-aware document parser

Specialized parser for LaTeX files that:
1. Recognizes \section, \subsection, \subsubsection hierarchy
2. Converts LaTeX commands to plain text
3. Builds proper hierarchical DocumentSection structure

Usage:
    from src.latex_parser import parse_latex_document

    doc = parse_latex_document(Path("paper.tex"))
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

try:
    from pylatexenc.latex2text import LatexNodes2Text
except ImportError:
    LatexNodes2Text = None
    logging.warning("pylatexenc not installed - LaTeX cleaning will be limited")

logger = logging.getLogger(__name__)


def clean_latex_text(text: str) -> str:
    """
    Clean LaTeX markup from text.

    Converts LaTeX commands to plain text.

    Args:
        text: Raw LaTeX text

    Returns:
        Cleaned plain text
    """
    if LatexNodes2Text is not None:
        try:
            converter = LatexNodes2Text()
            return converter.latex_to_text(text)
        except Exception as e:
            logger.warning(f"Failed to convert LaTeX with pylatexenc: {e}")

    # Fallback: simple regex-based cleaning
    # Remove common LaTeX commands
    cleaned = text

    # Remove \cite{...}
    cleaned = re.sub(r'\\cite\{[^}]*\}', '', cleaned)

    # Remove \ref{...}, \label{...}
    cleaned = re.sub(r'\\(ref|label)\{[^}]*\}', '', cleaned)

    # Remove \uv{text} → "text"
    cleaned = re.sub(r'\\uv\{([^}]*)\}', r'"\1"', cleaned)

    # Remove \textbf{text} → text
    cleaned = re.sub(r'\\text(bf|it|tt|sc)\{([^}]*)\}', r'\2', cleaned)

    # Remove environments: \begin{itemize}, \end{itemize}, etc.
    cleaned = re.sub(r'\\(begin|end)\{[^}]*\}', '', cleaned)

    # Remove \item
    cleaned = re.sub(r'\\item\s*', '• ', cleaned)

    # Remove \par
    cleaned = re.sub(r'\\par\b', '', cleaned)

    # Replace ~ with space
    cleaned = cleaned.replace('~', ' ')

    # Remove other backslash commands
    cleaned = re.sub(r'\\[a-zA-Z]+\s*', '', cleaned)

    # Clean up multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)

    return cleaned.strip()


def parse_latex_sections(content: str) -> List[dict]:
    r"""
    Parse LaTeX sectioning commands from content.

    Extracts \chapter, \section, \subsection, \subsubsection, \paragraph
    and their hierarchy levels.

    Args:
        content: LaTeX file content

    Returns:
        List of dicts with section metadata
    """
    sections = []

    # LaTeX sectioning hierarchy
    # Lower level number = higher in hierarchy
    section_patterns = [
        (r'\\chapter\*?\{([^}]+)\}', 0, 'chapter'),
        (r'\\section\*?\{([^}]+)\}', 1, 'section'),
        (r'\\subsection\*?\{([^}]+)\}', 2, 'subsection'),
        (r'\\subsubsection\*?\{([^}]+)\}', 3, 'subsubsection'),
        (r'\\paragraph\*?\{([^}]+)\}', 4, 'paragraph'),
    ]

    lines = content.split('\n')
    char_offset = 0

    for line_num, line in enumerate(lines, 1):
        line_start_offset = char_offset

        for pattern, level, name in section_patterns:
            match = re.search(pattern, line)
            if match:
                title_raw = match.group(1)
                title_clean = clean_latex_text(title_raw)

                # Find content until next section
                # (simplified - just take lines until next section command)
                content_lines = []
                for future_line in lines[line_num:]:  # Start from next line
                    # Stop if we hit another section command
                    if any(re.search(p[0], future_line) for p in section_patterns):
                        break
                    content_lines.append(future_line)

                content_raw = '\n'.join(content_lines[:20])  # Limit to first 20 lines
                content_clean = clean_latex_text(content_raw)

                sections.append({
                    'line_num': line_num,
                    'level': level,
                    'type': name,
                    'title_raw': title_raw,
                    'title_clean': title_clean,
                    'content_raw': content_raw,
                    'content_clean': content_clean,
                    'char_start': line_start_offset,
                    'char_end': line_start_offset + len(line),
                })

                logger.debug(f"Found {name} at line {line_num}: {title_clean[:50]}")
                break  # Only match first pattern per line

        char_offset += len(line) + 1  # +1 for newline

    logger.info(f"Parsed {len(sections)} LaTeX sections")
    return sections


def build_hierarchy(sections: List[dict]) -> List[dict]:
    """
    Build parent-child relationships for LaTeX sections.

    Args:
        sections: List of section dicts from parse_latex_sections

    Returns:
        Sections with added parent_id, children_ids, depth, ancestors
    """
    if not sections:
        return []

    # Add hierarchy metadata
    for i, section in enumerate(sections):
        section['section_id'] = f"sec_{i+1}"
        section['parent_id'] = None
        section['children_ids'] = []
        section['ancestors'] = []
        section['depth'] = 1

    # Build parent-child relationships
    for i, section in enumerate(sections):
        current_level = section['level']

        # Look backwards for parent (first section with lower level number)
        for j in range(i - 1, -1, -1):
            if sections[j]['level'] < current_level:
                # Found parent
                section['parent_id'] = sections[j]['section_id']
                sections[j]['children_ids'].append(section['section_id'])

                # Calculate depth
                section['depth'] = sections[j]['depth'] + 1

                # Build ancestors list
                ancestors = sections[j]['ancestors'].copy()
                ancestors.append(sections[j]['title_clean'])
                section['ancestors'] = ancestors

                break

    # Build path
    for section in sections:
        path_parts = section['ancestors'] + [section['title_clean']]
        section['path'] = ' > '.join(path_parts)

    return sections


def parse_latex_document(latex_path: Path) -> dict:
    """
    Parse LaTeX document with proper hierarchy extraction.

    Returns data compatible with ExtractedDocument structure.

    Args:
        latex_path: Path to .tex file

    Returns:
        Dictionary with document metadata and sections
    """
    logger.info(f"Parsing LaTeX document: {latex_path.name}")

    with open(latex_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse sections
    sections_raw = parse_latex_sections(content)

    if not sections_raw:
        logger.warning("No LaTeX sections found - file may not use \\section commands")
        return None

    # Build hierarchy
    sections = build_hierarchy(sections_raw)

    # Calculate statistics
    max_depth = max(s['depth'] for s in sections)
    num_roots = sum(1 for s in sections if s['depth'] == 1)

    # Full text (cleaned)
    full_text_clean = clean_latex_text(content)

    result = {
        'source_path': str(latex_path),
        'document_id': latex_path.stem,
        'sections': sections,
        'num_sections': len(sections),
        'hierarchy_depth': max_depth,
        'num_roots': num_roots,
        'full_text': full_text_clean,
        'total_chars': len(full_text_clean),
    }

    logger.info(
        f"LaTeX parsing complete: {len(sections)} sections, "
        f"depth={max_depth}, roots={num_roots}"
    )

    return result


# Example usage and testing
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python src/latex_parser.py <file.tex>")
        sys.exit(1)

    latex_file = Path(sys.argv[1])

    if not latex_file.exists():
        print(f"File not found: {latex_file}")
        sys.exit(1)

    result = parse_latex_document(latex_file)

    if result:
        print(f"\nDocument: {result['document_id']}")
        print(f"Sections: {result['num_sections']}")
        print(f"Hierarchy depth: {result['hierarchy_depth']}")
        print(f"Root sections: {result['num_roots']}")
        print(f"\nFirst 10 sections:")

        for section in result['sections'][:10]:
            indent = "  " * (section['depth'] - 1)
            print(f"  {indent}[{section['type']}] {section['title_clean'][:60]}")
    else:
        print("Failed to parse LaTeX document")
