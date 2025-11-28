"""
SSOT: Entity display configuration for Knowledge Graph visualization.

Provides unified color palettes, category mappings, and display names
for all 55+ entity types used in the Knowledge Graph.

Usage:
    from src.graph.entity_display import (
        ENTITY_COLORS,
        ENTITY_CATEGORIES,
        RELATIONSHIP_COLORS,
        CATEGORY_NAMES,
        CATEGORY_COLORS_RGB,
    )

This module is the Single Source of Truth for entity visualization.
Do NOT define these mappings elsewhere in the codebase.
"""

from typing import Dict, List, Tuple

# =============================================================================
# ENTITY COLORS (Hex) - Organized by Category
# =============================================================================

ENTITY_COLORS: Dict[str, str] = {
    # Core Entities (8) - Blues/Teals
    "standard": "#3498DB",  # Blue
    "organization": "#2980B9",  # Darker Blue
    "date": "#1ABC9C",  # Teal
    "clause": "#16A085",  # Darker Teal
    "topic": "#48C9B0",  # Light Teal
    "person": "#5DADE2",  # Light Blue
    "location": "#85C1E9",  # Very Light Blue
    "contract": "#2E86AB",  # Steel Blue
    # Regulatory Hierarchy (6) - Purples
    "regulation": "#9B59B6",  # Purple
    "decree": "#8E44AD",  # Darker Purple
    "directive": "#BB8FCE",  # Light Purple
    "treaty": "#7D3C98",  # Deep Purple
    "legal_provision": "#A569BD",  # Medium Purple
    "requirement": "#D7BDE2",  # Pale Purple
    # Authorization (2) - Oranges
    "permit": "#E67E22",  # Orange
    "license_condition": "#D35400",  # Darker Orange
    # Nuclear Technical (9) - Reds/Pinks
    "reactor": "#E74C3C",  # Red
    "facility": "#C0392B",  # Darker Red
    "system": "#F1948A",  # Light Red
    "safety_function": "#EC7063",  # Coral
    "fuel_type": "#CD6155",  # Indian Red
    "isotope": "#E57373",  # Light Coral
    "radiation_source": "#EF5350",  # Bright Red
    "waste_category": "#F44336",  # Material Red
    "dose_limit": "#FF8A80",  # Red Accent
    # Events (4) - Yellows/Ambers
    "incident": "#F39C12",  # Amber
    "emergency_classification": "#F1C40F",  # Yellow
    "inspection": "#D4AC0D",  # Dark Yellow
    "decommissioning_phase": "#F7DC6F",  # Light Yellow
    # Liability (1) - Brown
    "liability_regime": "#A0522D",  # Sienna
    # Legal Terminology (2) - Grays
    "legal_term": "#7F8C8D",  # Gray
    "definition": "#95A5A6",  # Light Gray
    # Czech Legal Types (8) - Greens
    "vyhlaska": "#27AE60",  # Green
    "narizeni": "#229954",  # Darker Green
    "sbirka_zakonu": "#58D68D",  # Light Green
    "metodicky_pokyn": "#82E0AA",  # Pale Green
    "sujb_rozhodnuti": "#1E8449",  # Forest Green
    "bezpecnostni_dokumentace": "#2ECC71",  # Emerald
    "limitni_stav": "#52BE80",  # Medium Green
    "mezni_hodnota": "#73C6B6",  # Sea Green
    # Technical Parameters (7) - Cyans
    "numeric_threshold": "#00BCD4",  # Cyan
    "measurement_unit": "#00ACC1",  # Darker Cyan
    "time_period": "#4DD0E1",  # Light Cyan
    "frequency": "#26C6DA",  # Medium Cyan
    "percentage": "#80DEEA",  # Pale Cyan
    "temperature": "#FF7043",  # Deep Orange
    "pressure": "#FFAB91",  # Light Orange
    # Process Types (5) - Indigos
    "radiation_activity": "#5C6BC0",  # Indigo
    "maintenance_activity": "#7986CB",  # Light Indigo
    "emergency_procedure": "#3F51B5",  # Material Indigo
    "training_requirement": "#9FA8DA",  # Pale Indigo
    "documentation_requirement": "#C5CAE9",  # Very Light Indigo
    # Compliance Types (3) - Deep Colors
    "compliance_gap": "#D32F2F",  # Deep Red (problems)
    "risk_factor": "#FFA000",  # Amber (warnings)
    "mitigation_measure": "#388E3C",  # Deep Green (solutions)
}

# =============================================================================
# ENTITY CATEGORIES - Maps entity type to category
# =============================================================================

ENTITY_CATEGORIES: Dict[str, str] = {
    # Core Entities
    "standard": "core",
    "organization": "core",
    "date": "core",
    "clause": "core",
    "topic": "core",
    "person": "core",
    "location": "core",
    "contract": "core",
    # Regulatory Hierarchy
    "regulation": "regulatory",
    "decree": "regulatory",
    "directive": "regulatory",
    "treaty": "regulatory",
    "legal_provision": "regulatory",
    "requirement": "regulatory",
    # Authorization
    "permit": "authorization",
    "license_condition": "authorization",
    # Nuclear Technical
    "reactor": "nuclear_technical",
    "facility": "nuclear_technical",
    "system": "nuclear_technical",
    "safety_function": "nuclear_technical",
    "fuel_type": "nuclear_technical",
    "isotope": "nuclear_technical",
    "radiation_source": "nuclear_technical",
    "waste_category": "nuclear_technical",
    "dose_limit": "nuclear_technical",
    # Events
    "incident": "events",
    "emergency_classification": "events",
    "inspection": "events",
    "decommissioning_phase": "events",
    # Liability
    "liability_regime": "liability",
    # Legal Terminology
    "legal_term": "legal_terminology",
    "definition": "legal_terminology",
    # Czech Legal Types
    "vyhlaska": "czech_legal",
    "narizeni": "czech_legal",
    "sbirka_zakonu": "czech_legal",
    "metodicky_pokyn": "czech_legal",
    "sujb_rozhodnuti": "czech_legal",
    "bezpecnostni_dokumentace": "czech_legal",
    "limitni_stav": "czech_legal",
    "mezni_hodnota": "czech_legal",
    # Technical Parameters
    "numeric_threshold": "technical_parameters",
    "measurement_unit": "technical_parameters",
    "time_period": "technical_parameters",
    "frequency": "technical_parameters",
    "percentage": "technical_parameters",
    "temperature": "technical_parameters",
    "pressure": "technical_parameters",
    # Process Types
    "radiation_activity": "processes",
    "maintenance_activity": "processes",
    "emergency_procedure": "processes",
    "training_requirement": "processes",
    "documentation_requirement": "processes",
    # Compliance Types
    "compliance_gap": "compliance",
    "risk_factor": "compliance",
    "mitigation_measure": "compliance",
}

# =============================================================================
# RELATIONSHIP COLORS (Hex)
# =============================================================================

RELATIONSHIP_COLORS: Dict[str, str] = {
    # Compliance
    "complies_with": "#27AE60",  # Green
    "contradicts": "#E74C3C",  # Red
    "partially_satisfies": "#F39C12",  # Amber
    "specifies_requirement": "#9B59B6",  # Purple
    "requires_clause": "#8E44AD",  # Dark Purple
    # Regulatory
    "implements": "#3498DB",  # Blue
    "transposes": "#2980B9",  # Dark Blue
    "superseded_by": "#95A5A6",  # Gray
    "supersedes": "#7F8C8D",  # Dark Gray
    "amends": "#1ABC9C",  # Teal
    # Document Structure
    "contains_clause": "#E67E22",  # Orange
    "contains_provision": "#D35400",  # Dark Orange
    "contains": "#F39C12",  # Amber
    "part_of": "#D4AC0D",  # Dark Amber
    # Citations
    "references": "#5DADE2",  # Light Blue
    "referenced_by": "#85C1E9",  # Very Light Blue
    "cites_provision": "#48C9B0",  # Light Teal
    "based_on": "#16A085",  # Dark Teal
    # Authorization
    "issued_by": "#9B59B6",  # Purple
    "granted_by": "#8E44AD",  # Dark Purple
    "enforced_by": "#7D3C98",  # Deep Purple
    "subject_to_inspection": "#BB8FCE",  # Light Purple
    "supervises": "#A569BD",  # Medium Purple
    # Nuclear Technical
    "regulated_by": "#E74C3C",  # Red
    "operated_by": "#C0392B",  # Dark Red
    "has_system": "#F1948A",  # Light Red
    "performs_function": "#EC7063",  # Coral
    "uses_fuel": "#CD6155",  # Indian Red
    "contains_isotope": "#E57373",  # Light Coral
    "produces_waste": "#EF5350",  # Bright Red
    "has_dose_limit": "#FF8A80",  # Red Accent
    # Temporal
    "effective_date": "#1ABC9C",  # Teal
    "expiry_date": "#16A085",  # Dark Teal
    "signed_on": "#48C9B0",  # Light Teal
    "decommissioned_on": "#73C6B6",  # Sea Green
    # Content
    "covers_topic": "#5C6BC0",  # Indigo
    "applies_to": "#7986CB",  # Light Indigo
    # Terminology
    "definition_of": "#7F8C8D",  # Gray
    # Provenance
    "mentioned_in": "#BDC3C7",  # Silver
    "defined_in": "#95A5A6",  # Light Gray
    "documented_in": "#7F8C8D",  # Gray
}

# =============================================================================
# CATEGORY DISPLAY NAMES (Czech)
# =============================================================================

CATEGORY_NAMES: Dict[str, str] = {
    "core": "Základní entity",
    "regulatory": "Regulatorní hierarchie",
    "authorization": "Autorizace",
    "nuclear_technical": "Jaderně technické",
    "events": "Události",
    "liability": "Odpovědnost",
    "legal_terminology": "Právní terminologie",
    "czech_legal": "České právní typy",
    "technical_parameters": "Technické parametry",
    "processes": "Procesní typy",
    "compliance": "Compliance typy",
}

# =============================================================================
# CATEGORY COLORS (RGB tuples for Gephi)
# =============================================================================

CATEGORY_COLORS_RGB: Dict[str, Tuple[int, int, int]] = {
    "core": (52, 152, 219),  # Blue
    "regulatory": (155, 89, 182),  # Purple
    "authorization": (230, 126, 34),  # Orange
    "nuclear_technical": (231, 76, 60),  # Red
    "events": (241, 196, 15),  # Yellow
    "liability": (160, 82, 45),  # Brown
    "legal_terminology": (127, 140, 141),  # Gray
    "czech_legal": (39, 174, 96),  # Green
    "technical_parameters": (0, 188, 212),  # Cyan
    "processes": (92, 107, 192),  # Indigo
    "compliance": (211, 47, 47),  # Deep Red
}


def get_entity_color(entity_type: str, default: str = "#808080") -> str:
    """Get hex color for an entity type."""
    return ENTITY_COLORS.get(entity_type.lower(), default)


def get_entity_category(entity_type: str, default: str = "core") -> str:
    """Get category for an entity type."""
    return ENTITY_CATEGORIES.get(entity_type.lower(), default)


def get_relationship_color(relationship_type: str, default: str = "#999999") -> str:
    """Get hex color for a relationship type."""
    return RELATIONSHIP_COLORS.get(relationship_type.lower(), default)


def get_category_name(category: str, default: str = "Ostatní") -> str:
    """Get display name for a category."""
    return CATEGORY_NAMES.get(category, default)


def get_category_rgb(category: str, default: Tuple[int, int, int] = (128, 128, 128)) -> Tuple[int, int, int]:
    """Get RGB color tuple for a category (for Gephi)."""
    return CATEGORY_COLORS_RGB.get(category, default)


def get_entities_by_category() -> Dict[str, List[str]]:
    """Get entity types grouped by category."""
    result: Dict[str, List[str]] = {}
    for entity_type, category in ENTITY_CATEGORIES.items():
        if category not in result:
            result[category] = []
        result[category].append(entity_type)
    return result
