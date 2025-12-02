"""
Graphiti Entity Type Definitions for Czech Legal/Nuclear Document Processing.

This module defines 55 Pydantic entity models for GPT-4o-mini structured extraction via Graphiti.
Expands the original 32 EntityTypes from models.py with domain-specific types.

New Entity Categories:
- Czech Legal Types (+8): vyhlaska, narizeni, sbirka_zakonu, metodicky_pokyn, etc.
- Technical Parameters (+7): numeric_threshold, measurement_unit, time_period, etc.
- Process Types (+5): radiation_activity, maintenance_activity, emergency_procedure, etc.
- Compliance Types (+3): compliance_gap, risk_factor, mitigation_measure

Architecture:
- Pydantic models provide schema validation for GPT-4o-mini structured output
- Each model maps to Graphiti's EntityType for storage in Neo4j
- Backward compatible with models.py EntityType enum
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

import re

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# EXTENDED ENTITY TYPE ENUM (55 types)
# =============================================================================


class GraphitiEntityType(str, Enum):
    """
    Extended entity types for Graphiti extraction (55 total).

    Superset of models.EntityType (32) + 23 new domain-specific types.
    String-based enum for Graphiti compatibility.
    """

    # ==================== CORE ENTITIES (8) - from models.py ====================
    STANDARD = "standard"
    ORGANIZATION = "organization"
    DATE = "date"
    CLAUSE = "clause"
    TOPIC = "topic"
    PERSON = "person"
    LOCATION = "location"
    CONTRACT = "contract"

    # ==================== REGULATORY HIERARCHY (6) - from models.py ====================
    REGULATION = "regulation"
    DECREE = "decree"
    DIRECTIVE = "directive"
    TREATY = "treaty"
    LEGAL_PROVISION = "legal_provision"
    REQUIREMENT = "requirement"

    # ==================== AUTHORIZATION (2) - from models.py ====================
    PERMIT = "permit"
    LICENSE_CONDITION = "license_condition"

    # ==================== NUCLEAR TECHNICAL (9) - from models.py ====================
    REACTOR = "reactor"
    FACILITY = "facility"
    SYSTEM = "system"
    SAFETY_FUNCTION = "safety_function"
    FUEL_TYPE = "fuel_type"
    ISOTOPE = "isotope"
    RADIATION_SOURCE = "radiation_source"
    WASTE_CATEGORY = "waste_category"
    DOSE_LIMIT = "dose_limit"

    # ==================== EVENTS (4) - from models.py ====================
    INCIDENT = "incident"
    EMERGENCY_CLASSIFICATION = "emergency_classification"
    INSPECTION = "inspection"
    DECOMMISSIONING_PHASE = "decommissioning_phase"

    # ==================== LIABILITY (1) - from models.py ====================
    LIABILITY_REGIME = "liability_regime"

    # ==================== LEGAL TERMINOLOGY (2) - from models.py ====================
    LEGAL_TERM = "legal_term"
    DEFINITION = "definition"

    # ==================== NEW: CZECH LEGAL TYPES (+8) ====================
    VYHLASKA = "vyhlaska"  # Czech decree (vyhláška č. XXX/YYYY Sb.)
    NARIZENI = "narizeni"  # Government regulation (nařízení vlády)
    SBIRKA_ZAKONU = "sbirka_zakonu"  # Collection of Laws reference
    METODICKY_POKYN = "metodicky_pokyn"  # Methodological guideline
    SUJB_ROZHODNUTI = "sujb_rozhodnuti"  # SÚJB regulatory decision
    BEZPECNOSTNI_DOKUMENTACE = "bezpecnostni_dokumentace"  # Safety documentation
    LIMITNI_STAV = "limitni_stav"  # Limiting condition for operation (LCO)
    MEZNI_HODNOTA = "mezni_hodnota"  # Threshold value

    # ==================== NEW: TECHNICAL PARAMETERS (+7) ====================
    NUMERIC_THRESHOLD = "numeric_threshold"  # Numeric limits (max 5 MW, <0.5 mSv)
    MEASUREMENT_UNIT = "measurement_unit"  # Units (Bq, Sv, mSv, kBq)
    TIME_PERIOD = "time_period"  # Periods (5 let, 30 dnů)
    FREQUENCY = "frequency"  # Frequency (denně, měsíčně)
    PERCENTAGE = "percentage"  # Percentages (19.75%)
    TEMPERATURE = "temperature"  # Temperatures (350°C)
    PRESSURE = "pressure"  # Pressures (15 MPa)

    # ==================== NEW: PROCESS TYPES (+5) ====================
    RADIATION_ACTIVITY = "radiation_activity"  # Radiation activities (měření, monitoring)
    MAINTENANCE_ACTIVITY = "maintenance_activity"  # Maintenance (revize, oprava)
    EMERGENCY_PROCEDURE = "emergency_procedure"  # Emergency procedures
    TRAINING_REQUIREMENT = "training_requirement"  # Training requirements
    DOCUMENTATION_REQUIREMENT = "documentation_requirement"  # Documentation requirements

    # ==================== NEW: COMPLIANCE TYPES (+3) ====================
    COMPLIANCE_GAP = "compliance_gap"  # Identified non-compliance
    RISK_FACTOR = "risk_factor"  # Identified risk
    MITIGATION_MEASURE = "mitigation_measure"  # Risk mitigation action


# =============================================================================
# BASE PYDANTIC MODELS FOR GRAPHITI
# =============================================================================


class GraphitiEntityBase(BaseModel):
    """
    Base model for all Graphiti entities.

    Provides common fields required by Graphiti for node creation.
    Note: 'name' and 'summary' are reserved attributes in Graphiti,
    so we use 'label' and 'description' instead.

    Important: Uses extra="forbid" to generate additionalProperties: false
    in JSON schema, which is required for OpenAI structured outputs.
    """

    model_config = {"extra": "forbid", "use_enum_values": True}

    label: str = Field(
        ...,
        description="Primary name/value of the entity",
        min_length=1,
        max_length=500
    )
    entity_type: GraphitiEntityType = Field(
        ...,
        description="Type classification of the entity"
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score (0.0-1.0)"
    )
    entity_description: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Brief description of the entity"
    )


# =============================================================================
# CZECH LEGAL ENTITY MODELS
# =============================================================================


class VyhlaskaEntity(GraphitiEntityBase):
    """
    Czech decree (vyhláška).

    Example: "vyhláška č. 422/2016 Sb." - implementing regulation for nuclear facilities.
    """

    entity_type: Literal[GraphitiEntityType.VYHLASKA] = GraphitiEntityType.VYHLASKA
    cislo: Optional[str] = Field(default=None, description="Decree number (e.g., '422/2016')")
    rok: Optional[int] = Field(default=None, ge=1900, le=2100, description="Year of issue")
    sbirka: Literal["Sb.", "Sb.m.s."] = Field(default="Sb.", description="Collection type")
    nazev: Optional[str] = Field(default=None, description="Full title in Czech")
    ministerstvo: Optional[str] = Field(default=None, description="Issuing ministry")
    ucinnost_od: Optional[datetime] = Field(default=None, description="Effective from date")
    parent_zakon: Optional[str] = Field(default=None, description="Parent law reference")

    @field_validator("cislo")
    @classmethod
    def validate_cislo(cls, v: Optional[str]) -> Optional[str]:
        """Validate Czech decree number format (e.g., '422/2016')."""
        if v is not None and not re.match(r"^\d+/\d{4}$", v):
            raise ValueError(f"Invalid decree number format: '{v}'. Expected format: '422/2016'")
        return v


class NarizeniEntity(GraphitiEntityBase):
    """
    Government regulation (nařízení vlády).

    Example: "nařízení vlády č. 361/2007 Sb." - occupational health requirements.
    """

    entity_type: Literal[GraphitiEntityType.NARIZENI] = GraphitiEntityType.NARIZENI
    cislo: Optional[str] = Field(default=None, description="Regulation number")
    rok: Optional[int] = Field(default=None, ge=1900, le=2100, description="Year of issue")
    nazev: Optional[str] = Field(default=None, description="Full title in Czech")
    ucinnost_od: Optional[datetime] = Field(default=None, description="Effective from date")

    @field_validator("cislo")
    @classmethod
    def validate_cislo(cls, v: Optional[str]) -> Optional[str]:
        """Validate Czech regulation number format (e.g., '361/2007')."""
        if v is not None and not re.match(r"^\d+/\d{4}$", v):
            raise ValueError(f"Invalid regulation number format: '{v}'. Expected format: '361/2007'")
        return v


class SbirkaZakonuEntity(GraphitiEntityBase):
    """
    Collection of Laws reference (Sbírka zákonů).

    Standardized reference format for Czech legislation.
    """

    entity_type: Literal[GraphitiEntityType.SBIRKA_ZAKONU] = GraphitiEntityType.SBIRKA_ZAKONU
    cislo: str = Field(..., description="Reference number (e.g., '263/2016')")
    typ: Literal["zákon", "vyhláška", "nařízení", "ústavní zákon", "jiné"] = Field(
        default="zákon",
        description="Type of legal act"
    )
    castka: Optional[int] = Field(default=None, description="Issue number (částka)")

    @field_validator("cislo")
    @classmethod
    def validate_cislo(cls, v: str) -> str:
        """Validate Czech legal reference number format (e.g., '263/2016')."""
        if not re.match(r"^\d+/\d{4}$", v):
            raise ValueError(f"Invalid reference number format: '{v}'. Expected format: '263/2016'")
        return v


class MetodickyPokynEntity(GraphitiEntityBase):
    """
    Methodological guideline (metodický pokyn).

    Non-binding guidance document from SÚJB or other authorities.
    """

    entity_type: Literal[GraphitiEntityType.METODICKY_POKYN] = GraphitiEntityType.METODICKY_POKYN
    vydavatel: str = Field(default="SÚJB", description="Issuing authority")
    oblast: Optional[str] = Field(default=None, description="Subject area")
    datum_vydani: Optional[datetime] = Field(default=None, description="Issue date")
    zavaznost: Literal["doporučující", "závazný", "informativní"] = Field(
        default="doporučující",
        description="Binding nature"
    )


class SujbRozhodnutiEntity(GraphitiEntityBase):
    """
    SÚJB regulatory decision (rozhodnutí SÚJB).

    Administrative decision by the State Office for Nuclear Safety.
    """

    entity_type: Literal[GraphitiEntityType.SUJB_ROZHODNUTI] = GraphitiEntityType.SUJB_ROZHODNUTI
    cislo_jednaci: Optional[str] = Field(default=None, description="File reference number")
    typ_rozhodnuti: Optional[str] = Field(default=None, description="Decision type")
    datum_rozhodnuti: Optional[datetime] = Field(default=None, description="Decision date")
    pravni_moc: Optional[datetime] = Field(default=None, description="Date of legal force")
    adresat: Optional[str] = Field(default=None, description="Addressee (facility/operator)")


class BezpecnostniDokumentaceEntity(GraphitiEntityBase):
    """
    Safety documentation (bezpečnostní dokumentace).

    Required documentation for nuclear facility licensing.
    """

    entity_type: Literal[GraphitiEntityType.BEZPECNOSTNI_DOKUMENTACE] = GraphitiEntityType.BEZPECNOSTNI_DOKUMENTACE
    typ_dokumentace: Literal[
        "předběžná bezpečnostní zpráva",
        "bezpečnostní zpráva",
        "limits and conditions",
        "havarijní řád",
        "jiné"
    ] = Field(..., description="Documentation type")
    faze: Optional[int] = Field(default=None, ge=1, le=5, description="Licensing phase (§9)")
    verze: Optional[str] = Field(default=None, description="Version number")


class LimitniStavEntity(GraphitiEntityBase):
    """
    Limiting condition for operation (limitní stav - LCO).

    Operational limit specified in facility license.
    """

    entity_type: Literal[GraphitiEntityType.LIMITNI_STAV] = GraphitiEntityType.LIMITNI_STAV
    parametr: str = Field(..., description="Parameter name")
    hodnota: float = Field(..., description="Limit value")
    jednotka: str = Field(..., description="Unit of measurement")
    typ_limitu: Literal["max", "min", "rozsah", "toleranční"] = Field(
        default="max",
        description="Limit type"
    )
    akce_pri_prekroceni: Optional[str] = Field(default=None, description="Action if exceeded")


class MezniHodnotaEntity(GraphitiEntityBase):
    """
    Threshold value (mezní hodnota).

    Regulatory threshold for various parameters.
    """

    entity_type: Literal[GraphitiEntityType.MEZNI_HODNOTA] = GraphitiEntityType.MEZNI_HODNOTA
    hodnota: float = Field(..., description="Threshold value")
    jednotka: str = Field(..., description="Unit")
    kategorie: Optional[str] = Field(default=None, description="Category (e.g., 'radiační ochrana')")
    zdroj: Optional[str] = Field(default=None, description="Source regulation")


# =============================================================================
# TECHNICAL PARAMETER MODELS
# =============================================================================


class NumericThresholdEntity(GraphitiEntityBase):
    """
    Numeric threshold/limit (číselný limit).

    Example: "max 5 MW", "<0.5 mSv/rok", "minimálně 30 dnů"
    """

    entity_type: Literal[GraphitiEntityType.NUMERIC_THRESHOLD] = GraphitiEntityType.NUMERIC_THRESHOLD
    hodnota: float = Field(..., description="Numeric value")
    jednotka: str = Field(..., description="Unit")
    operator: Literal["=", "<", "<=", ">", ">=", "~", "rozsah"] = Field(
        default="=",
        description="Comparison operator"
    )
    kontext: Optional[str] = Field(default=None, description="Context of the threshold")


class MeasurementUnitEntity(GraphitiEntityBase):
    """
    Measurement unit (měřicí jednotka).

    Example: Bq, Sv, mSv, kBq, GBq, MW, kW
    """

    entity_type: Literal[GraphitiEntityType.MEASUREMENT_UNIT] = GraphitiEntityType.MEASUREMENT_UNIT
    symbol: str = Field(..., description="Unit symbol")
    nazev: Optional[str] = Field(default=None, description="Full name")
    kategorie: Literal[
        "aktivita", "dávka", "výkon", "tlak", "teplota", "délka", "hmotnost", "jiné"
    ] = Field(..., description="Unit category")
    si_zaklad: Optional[str] = Field(default=None, description="SI base unit")


class TimePeriodEntity(GraphitiEntityBase):
    """
    Time period (časové období).

    Example: "5 let", "30 dnů", "12 měsíců", "kalendářní rok"
    """

    entity_type: Literal[GraphitiEntityType.TIME_PERIOD] = GraphitiEntityType.TIME_PERIOD
    hodnota: Optional[float] = Field(default=None, description="Numeric value")
    jednotka: Literal["sekunda", "minuta", "hodina", "den", "týden", "měsíc", "rok"] = Field(
        ...,
        description="Time unit"
    )
    typ: Literal["délka", "lhůta", "perioda", "interval"] = Field(
        default="délka",
        description="Period type"
    )


class FrequencyEntity(GraphitiEntityBase):
    """
    Frequency specification (frekvence).

    Example: "denně", "měsíčně", "každých 6 měsíců", "nepřetržitě"
    """

    entity_type: Literal[GraphitiEntityType.FREQUENCY] = GraphitiEntityType.FREQUENCY
    hodnota: Optional[str] = Field(default=None, description="Frequency value")
    typ: Literal["pravidelná", "nepravidelná", "kontinuální", "jednorázová"] = Field(
        default="pravidelná",
        description="Frequency type"
    )


class PercentageEntity(GraphitiEntityBase):
    """
    Percentage value (procentuální hodnota).

    Example: "19.75%", "maximálně 5%", "alespoň 80%"
    """

    entity_type: Literal[GraphitiEntityType.PERCENTAGE] = GraphitiEntityType.PERCENTAGE
    hodnota: float = Field(..., ge=0.0, le=100.0, description="Percentage value")
    kontext: Optional[str] = Field(default=None, description="Context")


class TemperatureEntity(GraphitiEntityBase):
    """
    Temperature value (teplota).

    Example: "350°C", "maximálně 100°C", "provozní teplota 280°C"
    """

    entity_type: Literal[GraphitiEntityType.TEMPERATURE] = GraphitiEntityType.TEMPERATURE
    hodnota: float = Field(..., description="Temperature value")
    jednotka: Literal["°C", "K", "°F"] = Field(default="°C", description="Temperature unit")
    typ: Literal["provozní", "maximální", "minimální", "projektová"] = Field(
        default="provozní",
        description="Temperature type"
    )


class PressureEntity(GraphitiEntityBase):
    """
    Pressure value (tlak).

    Example: "15 MPa", "projektový tlak 17 MPa"
    """

    entity_type: Literal[GraphitiEntityType.PRESSURE] = GraphitiEntityType.PRESSURE
    hodnota: float = Field(..., description="Pressure value")
    jednotka: Literal["Pa", "kPa", "MPa", "bar", "atm"] = Field(default="MPa", description="Pressure unit")
    typ: Literal["provozní", "maximální", "projektový", "zkušební"] = Field(
        default="provozní",
        description="Pressure type"
    )


# =============================================================================
# PROCESS TYPE MODELS
# =============================================================================


class RadiationActivityEntity(GraphitiEntityBase):
    """
    Radiation activity (radiační činnost).

    Example: "měření", "monitoring", "kontrola ozáření", "osobní dozimetrie"
    """

    entity_type: Literal[GraphitiEntityType.RADIATION_ACTIVITY] = GraphitiEntityType.RADIATION_ACTIVITY
    typ_cinnosti: Optional[str] = Field(default=None, description="Activity type")
    frekvence: Optional[str] = Field(default=None, description="Frequency")
    zodpovedna_osoba: Optional[str] = Field(default=None, description="Responsible person/role")
    kategorie: Literal["monitoring", "měření", "kontrola", "ochrana", "jiné"] = Field(
        default="monitoring",
        description="Activity category"
    )


class MaintenanceActivityEntity(GraphitiEntityBase):
    """
    Maintenance activity (údržbová činnost).

    Example: "revize", "oprava", "preventivní údržba", "periodická kontrola"
    """

    entity_type: Literal[GraphitiEntityType.MAINTENANCE_ACTIVITY] = GraphitiEntityType.MAINTENANCE_ACTIVITY
    typ: Literal["preventivní", "korektivní", "prediktivní", "revize", "zkouška"] = Field(
        ...,
        description="Maintenance type"
    )
    perioda: Optional[str] = Field(default=None, description="Maintenance period")
    system: Optional[str] = Field(default=None, description="Target system")


class EmergencyProcedureEntity(GraphitiEntityBase):
    """
    Emergency procedure (havarijní postup).

    Example: "EOP-01: Ztráta chladiva", "Evakuační plán"
    """

    entity_type: Literal[GraphitiEntityType.EMERGENCY_PROCEDURE] = GraphitiEntityType.EMERGENCY_PROCEDURE
    oznaceni: Optional[str] = Field(default=None, description="Procedure identifier")
    typ_udalosti: Optional[str] = Field(default=None, description="Event type")
    priorita: Literal["okamžitá", "vysoká", "střední", "nízká"] = Field(
        default="vysoká",
        description="Priority level"
    )


class TrainingRequirementEntity(GraphitiEntityBase):
    """
    Training requirement (požadavek na školení).

    Example: "radiační ochrana - vstupní školení", "periodické přezkoušení obsluhy"
    """

    entity_type: Literal[GraphitiEntityType.TRAINING_REQUIREMENT] = GraphitiEntityType.TRAINING_REQUIREMENT
    typ_skoleni: str = Field(..., description="Training type")
    frekvence: Optional[str] = Field(default=None, description="Training frequency")
    cilova_skupina: Optional[str] = Field(default=None, description="Target group")
    povinnost: Literal["povinné", "doporučené", "volitelné"] = Field(
        default="povinné",
        description="Requirement level"
    )


class DocumentationRequirementEntity(GraphitiEntityBase):
    """
    Documentation requirement (požadavek na dokumentaci).

    Example: "předložit bezpečnostní zprávu", "vést provozní deník"
    """

    entity_type: Literal[GraphitiEntityType.DOCUMENTATION_REQUIREMENT] = GraphitiEntityType.DOCUMENTATION_REQUIREMENT
    typ_dokumentu: str = Field(..., description="Document type")
    lhuta: Optional[str] = Field(default=None, description="Deadline")
    adresat: Optional[str] = Field(default=None, description="Recipient (e.g., SÚJB)")
    archivace: Optional[str] = Field(default=None, description="Retention period")


# =============================================================================
# COMPLIANCE TYPE MODELS
# =============================================================================


class ComplianceGapEntity(GraphitiEntityBase):
    """
    Identified compliance gap (identifikovaný nesoulad).

    Result of compliance checking - non-conformity with requirement.
    """

    entity_type: Literal[GraphitiEntityType.COMPLIANCE_GAP] = GraphitiEntityType.COMPLIANCE_GAP
    zavaznost: Literal["kritická", "vysoká", "střední", "nízká"] = Field(
        ...,
        description="Severity level"
    )
    pozadavek: str = Field(..., description="Violated requirement reference")
    popis: str = Field(..., description="Gap description")
    doporucena_naprava: Optional[str] = Field(default=None, description="Recommended remediation")


class RiskFactorEntity(GraphitiEntityBase):
    """
    Identified risk factor (identifikovaný rizikový faktor).

    Risk identified during safety analysis.
    """

    entity_type: Literal[GraphitiEntityType.RISK_FACTOR] = GraphitiEntityType.RISK_FACTOR
    kategorie: Literal[
        "provozní", "bezpečnostní", "radiační", "environmentální", "finanční", "jiné"
    ] = Field(..., description="Risk category")
    pravdepodobnost: Literal["velmi nízká", "nízká", "střední", "vysoká", "velmi vysoká"] = Field(
        default="střední",
        description="Probability"
    )
    dopad: Literal["zanedbatelný", "malý", "střední", "velký", "katastrofální"] = Field(
        default="střední",
        description="Impact"
    )


class MitigationMeasureEntity(GraphitiEntityBase):
    """
    Risk mitigation measure (opatření ke zmírnění rizika).

    Action to reduce identified risk.
    """

    entity_type: Literal[GraphitiEntityType.MITIGATION_MEASURE] = GraphitiEntityType.MITIGATION_MEASURE
    typ: Literal["prevence", "detekce", "korekce", "kompenzace"] = Field(
        ...,
        description="Measure type"
    )
    popis: str = Field(..., description="Measure description")
    zodpovednost: Optional[str] = Field(default=None, description="Responsible party")
    stav: Literal["plánované", "implementované", "ověřené"] = Field(
        default="plánované",
        description="Implementation status"
    )


# =============================================================================
# GENERIC ENTITY FOR FALLBACK
# =============================================================================


class GenericEntity(GraphitiEntityBase):
    """
    Generic entity for types from original models.py.

    Used for entity types that don't need specialized Pydantic models.

    Note: We intentionally avoid Dict[str, Any] fields as OpenAI's structured
    outputs require additionalProperties: false on all object schemas.
    """

    # Inherits model_config from GraphitiEntityBase

    # Accept any entity type - uses base class entity_type
    entity_type: GraphitiEntityType = Field(..., description="Entity type")
    # Optional notes field for any additional context (string, not Dict)
    notes: str = Field(default="", description="Additional notes or context")


# =============================================================================
# UNION TYPE FOR ALL ENTITIES
# =============================================================================


GraphitiEntity = Union[
    # Czech Legal Types
    VyhlaskaEntity,
    NarizeniEntity,
    SbirkaZakonuEntity,
    MetodickyPokynEntity,
    SujbRozhodnutiEntity,
    BezpecnostniDokumentaceEntity,
    LimitniStavEntity,
    MezniHodnotaEntity,
    # Technical Parameters
    NumericThresholdEntity,
    MeasurementUnitEntity,
    TimePeriodEntity,
    FrequencyEntity,
    PercentageEntity,
    TemperatureEntity,
    PressureEntity,
    # Process Types
    RadiationActivityEntity,
    MaintenanceActivityEntity,
    EmergencyProcedureEntity,
    TrainingRequirementEntity,
    DocumentationRequirementEntity,
    # Compliance Types
    ComplianceGapEntity,
    RiskFactorEntity,
    MitigationMeasureEntity,
    # Generic fallback
    GenericEntity,
]


# =============================================================================
# TYPE MAPPING FOR GRAPHITI
# =============================================================================


# Map entity type enum to Pydantic model class
ENTITY_TYPE_TO_MODEL: Dict[GraphitiEntityType, type] = {
    # Czech Legal Types
    GraphitiEntityType.VYHLASKA: VyhlaskaEntity,
    GraphitiEntityType.NARIZENI: NarizeniEntity,
    GraphitiEntityType.SBIRKA_ZAKONU: SbirkaZakonuEntity,
    GraphitiEntityType.METODICKY_POKYN: MetodickyPokynEntity,
    GraphitiEntityType.SUJB_ROZHODNUTI: SujbRozhodnutiEntity,
    GraphitiEntityType.BEZPECNOSTNI_DOKUMENTACE: BezpecnostniDokumentaceEntity,
    GraphitiEntityType.LIMITNI_STAV: LimitniStavEntity,
    GraphitiEntityType.MEZNI_HODNOTA: MezniHodnotaEntity,
    # Technical Parameters
    GraphitiEntityType.NUMERIC_THRESHOLD: NumericThresholdEntity,
    GraphitiEntityType.MEASUREMENT_UNIT: MeasurementUnitEntity,
    GraphitiEntityType.TIME_PERIOD: TimePeriodEntity,
    GraphitiEntityType.FREQUENCY: FrequencyEntity,
    GraphitiEntityType.PERCENTAGE: PercentageEntity,
    GraphitiEntityType.TEMPERATURE: TemperatureEntity,
    GraphitiEntityType.PRESSURE: PressureEntity,
    # Process Types
    GraphitiEntityType.RADIATION_ACTIVITY: RadiationActivityEntity,
    GraphitiEntityType.MAINTENANCE_ACTIVITY: MaintenanceActivityEntity,
    GraphitiEntityType.EMERGENCY_PROCEDURE: EmergencyProcedureEntity,
    GraphitiEntityType.TRAINING_REQUIREMENT: TrainingRequirementEntity,
    GraphitiEntityType.DOCUMENTATION_REQUIREMENT: DocumentationRequirementEntity,
    # Compliance Types
    GraphitiEntityType.COMPLIANCE_GAP: ComplianceGapEntity,
    GraphitiEntityType.RISK_FACTOR: RiskFactorEntity,
    GraphitiEntityType.MITIGATION_MEASURE: MitigationMeasureEntity,
}


def get_all_entity_types() -> List[GraphitiEntityType]:
    """Get list of all 55 entity types."""
    return list(GraphitiEntityType)


def get_entity_model(entity_type: GraphitiEntityType) -> type:
    """
    Get Pydantic model class for entity type.

    Returns GenericEntity for types without specialized models.
    """
    return ENTITY_TYPE_TO_MODEL.get(entity_type, GenericEntity)


def get_entity_type_categories() -> Dict[str, List[GraphitiEntityType]]:
    """
    Get entity types organized by category.

    Useful for UI display and filtering.
    """
    return {
        "core": [
            GraphitiEntityType.STANDARD,
            GraphitiEntityType.ORGANIZATION,
            GraphitiEntityType.DATE,
            GraphitiEntityType.CLAUSE,
            GraphitiEntityType.TOPIC,
            GraphitiEntityType.PERSON,
            GraphitiEntityType.LOCATION,
            GraphitiEntityType.CONTRACT,
        ],
        "regulatory": [
            GraphitiEntityType.REGULATION,
            GraphitiEntityType.DECREE,
            GraphitiEntityType.DIRECTIVE,
            GraphitiEntityType.TREATY,
            GraphitiEntityType.LEGAL_PROVISION,
            GraphitiEntityType.REQUIREMENT,
        ],
        "authorization": [
            GraphitiEntityType.PERMIT,
            GraphitiEntityType.LICENSE_CONDITION,
        ],
        "nuclear_technical": [
            GraphitiEntityType.REACTOR,
            GraphitiEntityType.FACILITY,
            GraphitiEntityType.SYSTEM,
            GraphitiEntityType.SAFETY_FUNCTION,
            GraphitiEntityType.FUEL_TYPE,
            GraphitiEntityType.ISOTOPE,
            GraphitiEntityType.RADIATION_SOURCE,
            GraphitiEntityType.WASTE_CATEGORY,
            GraphitiEntityType.DOSE_LIMIT,
        ],
        "events": [
            GraphitiEntityType.INCIDENT,
            GraphitiEntityType.EMERGENCY_CLASSIFICATION,
            GraphitiEntityType.INSPECTION,
            GraphitiEntityType.DECOMMISSIONING_PHASE,
        ],
        "liability": [
            GraphitiEntityType.LIABILITY_REGIME,
        ],
        "legal_terminology": [
            GraphitiEntityType.LEGAL_TERM,
            GraphitiEntityType.DEFINITION,
        ],
        "czech_legal": [
            GraphitiEntityType.VYHLASKA,
            GraphitiEntityType.NARIZENI,
            GraphitiEntityType.SBIRKA_ZAKONU,
            GraphitiEntityType.METODICKY_POKYN,
            GraphitiEntityType.SUJB_ROZHODNUTI,
            GraphitiEntityType.BEZPECNOSTNI_DOKUMENTACE,
            GraphitiEntityType.LIMITNI_STAV,
            GraphitiEntityType.MEZNI_HODNOTA,
        ],
        "technical_parameters": [
            GraphitiEntityType.NUMERIC_THRESHOLD,
            GraphitiEntityType.MEASUREMENT_UNIT,
            GraphitiEntityType.TIME_PERIOD,
            GraphitiEntityType.FREQUENCY,
            GraphitiEntityType.PERCENTAGE,
            GraphitiEntityType.TEMPERATURE,
            GraphitiEntityType.PRESSURE,
        ],
        "processes": [
            GraphitiEntityType.RADIATION_ACTIVITY,
            GraphitiEntityType.MAINTENANCE_ACTIVITY,
            GraphitiEntityType.EMERGENCY_PROCEDURE,
            GraphitiEntityType.TRAINING_REQUIREMENT,
            GraphitiEntityType.DOCUMENTATION_REQUIREMENT,
        ],
        "compliance": [
            GraphitiEntityType.COMPLIANCE_GAP,
            GraphitiEntityType.RISK_FACTOR,
            GraphitiEntityType.MITIGATION_MEASURE,
        ],
    }
