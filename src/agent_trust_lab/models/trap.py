from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VariationRule(BaseModel):
    field: str
    generator: str


class StructuralVariation(BaseModel):
    mode: str
    injection: str


class CodeVerification(BaseModel):
    type: str
    command: str
    expected_error_pattern: Optional[str] = None


class Remediation(BaseModel):
    problem: str
    cause: str
    fix: str


class EnhancedTrapDef(BaseModel):
    trap_id: str
    trap_type: str
    version: str = "1.0.0"
    severity: str
    difficulty: str
    category: str
    base_task: str
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    knowledge_source: Optional[str] = None
    context: Optional[str] = None
    trap_injection: str = ""
    expected_violation_signature: Optional[str] = None
    expected_behavior: Optional[str] = None
    judgement_hint: Optional[str] = None
    code_verification: Optional[CodeVerification] = None
    variation_rules: List[VariationRule] = Field(default_factory=list)
    structural_variation: List[StructuralVariation] = Field(default_factory=list)
    remediation: Optional[Remediation] = None
    mitre_atlas_id: Optional[str] = None
