"""
Minimal stub for reinvent_scoring.
Only scaffold_decorating is used; RL scoring components are not needed.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List


@dataclass
class ScoringFuncionParameters:
    name: str = ""
    parameters: List[Any] = field(default_factory=list)


@dataclass
class ComponentParameters:
    component_type: str = ""
    name: str = ""
    weight: float = 1.0
    specific_parameters: Any = None


@dataclass
class ComponentSummary:
    total_score: float = 0.0
    parameters: Any = None


@dataclass
class LoggableComponent:
    name: str = ""
    component_type: str = ""
    score: float = 0.0


@dataclass
class FinalSummary:
    total_score: float = 0.0
    scored_smiles: List[str] = field(default_factory=list)
    scaffold_log: List[Any] = field(default_factory=list)
    compounds_stats: Any = None


class ScoringFunctionNameEnum(str, Enum):
    CUSTOM_PRODUCT = "custom_product"
    CUSTOM_SUM = "custom_sum"


class ScoringFunctionComponentNameEnum(str, Enum):
    QED_SCORE = "qed_score"
    TANIMOTO_SIMILARITY = "tanimoto_similarity"


class ScoringFunctionFactory:
    @staticmethod
    def get_scoring_function(parameters):
        raise NotImplementedError("Scoring not available in scaffold_decorating mode")
