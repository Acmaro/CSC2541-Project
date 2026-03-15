"""
Minimal stub for reinvent_scoring.
Only scaffold_decorating is used; RL scoring components are not needed.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List

from reinvent_scoring.scoring.score_summary import ComponentSummary, FinalSummary
from reinvent_scoring.scoring.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum


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
class LoggableComponent:
    name: str = ""
    component_type: str = ""
    score: float = 0.0


class ScoringFunctionNameEnum(str, Enum):
    CUSTOM_PRODUCT = "custom_product"
    CUSTOM_SUM = "custom_sum"


class ScoringFunctionFactory:
    @staticmethod
    def get_scoring_function(parameters):
        raise NotImplementedError("Scoring not available in scaffold_decorating mode")
