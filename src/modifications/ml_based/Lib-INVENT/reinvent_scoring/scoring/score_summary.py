from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class ComponentSummary:
    total_score: float = 0.0
    parameters: Any = None


@dataclass
class FinalSummary:
    total_score: float = 0.0
    scored_smiles: List[str] = field(default_factory=list)
    scaffold_log: List[Any] = field(default_factory=list)
    compounds_stats: Any = None
