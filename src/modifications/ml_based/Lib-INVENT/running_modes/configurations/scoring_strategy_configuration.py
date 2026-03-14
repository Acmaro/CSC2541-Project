from dataclasses import dataclass

try:
    from reinvent_scoring import ScoringFuncionParameters
except ImportError:
    # reinvent_scoring is only needed for reinforcement_learning mode.
    # scaffold_decorating does not use ScoringStrategyConfiguration.
    ScoringFuncionParameters = object

from diversity_filters.diversity_filter_parameters import DiversityFilterParameters
from running_modes.configurations import ReactionFilterConfiguration


@dataclass
class ScoringStrategyConfiguration:
    reaction_filter: ReactionFilterConfiguration
    diversity_filter: DiversityFilterParameters
    scoring_function: ScoringFuncionParameters
    name: str
