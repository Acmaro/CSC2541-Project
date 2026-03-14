from dataclasses import dataclass

try:
    from reinvent_scoring import ScoringFuncionParameters
except ImportError:
    ScoringFuncionParameters = object

from running_modes.configurations import ReactionFilterConfiguration


@dataclass
class ScoringConfiguration:

    input: str
    output_folder: str
    reaction_filter: ReactionFilterConfiguration
    scoring_function: ScoringFuncionParameters
