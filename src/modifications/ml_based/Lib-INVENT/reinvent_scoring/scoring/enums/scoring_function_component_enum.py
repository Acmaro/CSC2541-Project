from enum import Enum


class ScoringFunctionComponentNameEnum(str, Enum):
    QED_SCORE = "qed_score"
    TANIMOTO_SIMILARITY = "tanimoto_similarity"
    MATCHING_SUBSTRUCTURE = "matching_substructure"
    CUSTOM_ALERTS = "custom_alerts"
    JACCARD_DISTANCE = "jaccard_distance"
    NUM_HBD = "num_hbd"
    NUM_RINGS = "num_rings"
    MOL_WEIGHT = "mol_weight"
    CLOGP = "clogp"
