"""Synthesizability scoring using AiZynthFinder."""

from __future__ import annotations

from aizynthfinder.aizynthfinder import AiZynthFinder

_finder: AiZynthFinder | None = None
_finder_config: str | None = None


def _get_finder(config_path: str, seed: int = 42) -> AiZynthFinder:
    """Return a cached AiZynthFinder instance, loading once on first call."""
    global _finder, _finder_config
    if _finder is None or _finder_config != config_path:
        print("Loading AiZynthFinder models (first call only)...")
        _finder = AiZynthFinder(configfile=config_path)
        _finder.expansion_policy.select_all()
        _finder.filter_policy.select_all()
        _finder.stock.select_all()
        _finder.config.iteration_limit = 200
        _finder.config.random_seed = seed
        _finder_config = config_path
        print("Ready.")
    return _finder


def score_smiles(smiles: str, config_path: str) -> dict:
    """
    Run retrosynthesis on a single SMILES and return synthesizability info.

    Args:
        smiles: SMILES string of the molecule to evaluate
        config_path: Path to aizynthfinder config YAML

    Returns:
        dict with keys: smiles, is_solved, num_routes, top_score
    """
    finder = _get_finder(config_path)
    finder.target_smiles = smiles
    finder.tree_search()
    finder.build_routes()

    stats = finder.extract_statistics()
    return {
        "smiles":     smiles,
        "is_solved":  stats["is_solved"],
        "num_routes": stats["number_of_routes"],
        "top_score":  stats["top_score"],
    }


def score_batch(smiles_list: list[str], config_path: str) -> list[dict]:
    """Score a list of SMILES strings, reusing the loaded model."""
    return [score_smiles(smi, config_path) for smi in smiles_list]


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Score synthesizability via AiZynthFinder")
    parser.add_argument("--smiles",  required=True, help="SMILES to evaluate")
    parser.add_argument("--config",  required=True, help="Path to aizynthfinder config YAML")
    args = parser.parse_args()

    result = score_smiles(args.smiles, args.config)
    print(json.dumps(result, indent=2))
