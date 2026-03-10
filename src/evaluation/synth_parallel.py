"""
synth_parallel.py — Parallel AiZynthFinder synthesizability scoring.

Uses ProcessPoolExecutor with a per-worker initializer so AiZynthFinder
loads its ONNX models exactly once per worker process (not once per molecule).

Usage (from notebook):
    from src.evaluation.synth_parallel import worker_init, score_one
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(
        max_workers=N_WORKERS,
        initializer=worker_init,
        initargs=(str(AIZYNTHFINDER_CONFIG),),
    ) as executor:
        futures = {executor.submit(score_one, smi): smi for smi in variants}
        for future in as_completed(futures):
            smi, is_solved = future.result()
"""

from __future__ import annotations

# Per-process AiZynthFinder singleton — populated by worker_init()
_finder = None


def worker_init(config_path: str) -> None:
    """
    Called once per worker process at pool startup.
    Loads AiZynthFinder and keeps it as a module-level global so
    subsequent score_one() calls reuse the same instance.
    """
    global _finder
    from aizynthfinder.aizynthfinder import AiZynthFinder

    _finder = AiZynthFinder(configfile=config_path)
    _finder.expansion_policy.select_all()
    _finder.filter_policy.select_all()
    _finder.stock.select_all()
    _finder.config.iteration_limit   = 200
    _finder.config.search.max_transforms = 6
    _finder.config.random_seed       = 42


def score_one(smi: str) -> tuple[str, bool]:
    """
    Score a single SMILES string using the process-local AiZynthFinder.
    Returns (smiles, is_solved). Safe to call from multiple processes
    simultaneously because each process has its own _finder instance.
    """
    global _finder
    try:
        _finder.target_smiles = smi
        _finder.tree_search()
        _finder.build_routes()
        stats = _finder.extract_statistics()
        return smi, bool(stats["is_solved"])
    except Exception:
        return smi, False
