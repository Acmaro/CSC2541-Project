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
_TIME_LIMIT = 120  # set by worker_init; used by score_one thread timeout


def worker_init(config_path: str, max_depth: int = 6, time_limit: int = 120) -> None:
    """
    Called once per worker process at pool startup.
    Loads AiZynthFinder and keeps it as a module-level global so
    subsequent score_one() calls reuse the same instance.

    Args:
        config_path: Path to aizynthfinder config.yml.
        max_depth:   Maximum retrosynthetic depth (max_transforms). Default 6.
        time_limit:  Hard wall-clock limit per molecule in seconds. Default 120.
    """
    global _finder
    import gc
    from aizynthfinder.aizynthfinder import AiZynthFinder

    _finder = AiZynthFinder(configfile=config_path)
    _finder.expansion_policy.select_all()
    _finder.filter_policy.select_all()
    _finder.stock.select_all()
    _finder.config.search.iteration_limit = 200
    _finder.config.search.max_transforms  = max_depth
    _finder.config.search.time_limit      = time_limit
    _finder.config.random_seed            = 42
    global _TIME_LIMIT
    _TIME_LIMIT = time_limit
    gc.collect()


def score_one(smi: str) -> tuple[str, bool]:
    """
    Score a single SMILES string using the process-local AiZynthFinder.
    Returns (smiles, is_solved).

    Runs tree_search() in a daemon thread so a hard Python-level timeout
    applies even when AiZynthFinder is stuck inside ONNX inference (where
    config.search.time_limit cannot interrupt it).  If the thread does not
    finish within _TIME_LIMIT seconds we return False immediately; the daemon
    thread is abandoned and the worker process is restarted by
    maxtasksperchild, which kills any lingering threads.
    """
    import gc
    import threading
    global _finder, _TIME_LIMIT

    if _finder is None:
        import sys
        print("[score_one] ERROR: _finder is None", flush=True, file=sys.stderr)
        return smi, False

    result = [False]
    exc    = [None]

    def _run() -> None:
        try:
            _finder.target_smiles = smi
            _finder.tree_search()
            _finder.build_routes()
            stats = _finder.extract_statistics()
            result[0] = bool(stats["is_solved"])
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=_TIME_LIMIT)

    if t.is_alive():
        import sys
        print(f"[score_one] TIMEOUT {_TIME_LIMIT}s: {smi[:60]}", flush=True, file=sys.stderr)

    if exc[0] is not None:
        import sys
        print(f"[score_one] {type(exc[0]).__name__}: {exc[0]} | {smi[:50]}", flush=True, file=sys.stderr)

    gc.collect()
    return smi, result[0]


def score_one_full(smi: str) -> dict:
    """
    Score a single SMILES and return full stats dict.
    Returns: {smiles, is_solved, num_routes, top_score}
    """
    global _finder
    try:
        _finder.target_smiles = smi
        _finder.tree_search()
        _finder.build_routes()
        stats = _finder.extract_statistics()
        return {
            "smiles":     smi,
            "is_solved":  bool(stats["is_solved"]),
            "num_routes": int(stats["number_of_routes"]),
            "top_score":  float(stats["top_score"]),
        }
    except Exception:
        return {"smiles": smi, "is_solved": False, "num_routes": 0, "top_score": 0.0}
