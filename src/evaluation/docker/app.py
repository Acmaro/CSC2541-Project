from __future__ import annotations

import pathlib
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from synthesizability import score_smiles, score_batch

app = FastAPI(title="AiZynthFinder Service", version="1.0")

DEFAULT_CONFIG = pathlib.Path("/data/config.yml")


# ── request / response models ─────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    smiles: list[str] = Field(..., description="List of SMILES strings to evaluate")
    config_path: str = Field(default=str(DEFAULT_CONFIG), description="Path to aizynthfinder config YAML")


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score")
def score(payload: ScoreRequest) -> dict[str, Any]:
    """
    Run AiZynthFinder retrosynthesis on a list of SMILES.
    Returns synthesizability results for each molecule.
    """
    config = pathlib.Path(payload.config_path)
    if not config.exists():
        raise HTTPException(status_code=500, detail=f"Config not found at {config}")

    try:
        results = score_batch(payload.smiles, str(config))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}") from exc

    solved = sum(r["is_solved"] for r in results)
    return {
        "num_molecules": len(results),
        "num_solved": solved,
        "results": results,
    }
