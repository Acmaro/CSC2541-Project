from __future__ import annotations

import pathlib
import time
from functools import lru_cache
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rdkit import Chem

from prexsyn.data.struct import PropertyRepr
from prexsyn.factories import load_model
from prexsyn.properties.brics import BRICSFragments
from prexsyn.properties.fingerprint import ECFP4, FCFP4
from prexsyn.properties.rdkit_descriptors import RDKitDescriptorUpperBound, RDKitDescriptors
from prexsyn.queries import Condition
from prexsyn.samplers.query import QuerySampler
from prexsyn_engine.synthesis import Synthesis

app = FastAPI(title="PrexSyn Service", version="1.0")

DEFAULT_MODEL_PATH = pathlib.Path("data/trained_models/v1_converted.yaml")
NUM_RDKIT_DESC = 4


# ── property singletons ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_props() -> tuple:
    return (
        ECFP4(name="ecfp4"),
        FCFP4(name="fcfp4"),
        RDKitDescriptors(name="rdkit_descriptors", num_evaluated_descriptors=NUM_RDKIT_DESC),
        RDKitDescriptorUpperBound(name="rdkit_descriptor_upper_bound"),
        BRICSFragments(name="brics", max_num_fragments=8),
    )


# ── custom Condition wrapper ──────────────────────────────────────────────────

class _TensorCondition(Condition):
    def __init__(self, prop_name: str, params: dict[str, torch.Tensor], weight: float = 1.0):
        super().__init__(weight=weight)
        self._prop_name = prop_name
        self._params = params

    def get_property_repr(self) -> PropertyRepr:
        return {self._prop_name: self._params}

    def score(self, synthesis: Synthesis, product: Chem.Mol) -> float:
        raise NotImplementedError


# ── request / response models ─────────────────────────────────────────────────

class SampleRequest(BaseModel):
    ecfp4: list[float] = Field(..., description="ECFP4 fingerprint vector (length 2048)")
    fcfp4: list[float] | None = Field(default=None)
    rdkit_desc_values: list[float] | None = Field(default=None)
    rdkit_desc_names: list[str] | None = Field(default=None)
    brics_fps: list[list[float]] | None = Field(default=None)
    brics_exists: list[bool] | None = Field(default=None)
    num_samples: int = Field(default=64, ge=1, le=512)
    source_smiles: str | None = Field(default=None)


# ── model loader ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, Any, str]:
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    facade, model = load_model(DEFAULT_MODEL_PATH, train=False)
    model = model.to(device)
    return facade, model, device


# ── query builder ─────────────────────────────────────────────────────────────

def _build_query(payload: SampleRequest, device: str):
    ecfp4_prop, fcfp4_prop, rdkit_prop, rdkit_ub_prop, brics_prop = _get_props()

    fp = torch.tensor(payload.ecfp4, dtype=torch.float32).unsqueeze(0).to(device)
    query = ecfp4_prop.eq(fp)

    if payload.fcfp4 is not None:
        fp_fcfp4 = torch.tensor(payload.fcfp4, dtype=torch.float32).unsqueeze(0).to(device)
        query = query & fcfp4_prop.eq(fp_fcfp4)

    if payload.rdkit_desc_values is not None and payload.rdkit_desc_names is not None:
        avail = rdkit_prop.available_descriptors
        pairs = [
            (name, val)
            for name, val in zip(payload.rdkit_desc_names, payload.rdkit_desc_values)
            if name in avail
        ][:NUM_RDKIT_DESC]
        if pairs:
            names, vals = zip(*pairs)
            types_t = torch.tensor([avail[n] for n in names], dtype=torch.long).unsqueeze(0).to(device)
            vals_t  = torch.tensor(list(vals), dtype=torch.float32).unsqueeze(0).to(device)
            query = query & _TensorCondition(rdkit_prop.name, {"values": vals_t, "types": types_t})

            ub_type = torch.tensor([[avail[names[0]]]], dtype=torch.long).to(device)
            ub_val  = torch.tensor([[vals[0]]], dtype=torch.float32).to(device)
            query = query & _TensorCondition(rdkit_ub_prop.name, {"values": ub_val, "types": ub_type})

    if payload.brics_fps is not None and payload.brics_exists is not None:
        fps_t = torch.tensor(payload.brics_fps, dtype=torch.float32).unsqueeze(0).to(device)
        ex_t  = torch.tensor(payload.brics_exists, dtype=torch.bool).unsqueeze(0).to(device)
        if ex_t.any():
            query = query & _TensorCondition(brics_prop.name, {"fingerprints": fps_t, "fingerprint_exists": ex_t})

    return query


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/sample")
def sample(payload: SampleRequest) -> dict[str, Any]:
    """
    Generate synthesizable molecules conditioned on pre-computed molecular features.
    Uses QuerySampler with a product-of-experts AND query over all provided properties.
    """
    if not DEFAULT_MODEL_PATH.exists():
        raise HTTPException(status_code=500, detail=f"Model file not found at {DEFAULT_MODEL_PATH}")
    if len(payload.ecfp4) != 2048:
        raise HTTPException(status_code=400, detail="ecfp4 must have length 2048")

    try:
        facade, model, device = _load_model()
        query = _build_query(payload, device)

        sampler = QuerySampler(
            model=model,
            token_def=facade.get_token_def(),
            num_samples=payload.num_samples,
        )
        detokenizer = facade.get_detokenizer()

        start = time.perf_counter()
        synthesis_repr = sampler.sample(query)
        syntheses = detokenizer(**synthesis_repr)
        runtime = time.perf_counter() - start

        generated: list[str] = []
        for syn in syntheses:
            if syn.stack_size() != 1:
                continue
            for prod in syn.top().to_list():
                smi = Chem.MolToSmiles(prod, canonical=True)
                if smi not in generated:
                    generated.append(smi)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sampling failed: {exc}") from exc

    return {
        "source_smiles":    payload.source_smiles,
        "num_samples":      payload.num_samples,
        "num_valid":        len(generated),
        "runtime_seconds":  round(runtime, 3),
        "generated_smiles": generated,
    }
