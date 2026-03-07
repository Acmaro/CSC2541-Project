"""Batch-sample synthesizable analogs via the PrexSyn /sample API."""

from __future__ import annotations

import json
import pathlib
import urllib.error
import urllib.request
from typing import Any

import numpy as np
from tqdm import tqdm

DEFAULT_URL = "http://100.65.172.100:8011/sample"
DEFAULT_NUM_SAMPLES = 64


def run_batch(
    npz: pathlib.Path,
    output: pathlib.Path,
    url: str = DEFAULT_URL,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    limit: int | None = None,
) -> pathlib.Path:
    """
    Send each molecule's pre-computed features to the /sample endpoint and
    collect generated SMILES.

    Args:
        npz: Path to .npz file produced by featurize_chembl.py
        output: Path to write the JSON results file
        url: PrexSyn API /sample endpoint
        num_samples: Number of samples to request per molecule
        limit: If set, only process the first `limit` molecules

    Returns:
        Path to the saved JSON file
    """
    data = np.load(npz, allow_pickle=True)
    smiles_arr       = data["smiles"]
    ecfp4_arr        = data["ecfp4"]
    fcfp4_arr        = data["fcfp4"]
    rdkit_vals_arr   = data["rdkit_desc_values"]
    rdkit_names      = data["rdkit_desc_names"].tolist()
    brics_fps_arr    = data["brics_fps"]
    brics_exists_arr = data["brics_exists"]

    n = limit or len(smiles_arr)
    print(f"Processing {n} molecules → {url}")

    results: list[dict[str, Any]] = []
    errors = 0

    for i in tqdm(range(n)):
        payload = {
            "ecfp4":             ecfp4_arr[i].tolist(),
            "fcfp4":             fcfp4_arr[i].tolist(),
            "rdkit_desc_values": rdkit_vals_arr[i].tolist(),
            "rdkit_desc_names":  rdkit_names,
            "brics_fps":         brics_fps_arr[i].tolist(),
            "brics_exists":      brics_exists_arr[i].tolist(),
            "source_smiles":     str(smiles_arr[i]),
            "num_samples":       num_samples,
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
            results.append(resp)
        except urllib.error.HTTPError as e:
            tqdm.write(f"[{i}] HTTP {e.code}: {e.read().decode()[:100]}")
            errors += 1
        except Exception as e:
            tqdm.write(f"[{i}] Error: {e}")
            errors += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))

    total = sum(r.get("num_valid", 0) for r in results)
    print(f"Done: {len(results)} succeeded, {errors} failed")
    print(f"Total generated molecules: {total}")
    print(f"Saved to {output}")
    return output
