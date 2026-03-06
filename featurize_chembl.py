"""Featurize ChEMBL molecules using the same property definitions as prexsyn."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, rdFingerprintGenerator, rdMolDescriptors
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

ECFP4_RADIUS = 2
FP_NBITS = 2048
MAX_BRICS_FRAGS = 8

_ECFP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP4_RADIUS, fpSize=FP_NBITS)
_FCFP_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=ECFP4_RADIUS, fpSize=FP_NBITS,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
)


def _ecfp4(mol: Chem.Mol) -> np.ndarray:
    return _ECFP_GEN.GetFingerprintAsNumPy(mol).astype(np.float32)


def _fcfp4(mol: Chem.Mol) -> np.ndarray:
    return _FCFP_GEN.GetFingerprintAsNumPy(mol).astype(np.float32)


def _rdkit_desc_names() -> list[str]:
    return list(rdMolDescriptors.Properties().GetPropertyNames())


def _rdkit_descriptors(mol: Chem.Mol, names: list[str]) -> np.ndarray:
    values = rdMolDescriptors.Properties(names).ComputeProperties(mol)
    return np.nan_to_num(np.array(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _brics_fragments(mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray]:
    fps = np.zeros((MAX_BRICS_FRAGS, FP_NBITS), dtype=np.float32)
    exists = np.zeros(MAX_BRICS_FRAGS, dtype=bool)
    try:
        frags = sorted(BRICS.BRICSDecompose(mol))[:MAX_BRICS_FRAGS]
    except Exception:
        return fps, exists
    for i, smi in enumerate(frags):
        frag_mol = Chem.MolFromSmiles(smi.replace("*", "C"))
        if frag_mol is not None:
            fps[i] = _ecfp4(frag_mol)
            exists[i] = True
    return fps, exists


def featurize(input: pathlib.Path, output: pathlib.Path) -> pathlib.Path:
    """Featurize molecules from a SMILES CSV and save as .npz."""
    df = pd.read_csv(input)
    smiles_list: list[str] = df["SMILES"].tolist()
    print(f"Loaded {len(smiles_list)} molecules from {input}")

    desc_names = _rdkit_desc_names()

    all_ecfp4, all_fcfp4, all_rdkit, all_brics, all_exists = [], [], [], [], []
    valid_smiles: list[str] = []
    skipped = 0

    for smi in tqdm(smiles_list, desc="Featurizing"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            skipped += 1
            continue
        all_ecfp4.append(_ecfp4(mol))
        all_fcfp4.append(_fcfp4(mol))
        all_rdkit.append(_rdkit_descriptors(mol, desc_names))
        fps, exists = _brics_fragments(mol)
        all_brics.append(fps)
        all_exists.append(exists)
        valid_smiles.append(Chem.MolToSmiles(mol))

    if skipped:
        print(f"Skipped {skipped} invalid molecules.")
    print(f"Featurized {len(valid_smiles)} molecules.")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        smiles=np.array(valid_smiles, dtype=object),
        ecfp4=np.stack(all_ecfp4),
        fcfp4=np.stack(all_fcfp4),
        rdkit_desc_values=np.stack(all_rdkit),
        rdkit_desc_names=np.array(desc_names, dtype=object),
        brics_fps=np.stack(all_brics),
        brics_exists=np.stack(all_exists),
    )
    print(f"Saved features to: {output}")
    return output
