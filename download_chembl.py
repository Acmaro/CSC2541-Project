"""Download and sample ChEMBL molecules."""

from __future__ import annotations

import gzip
import io
import pathlib
import random
import re
import urllib.request
from urllib.error import URLError

import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

CHEMBL_URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_35_chemreps.txt.gz"
CHEMBL_URL_FALLBACK = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_24_1/chembl_24_1_chemreps.txt.gz"
FORBIDDEN_ATOMS = re.compile(r"\[(Ag|Al|As|Au|B|Ba|Bi|Ca|Cd|Ce|Co|Cr|Cu|Dy|Er|Eu|Fe|Ga|Gd|Ge|Hg|Ho|In|Ir|K|La|Li|Lu|Mg|Mn|Mo|Na|Nb|Nd|Ni|Os|Pb|Pd|Pm|Pr|Pt|Re|Rh|Ru|Sb|Sc|Se|Sm|Sn|Sr|Tb|Tc|Te|Ti|Tl|Tm|V|W|Y|Yb|Zn|Zr)\]")
SMILES_MIN_LEN = 5
SMILES_MAX_LEN = 200


def _fetch_raw(url: str = CHEMBL_URL, cache: pathlib.Path | None = None) -> bytes:
    """Download the ChEMBL gz file, using cache if available."""
    if cache and cache.exists():
        print(f"Loading from cache: {cache}")
        return cache.read_bytes()
    try:
        raw = _download(url)
    except RuntimeError:
        print(f"Primary URL failed, trying fallback: {CHEMBL_URL_FALLBACK}")
        raw = _download(CHEMBL_URL_FALLBACK)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(raw)
        print(f"Cached to: {cache}")
    return raw


def _download(url: str) -> bytes:
    print(f"Downloading: {url}")
    try:
        response = urllib.request.urlopen(url, timeout=60)
    except URLError as e:
        raise RuntimeError(f"Failed to download {url}: {e}") from e
    total = int(response.headers.get("Content-Length", 0))
    chunks: list[bytes] = []
    with tqdm(total=total or None, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
        while chunk := response.read(1024 * 1024):
            chunks.append(chunk)
            pbar.update(len(chunk))
    return b"".join(chunks)


def _parse_gz(data: bytes) -> list[str]:
    smiles_list: list[str] = []
    with gzip.open(io.BytesIO(data), "rt", encoding="utf-8") as f:
        cols = f.readline().strip().split("\t")
        smi_idx = cols.index("canonical_smiles") if "canonical_smiles" in cols else 1
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > smi_idx and parts[smi_idx].strip():
                smiles_list.append(parts[smi_idx].strip())
    return smiles_list


def _is_valid(smi: str) -> bool:
    if not (SMILES_MIN_LEN <= len(smi) <= SMILES_MAX_LEN):
        return False
    if FORBIDDEN_ATOMS.search(smi):
        return False
    return Chem.MolFromSmiles(smi) is not None


def _canonicalize(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None


def download(
    output: pathlib.Path,
    url: str = CHEMBL_URL,
    cache: pathlib.Path | None = None,
) -> pathlib.Path:
    """Download ChEMBL, filter all valid molecules, and save the full set as CSV."""
    output.parent.mkdir(parents=True, exist_ok=True)

    raw = _fetch_raw(url=url, cache=cache)
    smiles_list = _parse_gz(raw)
    print(f"Parsed {len(smiles_list):,} SMILES from ChEMBL.")

    valid, seen = [], set()
    for smi in tqdm(smiles_list, desc="Filtering"):
        if not _is_valid(smi):
            continue
        canon = _canonicalize(smi)
        if canon and canon not in seen:
            seen.add(canon)
            valid.append(canon)

    print(f"{len(valid):,} molecules passed filtering.")
    pd.DataFrame({"SMILES": valid}).to_csv(output, index=False)
    print(f"Saved {len(valid):,} molecules to: {output}")
    return output


def sample(
    input: pathlib.Path,
    output: pathlib.Path,
    num_molecules: int = 1000,
    seed: int = 42,
) -> pathlib.Path:
    """Randomly sample N molecules from a filtered ChEMBL CSV."""
    df = pd.read_csv(input)
    valid = df["SMILES"].tolist()
    print(f"Loaded {len(valid):,} molecules from {input}")

    if len(valid) < num_molecules:
        raise RuntimeError(f"Only {len(valid)} molecules available, but {num_molecules} requested.")

    sampled = random.Random(seed).sample(valid, num_molecules)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"SMILES": sampled}).to_csv(output, index=False)
    print(f"Sampled {num_molecules} molecules → {output}")
    return output
