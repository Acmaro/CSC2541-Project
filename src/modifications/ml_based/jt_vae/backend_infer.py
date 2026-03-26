"""Backend shim executed inside the isolated JT-VAE environment."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import threading
from pathlib import Path

_cache_lock = threading.Lock()


def _update_cache_file(cache_path: Path, smi: str, variants: list) -> None:
    """Atomically add one seed result to the on-disk cache.

    Uses a lock so GPU multi-worker mode (multiple threads) is safe.
    This is called from the backend subprocess directly, so it survives
    a hard kill of the parent Jupyter kernel.
    """
    with _cache_lock:
        existing: dict = {}
        if cache_path.exists():
            try:
                existing = json.loads(cache_path.read_text())
            except Exception:
                existing = {}
        existing[smi] = variants
        _atomic_write_json(cache_path, existing)


def _atomic_write_json(path: Path, data: object) -> None:
    """Write JSON atomically: write to a sibling temp file then rename.

    This guarantees that ``path`` always contains a complete, valid JSON file
    even if the process is killed mid-write (the OS makes rename() atomic on
    all major platforms).
    """
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix='.tmp_jtvae_', suffix='.json'
    )
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            json.dump(data, f, indent=2)
        Path(tmp_name).replace(path)   # atomic rename
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='JT-VAE backend inference shim')
    parser.add_argument('--backend-root', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--vocab-path', required=True)
    # Single-molecule mode (legacy)
    parser.add_argument('--seed-smiles', default=None)
    # Batch mode: path to a JSON file containing a list of SMILES strings
    parser.add_argument('--seed-smiles-file', default=None)
    parser.add_argument('--num-variants', type=int, required=True)
    parser.add_argument('--hidden-size', type=int, default=450)
    parser.add_argument('--latent-size', type=int, default=56)
    parser.add_argument('--depth-t', type=int, default=20)
    parser.add_argument('--depth-g', type=int, default=3)
    parser.add_argument('--noise-scale', type=float, default=0.30)
    parser.add_argument('--attempts-per-variant', type=int, default=4)
    parser.add_argument('--random-seed', type=int, default=0)
    parser.add_argument('--prob-decode', action='store_true')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--cache-path', default=None,
                        help='Optional path to the persistent cache JSON file. '
                             'Each completed seed is written here immediately so '
                             'results survive a hard kill of the parent process.')
    return parser.parse_args()


def _enable_cpu_compat(torch_module) -> None:
    def _tensor_cuda(self, *args, **kwargs):
        return self

    def _module_cuda(self, *args, **kwargs):
        return self

    torch_module.Tensor.cuda = _tensor_cuda
    torch_module.nn.Module.cuda = _module_cuda


def _canonicalize(rdkit_chem, smiles: str) -> str | None:
    mol = rdkit_chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return rdkit_chem.MolToSmiles(mol)


def _resolve_device(torch_module, requested: str) -> str:
    if requested == 'auto':
        return 'cuda' if torch_module.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch_module.cuda.is_available():
        raise RuntimeError('JT-VAE requested CUDA, but torch.cuda.is_available() is False')
    return requested


def _candidate_scales(base_scale: float) -> list[float]:
    scales: list[float] = []
    for candidate in (base_scale, max(base_scale, 0.75), max(base_scale, 1.5)):
        if not any(abs(candidate - scale) < 1e-8 for scale in scales):
            scales.append(candidate)
    return scales


def _generate_variants(model, torch, Chem, seed_smiles, num_variants,
                        noise_scale, attempts_per_variant, prob_decode):
    """Generate variants for a single seed. Model must already be on the right device.

    Returns an empty list (with a stderr warning) when the seed contains junction-tree
    fragments absent from the ZINC training vocabulary (KeyError in get_index).
    """
    canon_seed = _canonicalize(Chem, seed_smiles)
    if canon_seed is None:
        return []
    variants: list[str] = []
    seen = {canon_seed}
    with torch.no_grad():
        try:
            latent_mean = model.encode_latent_mean([canon_seed])
        except KeyError as exc:
            print(f'[JT-VAE] SKIP {seed_smiles[:60]!r}: '
                  f'fragment not in ZINC vocab ({exc})', file=sys.stderr, flush=True)
            return []
        z_tree_mean, z_mol_mean = torch.chunk(latent_mean, 2, dim=1)
        for scale in _candidate_scales(noise_scale):
            remaining = num_variants - len(variants)
            if remaining <= 0:
                break
            max_attempts = max(remaining * attempts_per_variant, remaining)
            for _ in range(max_attempts):
                z_tree = z_tree_mean + torch.randn_like(z_tree_mean) * scale
                z_mol = z_mol_mean + torch.randn_like(z_mol_mean) * scale
                try:
                    decoded = model.decode(z_tree, z_mol, prob_decode)
                except Exception:
                    continue
                canon = _canonicalize(Chem, decoded) if decoded else None
                if canon is None or canon in seen:
                    continue
                seen.add(canon)
                variants.append(canon)
                if len(variants) >= num_variants:
                    break
    return variants


def main() -> None:
    args = parse_args()
    backend_root = Path(args.backend_root).resolve()
    sys.path.insert(0, str(backend_root))

    import rdkit.Chem as Chem
    from rdkit import RDLogger
    import torch

    RDLogger.DisableLog('rdApp.*')
    device = _resolve_device(torch, args.device)
    if device == 'cpu':
        _enable_cpu_compat(torch)

    from fast_jtnn import JTNNVAE, Vocab

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.random_seed)

    vocab_entries = [
        line.strip()
        for line in Path(args.vocab_path).read_text().splitlines()
        if line.strip()
    ]
    vocab = Vocab(vocab_entries)

    model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depth_t, args.depth_g)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    if device == 'cuda':
        model = model.cuda()
    model.eval()

    # Batch mode: process a list of SMILES from a JSON file
    if args.seed_smiles_file:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        seeds = json.loads(Path(args.seed_smiles_file).read_text())
        results = {}
        completed = [0]

        cache_path = Path(args.cache_path) if args.cache_path else None

        def _process_one(smi):
            try:
                variants = _generate_variants(
                    model, torch, Chem, smi,
                    args.num_variants, args.noise_scale,
                    args.attempts_per_variant, args.prob_decode,
                )
            except Exception as exc:
                print(f'[JT-VAE] ERROR {smi[:40]!r}: {exc}', file=sys.stderr, flush=True)
                variants = []
            # Write to cache BEFORE printing progress — guarantees that when the
            # progress line is visible, this seed is already persisted on disk.
            if cache_path is not None:
                _update_cache_file(cache_path, smi, variants)
            completed[0] += 1
            print(f'[JT-VAE] {completed[0]}/{len(seeds)} {smi[:40]}', file=sys.stderr, flush=True)
            return smi, variants

        # PyTorch releases the GIL during CUDA ops, so threads genuinely run in
        # parallel on GPU. On CPU the decode loop is Python-heavy and holds the
        # GIL continuously, making multiple threads counterproductive.
        n_workers = 1 if device == 'cpu' else min(args.num_workers, len(seeds))
        output_json_path = Path(args.output_json)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_process_one, smi): smi for smi in seeds}
            for future in as_completed(futures):
                smi, variants = future.result()
                results[smi] = variants
                # Atomic write: the previous checkpoint is never corrupted even
                # if the process is killed during this write.
                _atomic_write_json(
                    output_json_path, {'batch': results, 'device': device}
                )
    else:
        # Single-molecule mode (legacy)
        if not args.seed_smiles:
            raise ValueError('Either --seed-smiles or --seed-smiles-file is required')
        variants = _generate_variants(
            model, torch, Chem, args.seed_smiles,
            args.num_variants, args.noise_scale,
            args.attempts_per_variant, args.prob_decode,
        )
        _atomic_write_json(
            Path(args.output_json), {'variants': variants, 'device': device}
        )


if __name__ == '__main__':
    main()
