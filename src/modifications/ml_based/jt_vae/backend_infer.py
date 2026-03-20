"""Backend shim executed inside the isolated JT-VAE environment."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='JT-VAE backend inference shim')
    parser.add_argument('--backend-root', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--vocab-path', required=True)
    parser.add_argument('--seed-smiles', required=True)
    parser.add_argument('--num-variants', type=int, required=True)
    parser.add_argument('--hidden-size', type=int, default=450)
    parser.add_argument('--latent-size', type=int, default=56)
    parser.add_argument('--depth-t', type=int, default=20)
    parser.add_argument('--depth-g', type=int, default=3)
    parser.add_argument('--noise-scale', type=float, default=0.30)
    parser.add_argument('--attempts-per-variant', type=int, default=8)
    parser.add_argument('--random-seed', type=int, default=0)
    parser.add_argument('--prob-decode', action='store_true')
    parser.add_argument('--output-json', required=True)
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


def main() -> None:
    args = parse_args()
    backend_root = Path(args.backend_root).resolve()
    sys.path.insert(0, str(backend_root))

    import rdkit.Chem as Chem
    import torch
    from fast_jtnn import JTNNVAE, MolTree, Vocab
    from fast_jtnn.datautils import tensorize

    _enable_cpu_compat(torch)

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    vocab_entries = [
        line.strip()
        for line in Path(args.vocab_path).read_text().splitlines()
        if line.strip()
    ]
    vocab = Vocab(vocab_entries)

    model = JTNNVAE(
        vocab,
        args.hidden_size,
        args.latent_size,
        args.depth_t,
        args.depth_g,
    )
    state_dict = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    seed_smiles = _canonicalize(Chem, args.seed_smiles)
    if seed_smiles is None:
        raise ValueError(f'Invalid SMILES: {args.seed_smiles!r}')

    tree_batch = [MolTree(seed_smiles)]
    _, jtenc_holder, mpn_holder = tensorize(tree_batch, vocab, assm=False)
    latent_mean, _ = model.encode_latent(jtenc_holder, mpn_holder)
    z_tree_mean, z_mol_mean = torch.chunk(latent_mean, 2, dim=1)

    variants: list[str] = []
    seen = {seed_smiles}
    max_attempts = max(
        args.num_variants * args.attempts_per_variant,
        args.num_variants,
    )
    for _ in range(max_attempts):
        z_tree = z_tree_mean + torch.randn_like(z_tree_mean) * args.noise_scale
        z_mol = z_mol_mean + torch.randn_like(z_mol_mean) * args.noise_scale
        decoded = model.decode(z_tree, z_mol, args.prob_decode)
        canon = _canonicalize(Chem, decoded) if decoded else None
        if canon is None or canon in seen:
            continue
        seen.add(canon)
        variants.append(canon)
        if len(variants) >= args.num_variants:
            break

    Path(args.output_json).write_text(json.dumps({'variants': variants}, indent=2))


if __name__ == '__main__':
    main()
