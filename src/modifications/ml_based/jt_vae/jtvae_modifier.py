"""Project-side wrapper for an isolated JT-VAE backend."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem


class JTVAEModifier:
    """Generate JT-VAE variants by calling an isolated backend process.

    This wrapper keeps the main project environment independent from the
    JT-VAE dependency stack. The actual model execution happens in
    ``backend_infer.py`` using the interpreter configured by ``JT_VAE_PYTHON``.
    """

    def __init__(
        self,
        python_bin: str | Path | None = None,
        backend_root: str | Path | None = None,
        model_path: str | Path | None = None,
        vocab_path: str | Path | None = None,
        hidden_size: int = 450,
        latent_size: int = 56,
        depth_t: int = 20,
        depth_g: int = 3,
        noise_scale: float = 0.30,
        attempts_per_variant: int = 8,
        random_seed: int = 0,
        prob_decode: bool = False,
        device: str | None = None,
    ):
        base_dir = Path(__file__).resolve().parent
        project_root = base_dir.parents[3]
        default_backend_root = base_dir / 'vendor' / 'mol_opt' / 'main' / 'jt_vae'
        default_vocab = default_backend_root / 'data' / 'zinc' / 'vocab.txt'

        self.python_bin = Path(
            python_bin
            or os.environ.get('JT_VAE_PYTHON')
            or project_root / '.venv-jtvae' / 'bin' / 'python'
        )
        self.backend_root = Path(
            backend_root or os.environ.get('JT_VAE_HOME') or default_backend_root
        )
        self.model_path = Path(model_path or os.environ.get('JT_VAE_MODEL_PATH', ''))
        self.vocab_path = Path(
            vocab_path or os.environ.get('JT_VAE_VOCAB_PATH') or default_vocab
        )
        self.backend_script = base_dir / 'backend_infer.py'
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth_t = depth_t
        self.depth_g = depth_g
        self.noise_scale = noise_scale
        self.attempts_per_variant = attempts_per_variant
        self.random_seed = random_seed
        self.prob_decode = prob_decode
        self.device = device or os.environ.get('JT_VAE_DEVICE', 'auto')

    def modify(self, seed: str, n: int) -> list[str]:
        if n <= 0:
            return []

        canon_seed = self._canonicalize_or_raise(seed)
        self._validate_runtime()

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            output_path = Path(tmp.name)

        cmd = [
            str(self.python_bin),
            str(self.backend_script),
            '--backend-root', str(self.backend_root),
            '--model-path', str(self.model_path),
            '--vocab-path', str(self.vocab_path),
            '--seed-smiles', canon_seed,
            '--num-variants', str(n),
            '--hidden-size', str(self.hidden_size),
            '--latent-size', str(self.latent_size),
            '--depth-t', str(self.depth_t),
            '--depth-g', str(self.depth_g),
            '--noise-scale', str(self.noise_scale),
            '--attempts-per-variant', str(self.attempts_per_variant),
            '--random-seed', str(self.random_seed),
            '--device', self.device,
            '--output-json', str(output_path),
        ]
        if self.prob_decode:
            cmd.append('--prob-decode')

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        try:
            if result.returncode != 0:
                raise RuntimeError(
                    'JT-VAE backend failed '
                    f'(exit {result.returncode}):\n{result.stderr.strip()}'
                )

            payload = json.loads(output_path.read_text())
            variants = payload.get('variants', [])
            results: list[str] = []
            seen = {canon_seed}
            for smiles in variants:
                canon = self._canonicalize(smiles)
                if canon is None or canon in seen:
                    continue
                seen.add(canon)
                results.append(canon)
                if len(results) >= n:
                    break
            print(f"JT-VAE: generated {len(results)} variants from seed {seed!r}")
            return results
        finally:
            output_path.unlink(missing_ok=True)

    def _validate_runtime(self) -> None:
        if not self.backend_script.exists():
            raise FileNotFoundError(f'JT-VAE backend script not found: {self.backend_script}')
        if not self.backend_root.exists():
            raise FileNotFoundError(
                f'JT-VAE backend checkout not found: {self.backend_root}\n'
                'Expected vendored backend at: '
                'src/modifications/ml_based/jt_vae/vendor/mol_opt/main/jt_vae'
            )
        if not self.python_bin.exists():
            raise FileNotFoundError(
                f'JT-VAE Python interpreter not found: {self.python_bin}\n'
                'Create it with: bash scripts/setup_jt_vae_env.sh'
            )
        if not self.model_path.exists():
            raise FileNotFoundError(
                'JT-VAE model checkpoint not found. Set JT_VAE_MODEL_PATH '
                'or pass model_path=.\n'
                f'Current value: {self.model_path}'
            )
        if not self.vocab_path.exists():
            raise FileNotFoundError(f'JT-VAE vocabulary file not found: {self.vocab_path}')

    @staticmethod
    def _canonicalize(smiles: str) -> str | None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)

    def _canonicalize_or_raise(self, smiles: str) -> str:
        canon = self._canonicalize(smiles)
        if canon is None:
            raise ValueError(f'Invalid SMILES: {smiles!r}')
        return canon
