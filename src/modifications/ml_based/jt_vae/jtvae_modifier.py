"""Project-side wrapper for an isolated JT-VAE backend."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        attempts_per_variant: int = 4,
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

    # ── Public API ────────────────────────────────────────────────────────────

    def modify_batch(
        self,
        seeds: list,
        n: int,
        num_workers: int = 4,
        cache_path: str | Path | None = None,
    ) -> dict:
        """Generate variants for all seeds.

        On CPU with num_workers > 1: spawns num_workers independent subprocesses
        in parallel (each loads the model separately), bypassing the GIL.
        On CUDA: spawns one subprocess with num_workers internal threads
        (PyTorch releases the GIL during GPU ops).

        Args:
            seeds: List of seed SMILES strings.
            n: Number of variants to generate per seed.
            num_workers: Parallel subprocesses (CPU) or threads (CUDA).
            cache_path: Optional path to a JSON cache file. Already-cached seeds
                are skipped. Results are saved after each run completes.
        """
        if n <= 0:
            return {s: [] for s in seeds}

        self._validate_runtime()

        # ── Load cache ────────────────────────────────────────────────────────
        cache: dict = {}
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                try:
                    cache = json.loads(cache_path.read_text())
                except (json.JSONDecodeError, OSError):
                    cache = {}

        uncached = [s for s in seeds if s not in cache]
        n_cached = len(seeds) - len(uncached)
        if n_cached:
            print(f'JT-VAE cache: {n_cached}/{len(seeds)} seeds already cached, '
                  f'running {len(uncached)} new seeds.')
        if not uncached:
            return {s: cache[s] for s in seeds}

        # ── Dispatch: parallel subprocesses (CPU) vs single subprocess (CUDA) ─
        if self.device == 'cpu' and num_workers > 1 and len(uncached) > 1:
            new_results = self._run_parallel_cpu(uncached, n, num_workers, cache_path)
        else:
            new_results = self._run_single_subprocess(uncached, n, num_workers, cache_path)

        return {s: new_results.get(s, cache.get(s, [])) for s in seeds}

    def modify(self, seed: str, n: int) -> list[str]:
        if n <= 0:
            return []

        canon_seed = self._canonicalize_or_raise(seed)
        self._validate_runtime()

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            output_path = Path(tmp.name)

        cmd = self._build_cmd(
            output_path=output_path,
            n=n,
            seed_smiles=canon_seed,
        )
        if self.prob_decode:
            cmd.append('--prob-decode')

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
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
            print(f'JT-VAE: generated {len(results)} variants from seed {seed!r}')
            return results
        finally:
            output_path.unlink(missing_ok=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_parallel_cpu(
        self,
        uncached: list,
        n: int,
        num_workers: int,
        cache_path: Path | None,
    ) -> dict:
        """Spawn num_workers subprocesses in parallel, each handling a seed chunk.

        Each subprocess is an independent OS process with its own Python
        interpreter, so there is no GIL contention between them.
        """
        chunk_size = max(1, (len(uncached) + num_workers - 1) // num_workers)
        chunks = [uncached[i:i + chunk_size] for i in range(0, len(uncached), chunk_size)]
        actual_workers = len(chunks)
        print(f'JT-VAE: CPU parallel — {actual_workers} subprocesses × '
              f'~{chunk_size} seeds each')

        # Each worker writes to its own temp cache to avoid concurrent write races.
        chunk_caches: list[Path | None] = []
        if cache_path is not None:
            for i in range(actual_workers):
                chunk_caches.append(cache_path.parent / f'.jtvae_worker_{i}.json')
        else:
            chunk_caches = [None] * actual_workers

        new_results: dict = {}
        with ThreadPoolExecutor(max_workers=actual_workers) as pool:
            futures = {
                pool.submit(
                    self._run_one_chunk, chunk, n, worker_id, chunk_caches[worker_id]
                ): worker_id
                for worker_id, chunk in enumerate(chunks)
            }
            try:
                for future in as_completed(futures):
                    worker_id = futures[future]
                    try:
                        chunk_results = future.result()
                        new_results.update(chunk_results)
                    except Exception as exc:
                        print(f'[JT-VAE worker-{worker_id}] failed: {exc}')
            except BaseException:
                # Cancel pending futures and wait for running ones to stop
                for f in futures:
                    f.cancel()
                raise

        # Merge into main cache (after all workers finish — no race condition here)
        if cache_path is not None and new_results:
            try:
                on_disk = json.loads(cache_path.read_text()) if cache_path.exists() else {}
            except (json.JSONDecodeError, OSError):
                on_disk = {}
            on_disk.update(new_results)
            cache_path.write_text(json.dumps(on_disk, indent=2))
            print(f'JT-VAE cache: saved {len(new_results)} new seeds -> {cache_path}')

        # Clean up per-worker temp cache files
        for cc in chunk_caches:
            if cc is not None and cc.exists():
                cc.unlink(missing_ok=True)

        return new_results

    def _run_one_chunk(
        self,
        chunk_seeds: list,
        n: int,
        worker_id: int,
        chunk_cache_path: Path | None,
    ) -> dict:
        """Run one backend subprocess for a chunk of seeds. Called from a thread."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_in:
            seeds_path = Path(tmp_in.name)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_out:
            output_path = Path(tmp_out.name)

        seeds_path.write_text(json.dumps(chunk_seeds))
        cmd = self._build_cmd(
            output_path=output_path,
            n=n,
            seed_smiles_file=seeds_path,
            num_workers=1,  # Each subprocess is single-threaded; parallelism is across processes
            random_seed_offset=worker_id,
            cache_path=chunk_cache_path,
        )
        if self.prob_decode:
            cmd.append('--prob-decode')

        stderr_lines: list[str] = []
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        try:
            for line in proc.stderr:
                print(f'[worker-{worker_id}] {line}', end='', flush=True)
                stderr_lines.append(line)
            proc.wait()
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
        finally:
            chunk_results: dict = {}
            if output_path.exists() and output_path.stat().st_size > 0:
                try:
                    payload = json.loads(output_path.read_text())
                    chunk_results = payload.get('batch', {})
                except (json.JSONDecodeError, OSError):
                    pass
            seeds_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

        if proc.returncode != 0:
            raise RuntimeError(
                f'JT-VAE worker-{worker_id} failed (exit {proc.returncode}). '
                f'{len(chunk_results)}/{len(chunk_seeds)} seeds completed.\n'
                + ''.join(stderr_lines[-20:])
            )
        return chunk_results

    def _run_single_subprocess(
        self,
        uncached: list,
        n: int,
        num_workers: int,
        cache_path: Path | None,
    ) -> dict:
        """Run one subprocess handling all seeds (used for CUDA or single-worker CPU)."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_in:
            seeds_path = Path(tmp_in.name)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_out:
            output_path = Path(tmp_out.name)

        seeds_path.write_text(json.dumps(uncached))
        cmd = self._build_cmd(
            output_path=output_path,
            n=n,
            seed_smiles_file=seeds_path,
            num_workers=num_workers,
            cache_path=cache_path,
        )
        if self.prob_decode:
            cmd.append('--prob-decode')

        stderr_lines: list[str] = []
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        try:
            for line in proc.stderr:
                print(line, end='', flush=True)
                stderr_lines.append(line)
            proc.wait()
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
        finally:
            new_results: dict = {}
            if output_path.exists() and output_path.stat().st_size > 0:
                try:
                    payload = json.loads(output_path.read_text())
                    new_results = payload.get('batch', {})
                except (json.JSONDecodeError, OSError):
                    pass

            if cache_path is not None and new_results:
                try:
                    on_disk = json.loads(cache_path.read_text()) if cache_path.exists() else {}
                except (json.JSONDecodeError, OSError):
                    on_disk = {}
                on_disk.update(new_results)
                cache_path.write_text(json.dumps(on_disk, indent=2))
                print(f'\nJT-VAE cache: saved {len(new_results)} new seeds -> {cache_path}')

            seeds_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

        if proc.returncode != 0:
            raise RuntimeError(
                f'JT-VAE batch backend failed (exit {proc.returncode}). '
                f'{len(new_results)}/{len(uncached)} seeds completed before crash.\n'
                + ''.join(stderr_lines[-30:])
            )
        return new_results

    def _build_cmd(
        self,
        output_path: Path,
        n: int,
        seed_smiles: str | None = None,
        seed_smiles_file: Path | None = None,
        num_workers: int = 1,
        random_seed_offset: int = 0,
        cache_path: Path | None = None,
    ) -> list[str]:
        cmd = [
            str(self.python_bin),
            str(self.backend_script),
            '--backend-root', str(self.backend_root),
            '--model-path', str(self.model_path),
            '--vocab-path', str(self.vocab_path),
            '--num-variants', str(n),
            '--hidden-size', str(self.hidden_size),
            '--latent-size', str(self.latent_size),
            '--depth-t', str(self.depth_t),
            '--depth-g', str(self.depth_g),
            '--noise-scale', str(self.noise_scale),
            '--attempts-per-variant', str(self.attempts_per_variant),
            '--random-seed', str(self.random_seed + random_seed_offset),
            '--device', self.device,
            '--output-json', str(output_path),
            '--num-workers', str(num_workers),
        ]
        if seed_smiles is not None:
            cmd += ['--seed-smiles', seed_smiles]
        if seed_smiles_file is not None:
            cmd += ['--seed-smiles-file', str(seed_smiles_file)]
        if cache_path is not None:
            cmd += ['--cache-path', str(cache_path)]
        return cmd

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
