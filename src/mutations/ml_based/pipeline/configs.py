"""
configs.py — Build Lib-INVENT JSON configuration dicts.

All paths should be absolute so the configs work regardless of the CWD
when input.py is called.
"""

import json
import os


# ── RL config ─────────────────────────────────────────────────────────────────

def make_rl_config(
    *,
    scaffolds: list[str],
    target_smiles: str,
    actor_path: str,
    critic_path: str,
    output_model_path: str,
    logging_path: str,
    job_name: str = "fingerprint_mutation",
    n_steps: int = 200,
    batch_size: int = 64,
    learning_rate: float = 0.0001,
    sigma: int = 120,
    tanimoto_weight: float = 1.0,
    qed_weight: float = 0.0,
) -> dict:
    """
    Returns a Lib-INVENT reinforcement_learning config dict.

    Scoring: Tanimoto similarity (Morgan/ECFP4) to `target_smiles`.
             Optionally adds QED as a secondary objective (qed_weight > 0).
    """
    scoring_components = [
        {
            "component_type": "tanimoto_similarity",
            "model_path": None,
            "name": "Tanimoto to target",
            "smiles": [target_smiles],
            "specific_parameters": {
                "transformation": False,
            },
            "weight": tanimoto_weight,
        }
    ]

    if qed_weight > 0:
        scoring_components.append(
            {
                "component_type": "qed_score",
                "model_path": None,
                "name": "QED",
                "smiles": [],
                "specific_parameters": {},
                "weight": qed_weight,
            }
        )

    return {
        "run_type": "reinforcement_learning",
        "logging": {
            "sender": "",
            "recipient": "local",
            "logging_path": logging_path,
            "job_name": job_name,
            "job_id": "1",
        },
        "parameters": {
            "actor": actor_path,
            "critic": critic_path,
            "scaffolds": scaffolds,
            "n_steps": n_steps,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "randomize_scaffolds": False,
            "learning_strategy": {
                "name": "dap",
                "parameters": {"sigma": sigma},
            },
            "scoring_strategy": {
                "name": "standard",
                "reaction_filter": {
                    "type": "selective",
                    "reactions": {},
                },
                "diversity_filter": {
                    "name": "NoFilterWithPenalty",
                    "bucket_size": 25,
                    "minscore": 0.2,
                    "minsimilarity": 0.4,
                },
                "scoring_function": {
                    "name": "custom_sum",
                    "parallel": False,
                    "parameters": scoring_components,
                },
            },
            "output_model_path": output_model_path,
        },
    }


# ── scaffold_decorating config ────────────────────────────────────────────────

def make_decorate_config(
    *,
    model_path: str,
    scaffolds_smi_path: str,
    output_csv_path: str,
    logging_path: str,
    batch_size: int = 64,
    n_decorations: int = 128,
    randomize: bool = True,
) -> dict:
    """
    Returns a Lib-INVENT scaffold_decorating config dict.

    Reads scaffold SMILES from `scaffolds_smi_path` (one per line),
    generates `n_decorations` decorations per scaffold.
    """
    return {
        "run_type": "scaffold_decorating",
        "parameters": {
            "model_path": model_path,
            "input_scaffold_path": scaffolds_smi_path,
            "output_path": output_csv_path,
            "logging_path": logging_path,
            "batch_size": batch_size,
            "number_of_decorations_per_scaffold": n_decorations,
            "randomize": randomize,
            "sample_uniquely": True,
        },
    }


# ── I/O helpers ───────────────────────────────────────────────────────────────

def write_json(config: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def write_scaffolds_smi(scaffolds: list[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(scaffolds) + "\n")
