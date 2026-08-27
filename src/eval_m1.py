#!/usr/bin/env python3
"""Compare M0 and M1 on identical blocked QE structures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from ase.io import read


def metric(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=float).reshape(-1)
    return {"mae": float(np.mean(np.abs(flat))), "rmse": float(np.sqrt(np.mean(flat**2))), "max_abs": float(np.max(np.abs(flat)))}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def predict(frames, model: Path, device: str, dtype: str):
    from mace.calculators import MACECalculator

    calculator = MACECalculator(model_paths=str(model), device=device, default_dtype=dtype)
    output, seconds = [], []
    for atoms in frames:
        probe = atoms.copy()
        probe.calc = calculator
        started = time.perf_counter()
        output.append((float(probe.get_potential_energy()), np.asarray(probe.get_forces(), dtype=float)))
        seconds.append(time.perf_counter() - started)
    return output, seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/m1"))
    parser.add_argument("--m0-model", type=Path, required=True)
    parser.add_argument("--m1-model", type=Path, required=True)
    parser.add_argument("--m1-label", default="M1 fine-tune")
    parser.add_argument("--out", type=Path, default=Path("artifacts/m1"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64")
    args = parser.parse_args()

    split_names = ("train", "valid", "test_internal", "test_overlap")
    split_frames = {name: read(args.data_dir / f"{name}.xyz", index=":", format="extxyz") for name in split_names}
    frames = [atoms for name in split_names for atoms in split_frames[name]]
    identities = [(atoms.info["m1_split"], atoms.info["source_group"], int(atoms.info["source_frame"])) for atoms in frames]

    m0, m0_seconds = predict(frames, args.m0_model, args.device, args.dtype)
    m1, m1_seconds = predict(frames, args.m1_model, args.device, args.dtype)
    predicted = {"M0": m0, "M1": m1}

    rows, groups = [], []
    for split in split_names:
        indices = [i for i, identity in enumerate(identities) if identity[0] == split]
        source_groups = sorted({identities[i][1] for i in indices})
        for source_group in source_groups:
            selected = [i for i in indices if identities[i][1] == source_group]
            selected.sort(key=lambda i: identities[i][2])
            dft_energy = np.asarray([float(frames[i].info["REF_energy"]) for i in selected])
            dft_forces = [np.asarray(frames[i].arrays["REF_forces"], dtype=float) for i in selected]
            movable = [np.asarray(frames[i].arrays["movable_mask"], dtype=bool) for i in selected]
            symbols = [np.asarray(frames[i].get_chemical_symbols()) for i in selected]
            dft_relative = dft_energy - dft_energy[-1]
            model_metrics = {}
            for model_name in ("M0", "M1"):
                model_energy = np.asarray([predicted[model_name][i][0] for i in selected])
                model_relative = model_energy - model_energy[-1]
                energy_error = model_relative - dft_relative
                movable_error = np.concatenate([(predicted[model_name][i][1] - force)[mask] for i, force, mask in zip(selected, dft_forces, movable)])
                h_chunks = [(predicted[model_name][i][1] - force)[symbol == "H"] for i, force, symbol in zip(selected, dft_forces, symbols) if np.any(symbol == "H")]
                model_metrics[model_name] = {
                    "relative_energy_error_eV": metric(energy_error),
                    "movable_force_component_error_eV_per_A": metric(movable_error),
                    "h_force_component_error_eV_per_A": metric(np.concatenate(h_chunks)) if h_chunks else None,
                }
                for local, i in enumerate(selected):
                    rows.append({
                        "split": split,
                        "source_group": source_group,
                        "source_frame": identities[i][2],
                        "model": model_name,
                        "dft_delta_to_split_final_eV": float(dft_relative[local]),
                        "model_delta_to_split_final_eV": float(model_relative[local]),
                        "relative_energy_error_eV": float(energy_error[local]),
                    })
            groups.append({
                "split": split,
                "source_group": source_group,
                "n_frames": len(selected),
                "relative_energy_reference": "last frame in this split/source-group block",
                "metrics": model_metrics,
            })

    result = {
        "schema_version": 1,
        "comparison": f"MACE-MP-0-small versus {args.m1_label}",
        "m0_model_sha256": sha256(args.m0_model),
        "m1_model_sha256": sha256(args.m1_model),
        "device": args.device,
        "dtype": args.dtype,
        "n_evaluations_per_model": len(frames),
        "inference_seconds": {"M0": float(sum(m0_seconds)), "M1": float(sum(m1_seconds))},
        "groups": groups,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.out / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, m0_force, m1_force, m0_energy, m1_energy = [], [], [], [], []
    for group in groups:
        labels.append(f"{group['split']}\n{group['source_group']}")
        m0_force.append(group["metrics"]["M0"]["movable_force_component_error_eV_per_A"]["mae"])
        m1_force.append(group["metrics"]["M1"]["movable_force_component_error_eV_per_A"]["mae"])
        m0_energy.append(group["metrics"]["M0"]["relative_energy_error_eV"]["mae"])
        m1_energy.append(group["metrics"]["M1"]["relative_energy_error_eV"]["mae"])
    x = np.arange(len(labels)); width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)
    axes[0].bar(x - width/2, m0_energy, width, label="M0"); axes[0].bar(x + width/2, m1_energy, width, label="M1")
    axes[0].set_ylabel("Relative-energy MAE (eV)")
    axes[1].bar(x - width/2, m0_force, width, label="M0"); axes[1].bar(x + width/2, m1_force, width, label="M1")
    axes[1].set_ylabel("Movable-force component MAE (eV/Å)")
    for axis in axes:
        axis.set_xticks(x, labels, fontsize=7); axis.grid(axis="y", alpha=0.2); axis.legend(frameon=False)
    fig.suptitle(f"{args.m1_label} versus M0")
    fig.tight_layout()
    fig.savefig(args.out / "m0_vs_m1.png")
    fig.savefig(args.out / "m0_vs_m1.svg")
    print(json.dumps({"groups": len(groups), "frames": len(frames), "seconds": result["inference_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
