#!/usr/bin/env python3
"""Evaluate MACE-MP-0 on method-separated DFT relaxation trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.io import read


def metric(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=float).reshape(-1)
    return {
        "mae": float(np.mean(np.abs(flat))),
        "rmse": float(np.sqrt(np.mean(flat**2))),
        "max_abs": float(np.max(np.abs(flat))),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/m0_benchmark.xyz"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/m0_relaxations"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    args = parser.parse_args()

    from mace.calculators import mace_mp

    frames = read(args.data, index=":", format="extxyz")
    started = time.perf_counter()
    calculator = mace_mp(model=str(args.checkpoint), device=args.device, default_dtype=args.dtype)
    load_seconds = time.perf_counter() - started

    records = []
    for index, atoms in enumerate(frames):
        probe = atoms.copy()
        probe.calc = calculator
        tick = time.perf_counter()
        predicted_energy = float(probe.get_potential_energy())
        predicted_forces = np.asarray(probe.get_forces(), dtype=float)
        elapsed = time.perf_counter() - tick
        reference_forces = np.asarray(atoms.arrays["REF_forces"], dtype=float)
        movable = np.asarray(atoms.arrays["movable_mask"], dtype=bool)
        symbols = np.asarray(atoms.get_chemical_symbols())
        h_mask = symbols == "H"
        records.append({
            "dataset_index": index,
            "source_group": atoms.info["source_group"],
            "source_frame": int(atoms.info["source_frame"]),
            "benchmark_role": atoms.info["benchmark_role"],
            "code": atoms.info["code"],
            "physics": atoms.info["physics"],
            "n_atoms": len(atoms),
            "dft_energy_eV": float(atoms.info["REF_energy"]),
            "mace_energy_eV": predicted_energy,
            "dft_forces": reference_forces,
            "mace_forces": predicted_forces,
            "movable": movable,
            "h_mask": h_mask,
            "inference_seconds": elapsed,
        })

    grouped = defaultdict(list)
    for record in records:
        grouped[record["source_group"]].append(record)

    group_metrics = []
    csv_rows = []
    for group_id, group in grouped.items():
        group.sort(key=lambda item: item["source_frame"])
        dft_energy = np.asarray([item["dft_energy_eV"] for item in group])
        mace_energy = np.asarray([item["mace_energy_eV"] for item in group])
        dft_delta = dft_energy - dft_energy[-1]
        mace_delta = mace_energy - mace_energy[-1]
        energy_error = mace_delta - dft_delta
        all_force_error = np.concatenate([
            item["mace_forces"] - item["dft_forces"] for item in group
        ])
        movable_force_error = np.concatenate([
            (item["mace_forces"] - item["dft_forces"])[item["movable"]]
            for item in group
        ])
        h_errors = np.concatenate([
            (item["mace_forces"] - item["dft_forces"])[item["h_mask"]]
            for item in group if np.any(item["h_mask"])
        ]) if any(np.any(item["h_mask"]) for item in group) else None
        correlation = float(np.corrcoef(dft_delta, mace_delta)[0, 1]) if len(group) > 2 and np.std(dft_delta) > 0 and np.std(mace_delta) > 0 else None
        result = {
            "source_group": group_id,
            "benchmark_role": group[0]["benchmark_role"],
            "code": group[0]["code"],
            "physics": group[0]["physics"],
            "n_frames": len(group),
            "n_atoms": group[0]["n_atoms"],
            "relative_energy_reference": "last frame of source trajectory",
            "relative_energy_error_eV": metric(energy_error),
            "relative_energy_pearson_r": correlation,
            "all_force_component_error_eV_per_A": metric(all_force_error),
            "movable_force_component_error_eV_per_A": metric(movable_force_error),
            "h_force_component_error_eV_per_A": metric(h_errors) if h_errors is not None else None,
            "inference_seconds": float(sum(item["inference_seconds"] for item in group)),
        }
        group_metrics.append(result)
        for item, dft_value, mace_value, error in zip(group, dft_delta, mace_delta, energy_error):
            csv_rows.append({
                "source_group": group_id,
                "source_frame": item["source_frame"],
                "dft_delta_to_final_eV": float(dft_value),
                "mace_delta_to_final_eV": float(mace_value),
                "relative_energy_error_eV": float(error),
                "movable_force_component_mae_eV_per_A": metric((item["mace_forces"] - item["dft_forces"])[item["movable"]])["mae"],
                "inference_seconds": item["inference_seconds"],
            })

    result = {
        "schema_version": 1,
        "model": "MACE-MP-0-small",
        "training_frames": 0,
        "evaluation_protocol": "Per-calculation metrics; energies referenced to each trajectory's final frame; forces compared directly.",
        "device": args.device,
        "dtype": args.dtype,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("mace-torch", "torch", "ase", "numpy")},
        "checkpoint_filename": args.checkpoint.name,
        "checkpoint_sha256": sha256(args.checkpoint),
        "model_load_seconds": load_seconds,
        "inference_seconds_total": float(sum(item["inference_seconds"] for item in records)),
        "n_frames": len(records),
        "groups": group_metrics,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.out / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), dpi=180)
    for axis, (group_id, group) in zip(axes.flat, grouped.items()):
        group.sort(key=lambda item: item["source_frame"])
        x = np.asarray([item["source_frame"] for item in group])
        dft = np.asarray([item["dft_energy_eV"] for item in group])
        mace = np.asarray([item["mace_energy_eV"] for item in group])
        axis.plot(x, dft - dft[-1], "o-", ms=2.6, lw=1.0, label="DFT")
        axis.plot(x, mace - mace[-1], "s--", ms=2.3, lw=1.0, label="MACE-MP-0")
        axis.axhline(0, color="0.8", lw=0.7)
        axis.set_title(group_id.replace("_", " "), fontsize=9)
        axis.set_xlabel("source ionic frame")
        axis.set_ylabel(r"$E_i-E_{final}$ (eV)")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("M0: zero-shot MACE-MP-0 on NbP relaxation trajectories")
    fig.tight_layout()
    fig.savefig(args.out / "energy_trajectories.png")
    fig.savefig(args.out / "energy_trajectories.svg")

    labels = [item["source_group"].replace("_", "\n") for item in group_metrics]
    energy_mae = [item["relative_energy_error_eV"]["mae"] for item in group_metrics]
    force_mae = [item["movable_force_component_error_eV_per_A"]["mae"] for item in group_metrics]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=180)
    axes[0].bar(labels, energy_mae, color="#35618f")
    axes[0].set_ylabel("Relative-energy MAE (eV)")
    axes[1].bar(labels, force_mae, color="#b55d34")
    axes[1].set_ylabel("Movable-force component MAE (eV/Å)")
    for axis in axes:
        axis.tick_params(axis="x", labelsize=7)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Zero-shot errors by source calculation")
    fig.tight_layout()
    fig.savefig(args.out / "error_summary.png")
    fig.savefig(args.out / "error_summary.svg")
    print(json.dumps({"n_frames": len(records), "model_load_seconds": load_seconds, "inference_seconds_total": result["inference_seconds_total"]}, indent=2))


if __name__ == "__main__":
    main()
