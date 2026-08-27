#!/usr/bin/env python3
"""Zero-shot MACE-MP-0 evaluation on the fixed-composition r0-r8 scan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def scanned_h_index(frames) -> int:
    symbols = np.array(frames[0].get_chemical_symbols())
    h_indices = np.where(symbols == "H")[0]
    positions = np.stack([atoms.positions for atoms in frames])
    return int(h_indices[np.argmax(positions[:, h_indices, 2].std(axis=0))])


def heights(frames, h_index: int) -> np.ndarray:
    values = []
    for atoms in frames:
        symbols = np.array(atoms.get_chemical_symbols())
        top = atoms.positions[symbols != "H", 2].max()
        values.append(atoms.positions[h_index, 2] - top)
    return np.asarray(values)


def metric(error: np.ndarray) -> dict:
    flat = np.asarray(error, dtype=float).reshape(-1)
    return {
        "mae": float(np.abs(flat).mean()),
        "rmse": float(np.sqrt(np.mean(flat ** 2))),
        "max_abs": float(np.abs(flat).max()),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/zscan.xyz"))
    ap.add_argument("--out", type=Path, default=Path("results/zeroshot"))
    ap.add_argument("--model", default="small", choices=["small", "medium", "large"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    args = ap.parse_args()

    from ase.io import read
    from mace.calculators import mace_mp
    from mace.calculators.foundations_models import download_mace_mp_checkpoint

    frames = read(args.data, index=":", format="extxyz")
    h_index = scanned_h_index(frames)
    z = heights(frames, h_index)
    order = np.argsort(z)
    frames = [frames[i] for i in order]
    z = z[order]
    dft_e = np.asarray([float(at.info["REF_energy"]) for at in frames])
    dft_f = np.stack([np.asarray(at.arrays["REF_forces"], dtype=float) for at in frames])

    started = time.perf_counter()
    checkpoint = Path(download_mace_mp_checkpoint(args.model))
    calc = mace_mp(model=str(checkpoint), device=args.device, default_dtype=args.dtype)
    load_seconds = time.perf_counter() - started
    mace_e, mace_f, inference_seconds = [], [], []
    for atoms in frames:
        probe = atoms.copy()
        probe.calc = calc
        t0 = time.perf_counter()
        mace_e.append(float(probe.get_potential_energy()))
        mace_f.append(np.asarray(probe.get_forces(), dtype=float))
        inference_seconds.append(time.perf_counter() - t0)
    mace_e, mace_f = np.asarray(mace_e), np.stack(mace_f)

    dft_rel, mace_rel = dft_e - dft_e.min(), mace_e - mace_e.min()
    rel_error, force_error = mace_rel - dft_rel, mace_f - dft_f
    labels = [at.info.get("r_label", f"point{i}") for i, at in enumerate(frames)]
    near_minimum = dft_rel <= 0.10
    short_range = z < 0.90
    large_height = z > 1.40
    result = {
        "schema_version": 1,
        "model": f"MACE-MP-0-{args.model}",
        "device": args.device,
        "dtype": args.dtype,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": {name: package_version(name) for name in ("mace-torch", "torch", "ase", "numpy")},
        "checkpoint_filename": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "model_load_seconds": load_seconds,
        "inference_seconds_total": float(sum(inference_seconds)),
        "inference_seconds_per_structure_mean": float(np.mean(inference_seconds)),
        "n_structures": len(frames),
        "n_atoms_per_structure": len(frames[0]),
        "scanned_h_index_zero_based": h_index,
        "dft_minimum_label": labels[int(np.argmin(dft_e))],
        "mace_minimum_label": labels[int(np.argmin(mace_e))],
        "relative_energy_error_eV": metric(rel_error),
        "relative_energy_pearson_r": float(np.corrcoef(dft_rel, mace_rel)[0, 1]),
        "relative_energy_regions_eV": {
            "near_minimum_dft_le_0.10_eV": {"n": int(near_minimum.sum()), **metric(rel_error[near_minimum])},
            "short_range_z_lt_0.90_A": {"n": int(short_range.sum()), **metric(rel_error[short_range])},
            "large_height_z_gt_1.40_A": {"n": int(large_height.sum()), **metric(rel_error[large_height])},
        },
        "all_force_component_error_eV_per_A": metric(force_error),
        "h_fz_error_eV_per_A": metric(force_error[:, h_index, 2]),
        "points": [],
    }
    for i, label in enumerate(labels):
        result["points"].append({
            "label": label,
            "h_height_A": float(z[i]),
            "dft_energy_eV": float(dft_e[i]),
            "mace_energy_eV": float(mace_e[i]),
            "dft_relative_energy_eV": float(dft_rel[i]),
            "mace_relative_energy_eV": float(mace_rel[i]),
            "relative_energy_error_eV": float(rel_error[i]),
            "dft_h_fz_eV_per_A": float(dft_f[i, h_index, 2]),
            "mace_h_fz_eV_per_A": float(mace_f[i, h_index, 2]),
            "h_fz_error_eV_per_A": float(force_error[i, h_index, 2]),
            "inference_seconds": float(inference_seconds[i]),
        })

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "mace_zeroshot.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.out / "mace_zeroshot.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result["points"][0].keys())
        writer.writeheader()
        writer.writerows(result["points"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.0), dpi=180, sharex=True)
    axes[0].plot(z, dft_rel, "o-", label="DFT (collinear SCF)")
    axes[0].plot(z, mace_rel, "s--", label=f"MACE-MP-0 {args.model} (zero-shot)")
    axes[0].set_ylabel(r"$E(z)-E_{min}$ (eV)")
    axes[0].legend(frameon=False)
    axes[1].plot(z, dft_f[:, h_index, 2], "o-", label="DFT")
    axes[1].plot(z, mace_f[:, h_index, 2], "s--", label="MACE")
    axes[1].axhline(0, color="0.75", lw=0.8)
    axes[1].set_xlabel("H height above topmost substrate atom (Å)")
    axes[1].set_ylabel(r"H $F_z$ (eV/Å)")
    axes[1].legend(frameon=False)
    fig.suptitle("Constrained H-height scan on NbP")
    fig.tight_layout()
    fig.savefig(args.out / "mace_zeroshot.png")

    keys = ("model", "relative_energy_error_eV", "all_force_component_error_eV_per_A",
            "h_fz_error_eV_per_A", "dft_minimum_label", "mace_minimum_label",
            "inference_seconds_total")
    print(json.dumps({key: result[key] for key in keys}, indent=2))


if __name__ == "__main__":
    main()
