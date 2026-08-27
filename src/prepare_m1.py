#!/usr/bin/env python3
"""Prepare blocked M1 fine-tuning splits and training-only energy alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from ase.io import read, write


SPLIT_RULES = {
    "train": {"qe_clean_relax": range(0, 6), "qe_h_relax_a": range(0, 26)},
    "valid": {"qe_clean_relax": range(6, 9), "qe_h_relax_a": range(26, 33)},
    "test_internal": {"qe_h_relax_a": range(33, 1000)},
    "test_overlap": {"qe_h_relax_b": range(0, 1000)},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_splits(frames) -> dict[str, list]:
    splits = {name: [] for name in SPLIT_RULES}
    for atoms in frames:
        group = atoms.info["source_group"]
        source_frame = int(atoms.info["source_frame"])
        for split, rules in SPLIT_RULES.items():
            if group in rules and source_frame in rules[group]:
                splits[split].append(atoms.copy())
    return splits


def composition_matrix(frames, atomic_numbers: list[int]) -> np.ndarray:
    return np.asarray([
        [int(np.count_nonzero(atoms.numbers == number)) for number in atomic_numbers]
        for atoms in frames
    ], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/m0_benchmark.xyz"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/m1"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/m1_split_manifest.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    args = parser.parse_args()

    from mace.calculators import mace_mp

    frames = read(args.data, index=":", format="extxyz")
    splits = select_splits(frames)
    if not all(splits.values()):
        raise ValueError({name: len(items) for name, items in splits.items()})

    calculator = mace_mp(model=str(args.checkpoint), device=args.device, default_dtype=args.dtype)
    mace_energies, inference_seconds = [], []
    for atoms in splits["train"]:
        probe = atoms.copy()
        probe.calc = calculator
        started = time.perf_counter()
        mace_energies.append(float(probe.get_potential_energy()))
        inference_seconds.append(time.perf_counter() - started)

    atomic_numbers = sorted(int(number) for number in set(np.concatenate([atoms.numbers for atoms in splits["train"]])))
    counts = composition_matrix(splits["train"], atomic_numbers)
    dft_energies = np.asarray([float(atoms.info["REF_energy"]) for atoms in splits["train"]])
    residual = dft_energies - np.asarray(mace_energies)
    elemental_shifts, _, rank, singular_values = np.linalg.lstsq(counts, residual, rcond=None)
    fit_error = counts @ elemental_shifts - residual

    args.out.mkdir(parents=True, exist_ok=True)
    split_records = {}
    for split, selected in splits.items():
        split_counts = composition_matrix(selected, atomic_numbers)
        for atoms, row in zip(selected, split_counts):
            raw = float(atoms.info["REF_energy"])
            atoms.info["REF_energy_raw"] = raw
            atoms.info["REF_energy"] = raw - float(row @ elemental_shifts)
            atoms.info["m1_split"] = split
        path = args.out / f"{split}.xyz"
        write(path, selected, format="extxyz")
        split_records[split] = {
            "path": f"data/m1/{path.name}",
            "frames": len(selected),
            "source_frames": [
                {"source_group": atoms.info["source_group"], "source_frame": int(atoms.info["source_frame"])}
                for atoms in selected
            ],
        }

    manifest = {
        "schema_version": 1,
        "name": "M1 blocked QE collinear fine-tuning split",
        "policy": "Contiguous source-frame blocks; no random frame split. r0-r8 remains external and absent from all M1 files.",
        "energy_alignment": "Least-squares elemental shifts fitted only to M1 training DFT-minus-MACE absolute energies; DFT relative energies and all forces are unchanged.",
        "checkpoint_filename": args.checkpoint.name,
        "checkpoint_sha256": sha256(args.checkpoint),
        "atomic_numbers": atomic_numbers,
        "elemental_shifts_eV": {str(number): float(value) for number, value in zip(atomic_numbers, elemental_shifts)},
        "alignment_matrix_rank": int(rank),
        "alignment_singular_values": singular_values.tolist(),
        "training_alignment_error_eV": {
            "mae": float(np.mean(np.abs(fit_error))),
            "rmse": float(np.sqrt(np.mean(fit_error**2))),
        },
        "alignment_inference_seconds": float(sum(inference_seconds)),
        "splits": split_records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"split_sizes": {name: len(items) for name, items in splits.items()}, "alignment": manifest["training_alignment_error_eV"], "seconds": manifest["alignment_inference_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
