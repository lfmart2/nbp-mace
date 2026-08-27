#!/usr/bin/env python3
"""External r0-r8 comparison of M0 and a fine-tuned M1 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read

from eval_m1 import metric, predict, sha256
from eval_zscan import heights, scanned_h_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/zscan.xyz"))
    parser.add_argument("--m0-model", type=Path, required=True)
    parser.add_argument("--m1-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/m1_5epoch/zscan"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64")
    args = parser.parse_args()

    frames = read(args.data, index=":", format="extxyz")
    h_index = scanned_h_index(frames)
    z = heights(frames, h_index)
    order = np.argsort(z)
    frames = [frames[index] for index in order]
    z = z[order]
    dft_energy = np.asarray([float(atoms.info["REF_energy"]) for atoms in frames])
    dft_forces = np.stack([np.asarray(atoms.arrays["REF_forces"], dtype=float) for atoms in frames])
    dft_relative = dft_energy - dft_energy.min()

    predictions, timings = {}, {}
    for name, model in (("M0", args.m0_model), ("M1", args.m1_model)):
        values, seconds = predict(frames, model, args.device, args.dtype)
        energy = np.asarray([value[0] for value in values])
        forces = np.stack([value[1] for value in values])
        relative = energy - energy.min()
        predictions[name] = {
            "relative_energy": relative,
            "h_fz": forces[:, h_index, 2],
            "metrics": {
                "relative_energy_error_eV": metric(relative - dft_relative),
                "h_fz_error_eV_per_A": metric(forces[:, h_index, 2] - dft_forces[:, h_index, 2]),
                "all_force_component_error_eV_per_A": metric(forces - dft_forces),
                "sampled_minimum_label": frames[int(np.argmin(energy))].info.get("r_label"),
            },
        }
        timings[name] = float(sum(seconds))

    result = {
        "schema_version": 1,
        "role": "External fixed-composition r0-r8 test; absent from M1 train/validation/model selection",
        "n_frames": len(frames),
        "m0_model_sha256": sha256(args.m0_model),
        "m1_model_sha256": sha256(args.m1_model),
        "dft_sampled_minimum_label": frames[int(np.argmin(dft_energy))].info.get("r_label"),
        "inference_seconds": timings,
        "models": {name: value["metrics"] for name, value in predictions.items()},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 7.2), dpi=180, sharex=True)
    axes[0].plot(z, dft_relative, "o-", label="DFT")
    for name, style in (("M0", "s--"), ("M1", "^--")):
        axes[0].plot(z, predictions[name]["relative_energy"], style, label=name)
    axes[0].set_ylabel(r"$E(z)-E_{min}$ (eV)"); axes[0].legend(frameon=False)
    axes[1].plot(z, dft_forces[:, h_index, 2], "o-", label="DFT")
    for name, style in (("M0", "s--"), ("M1", "^--")):
        axes[1].plot(z, predictions[name]["h_fz"], style, label=name)
    axes[1].axhline(0, color="0.8", lw=0.7)
    axes[1].set_xlabel("H height above topmost substrate atom (Å)")
    axes[1].set_ylabel(r"H $F_z$ (eV/Å)"); axes[1].legend(frameon=False)
    fig.suptitle("External r0-r8 test: M0 versus M1")
    fig.tight_layout()
    fig.savefig(args.out / "m0_vs_m1_zscan.png")
    fig.savefig(args.out / "m0_vs_m1_zscan.svg")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
