#!/usr/bin/env python3
"""Create public, machine-readable M1 training-history artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


EPOCH_RE = re.compile(r"Epoch (\d+):.*loss=([0-9.Ee+-]+), RMSE_E_per_atom=\s*([0-9.Ee+-]+) meV, RMSE_F=\s*([0-9.Ee+-]+) meV / A")
INITIAL_RE = re.compile(r"Initial:.*loss=([0-9.Ee+-]+), RMSE_E_per_atom=\s*([0-9.Ee+-]+) meV, RMSE_F=\s*([0-9.Ee+-]+) meV / A")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/m1_5epoch"))
    args = parser.parse_args()

    text = args.log.read_text(encoding="utf-8", errors="replace")
    initial_match = INITIAL_RE.search(text)
    if initial_match is None:
        raise ValueError("Initial validation metrics not found")
    history = [{"epoch": -1, "stage": "M0 initialization", "loss": float(initial_match.group(1)), "energy_rmse_meV_per_atom": float(initial_match.group(2)), "force_rmse_meV_per_A": float(initial_match.group(3))}]
    for match in EPOCH_RE.finditer(text):
        history.append({"epoch": int(match.group(1)), "stage": "fine-tuned", "loss": float(match.group(2)), "energy_rmse_meV_per_atom": float(match.group(3)), "force_rmse_meV_per_A": float(match.group(4))})
    if len(history) < 2:
        raise ValueError("No completed epochs found")

    result = {
        "schema_version": 1,
        "experiment": "M1 five-epoch QE collinear fine-tune",
        "status": "completed",
        "wall_seconds": args.wall_seconds,
        "epochs": len(history) - 1,
        "gradient_updates": 16 * (len(history) - 1),
        "config_sha256": sha256(args.config),
        "model_filename": args.model.name,
        "model_sha256": sha256(args.model),
        "split_manifest_sha256": sha256(args.split_manifest),
        "selection": "Epoch selected by weighted validation loss; external r0-r8 absent from training and selection.",
        "history": history,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "training_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.out / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [item["epoch"] for item in history]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=180)
    axes[0].plot(x, [item["force_rmse_meV_per_A"] for item in history], "o-", color="#b55d34")
    axes[0].set_ylabel("Validation force RMSE (meV/Å)")
    axes[1].plot(x, [item["energy_rmse_meV_per_atom"] for item in history], "o-", color="#35618f")
    axes[1].set_ylabel("Aligned-energy RMSE (meV/atom)")
    for axis in axes:
        axis.set_xlabel("epoch (-1 = M0 initialization)"); axis.grid(alpha=0.2)
    fig.suptitle("M1 validation trade-off during CPU fine-tuning")
    fig.tight_layout()
    fig.savefig(args.out / "training_history.png")
    fig.savefig(args.out / "training_history.svg")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
