#!/usr/bin/env python3
"""T1-01 step 2b: is M1's surface relaxation physical or an artefact?

Step 2 showed M1 moving the top Nb by ~0.4 A from the SOC-DFT geometry (M0: ~0.15 A).
This compares top interlayer spacings and top-layer forces across:
  - every relaxed DFT slab we have (final frame of each relaxation in m0_benchmark.xyz, and DFT r0),
  - the M0 / M1 relaxed slabs from step 2.
and evaluates M0 / M1 forces on the top layers at the final DFT geometries (vs the DFT forces there).
If M1 is near-zero force at the QE geometries it was trained on, but not at the VASP/SOC ones,
the drift reflects its training data (QE, 2x2) rather than random error.

Usage:
    python 02b_substrate_check.py --relaxed 2026-10-09_relax_r0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read

import endpoint_common as ec


def spacings(atoms, n=5):
    layers = ec.substrate_layers(atoms, tol=0.35)
    z = [L["z"] for L in layers[: n + 1]]
    return [L["species"] for L in layers[: n + 1]], np.diff(z)[::1] * -1, layers


def top_forces(atoms, forces, layers, n=3):
    return [float(np.mean(forces[L["idx"], 2])) for L in layers[:n]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relaxed", type=Path, required=True, help="step-2 output folder")
    ap.add_argument("--models", nargs="+", default=["M0", "M1"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = ec.out_dir("substrate_check")

    dft = {}
    r0 = ec.get_label(ec.load_zscan(), "r0")
    dft["DFT r0 (SOC geom, coll. labels) 1x1"] = r0
    bench = ec.env_path("NBP_DATA_DIR") / "m0_benchmark.xyz"
    frames = read(bench, ":", format="extxyz")
    for g in ("qe_clean_relax", "qe_h_relax_a", "vasp_clean_relax_30", "vasp_soc_clean_relax", "vasp_soc_h_relax"):
        fr = [a for a in frames if a.info.get("source_group") == g]
        if fr:
            a = fr[-1]
            dft[f"DFT {g} last frame ({len(a)} at, {np.linalg.norm(a.cell[0]):.2f} A)"] = a

    calcs = {m: ec.make_calc(m, args.device) for m in args.models}
    report = {"provenance": ec.provenance(), "rows": []}
    print("Top interlayer spacings (A, L1-L2, L2-L3, ...) and mean Fz on L1..L3 (eV/A, + = up)\n")
    for name, a in dft.items():
        sp_species, sp, layers = spacings(a)
        ref = a.arrays.get("REF_forces")
        line = {"structure": name, "layers": sp_species, "spacings_A": sp}
        print(f"{name}\n   layers {'/'.join(sp_species)}  spacings {np.array2string(sp, precision=3)}")
        if ref is not None:
            line["dft_Fz_L1_L3"] = top_forces(a, ref, layers)
            print(f"   DFT Fz L1-L3 {np.array2string(np.array(line['dft_Fz_L1_L3']), precision=3)}")
        for m, calc in calcs.items():
            p = a.copy()
            p.set_constraint()
            p.calc = calc
            f = p.get_forces()
            line[f"{m}_Fz_L1_L3"] = top_forces(p, f, layers)
            print(f"   {m}  Fz L1-L3 {np.array2string(np.array(line[f'{m}_Fz_L1_L3']), precision=3)}")
        report["rows"].append(line)

    print("\nModel-relaxed slabs from step 2:")
    for m in args.models:
        for case in ("slab", "full"):
            p = args.relaxed / f"{m}_{case}.extxyz"
            if not p.is_file():
                continue
            a = read(p)
            sp_species, sp, layers = spacings(a)
            dz = a.positions[:, 2] - (r0.positions[:, 2] if len(a) == len(r0)
                                      else np.delete(r0.positions[:, 2], ec.h_index(r0)))
            dz_layers = [float(np.mean(dz[L["idx"]])) for L in layers[:4]]
            report["rows"].append({"structure": f"{m} {case}", "layers": sp_species, "spacings_A": sp,
                                   "dz_vs_DFT_r0_L1_L4": dz_layers})
            print(f"{m} {case:<5s} layers {'/'.join(sp_species)}  spacings {np.array2string(sp, precision=3)}  "
                  f"dz vs DFT r0 (L1..L4) {np.array2string(np.array(dz_layers), precision=3)}")

    ec.write_json(out / "substrate_check.json", report)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
