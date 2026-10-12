#!/usr/bin/env python3
"""T1-01 step 3: lateral-curvature check. Compare M0/M1 H vibrational modes with the DFT
collinear finite-difference (ZPE) run at the same geometry.

This is the only existing DFT data that probes LATERAL H motion at the adsorption site,
so it is the main reliability test of M1 for a mostly lateral NEB path.
Both sides are harmonic, at the same frozen geometry, with the same atoms displaced.

Usage:
    python 03_h_modes.py
    python 03_h_modes.py --relaxed ..\\01_endpoints\\<date>_relax_r0   # also modes at the M0/M1 'full' minima
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from ase.io import read

import endpoint_common as ec

CM1 = 521.4709  # sqrt(eV / (A^2 amu)) expressed in cm^-1


def parse_vasp_modes(outcar: Path, natoms: int, free: list[int]) -> list[dict]:
    """Modes from the first 'Eigenvectors and eigenvalues of the dynamical matrix' block.

    VASP lists either every atom (zeros for fixed ones) or only the displaced atoms; both handled.
    """
    lines = outcar.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if "Eigenvectors and eigenvalues of the dynamical matrix" in l)
    except StopIteration:
        return []
    head = re.compile(r"^\s*(\d+)\s+f(/i)?\s*=.*?([-\d.]+)\s+cm-1")
    modes, i = [], start
    while i < len(lines):
        line = lines[i]
        if "Eigenvectors after division" in line or "ELASTIC" in line:
            break
        m = head.match(line)
        if m:
            cm = float(m.group(3)) * (-1 if m.group(2) else 1)
            rows = []
            k = i + 2
            while k < len(lines) and len(lines[k].split()) == 6:
                rows.append([float(x) for x in lines[k].split()[3:6]])
                k += 1
            vec = np.zeros((natoms, 3))
            if len(rows) == natoms:
                vec[:] = rows
            elif len(rows) == len(free):
                vec[free] = rows
            else:
                raise SystemExit(f"Unexpected eigenvector block size {len(rows)} in {outcar}")
            modes.append({"cm1": cm, "vec": vec})
            i = k
            continue
        i += 1
    return modes


def model_modes(atoms, calc, free: list[int], delta: float) -> list[dict]:
    """Central-difference Hessian over the free atoms, mass-weighted, diagonalised."""
    atoms = atoms.copy()
    atoms.set_constraint()
    atoms.calc = calc
    dof = [(a, c) for a in free for c in range(3)]
    n = len(dof)
    H = np.zeros((n, n))
    p0 = atoms.positions.copy()
    for col, (a, c) in enumerate(dof):
        fs = []
        for sgn in (+1, -1):
            atoms.positions = p0
            atoms.positions[a, c] += sgn * delta
            fs.append(atoms.get_forces()[free].reshape(-1))
        H[:, col] = -(fs[0] - fs[1]) / (2 * delta)
    atoms.positions = p0
    H = 0.5 * (H + H.T)
    m = np.repeat(atoms.get_masses()[free], 3)
    D = H / np.sqrt(np.outer(m, m))
    w, v = np.linalg.eigh(D)
    out = []
    for k in np.argsort(-w):
        vec = np.zeros((len(atoms), 3))
        vec[free] = v[:, k].reshape(-1, 3)
        out.append({"cm1": float(np.sign(w[k]) * np.sqrt(abs(w[k])) * CM1), "vec": vec})
    return out


def describe(mode, hi) -> dict:
    h = mode["vec"][hi]
    w = np.linalg.norm(mode["vec"], axis=1) ** 2
    hfrac = float(w[hi] / w.sum()) if w.sum() else 0.0
    hn = h / (np.linalg.norm(h) or 1)
    return {"cm1": mode["cm1"], "H_weight": hfrac, "H_dir": hn, "H_z_fraction": float(hn[2] ** 2),
            "character": "vertical (z)" if hn[2] ** 2 > 0.5 else "lateral (xy)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["M0", "M1"])
    ap.add_argument("--delta", type=float, default=0.01, help="finite-difference step (A)")
    ap.add_argument("--relaxed", type=Path, default=None, help="step-2 output folder (optional)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = ec.out_dir("h_modes")

    zpe = ec.env_path("NBP_ZPE_COLL_H")
    poscar = zpe / "POSCAR" if (zpe / "POSCAR").is_file() else zpe / "CONTCAR"
    geo, flags = ec.parse_poscar_flags(poscar)
    hi = ec.h_index(geo)
    free = [i for i, f in enumerate(flags) if any(f)] if flags else list(range(len(geo)))
    r0 = ec.get_label(ec.load_zscan(), "r0")
    _, mismatch = ec.match_atoms(geo, r0)
    print(f"ZPE geometry {poscar} ({len(geo)} atoms); displaced atoms: "
          f"{[f'{geo[i].symbol}{i}' for i in free][:10]}{' ...' if len(free) > 10 else ''}; "
          f"max mismatch vs z-scan r0 {mismatch:.3f} A")
    if len(free) > 12:
        print("NOTE: many displaced atoms; the model Hessian will take a while.")

    dft = parse_vasp_modes(zpe / "OUTCAR", len(geo), free)
    dft_h = sorted((describe(m, hi) for m in dft), key=lambda d: -d["H_weight"])[:3]
    dft_h.sort(key=lambda d: -d["cm1"])
    print("\nDFT (collinear, finite differences) H-dominated modes:")
    for d in dft_h:
        print(f"   {d['cm1']:8.1f} cm-1  {d['character']:<12s} H weight {d['H_weight']:.2f}  "
              f"dir ({d['H_dir'][0]:+.2f},{d['H_dir'][1]:+.2f},{d['H_dir'][2]:+.2f})")

    report = {"provenance": ec.provenance(), "zpe_dir": str(zpe), "outcar_sha256": ec.sha256(zpe / "OUTCAR"),
              "displaced_atoms": free, "mismatch_vs_r0_A": mismatch, "delta_A": args.delta,
              "dft_H_modes": dft_h, "models": {}}
    geoms = {"DFT r0/ZPE geometry": geo}
    if args.relaxed:
        for model in args.models:
            p = args.relaxed / f"{model}_full.extxyz"
            if p.is_file():
                a = read(p)
                mp, _ = ec.match_atoms(geo, a)
                geoms[f"{model}-relaxed (full)"] = (model, a, [int(mp[i]) for i in free], int(mp[hi]))

    for model in args.models:
        calc = ec.make_calc(model, args.device)
        report["models"][model] = {}
        for gname, g in geoms.items():
            if isinstance(g, tuple):
                if g[0] != model:
                    continue
                atoms, fr, h = g[1], g[2], g[3]
            else:
                atoms, fr, h = g, free, hi
            modes = sorted((describe(m, h) for m in model_modes(atoms, calc, fr, args.delta)),
                           key=lambda d: -d["H_weight"])[:3]
            modes.sort(key=lambda d: -d["cm1"])
            report["models"][model][gname] = modes
            print(f"\n[{model}] at {gname}:")
            for d in modes:
                ref = min(dft_h, key=lambda r: np.linalg.norm(np.abs(r["H_dir"]) - np.abs(d["H_dir"]))) if dft_h else None
                cmp = f"  vs DFT {ref['cm1']:7.1f} ({(d['cm1'] / ref['cm1'] - 1) * 100:+5.1f}%)" if ref and ref["cm1"] else ""
                print(f"   {d['cm1']:8.1f} cm-1  {d['character']:<12s} H weight {d['H_weight']:.2f}  "
                      f"dir ({d['H_dir'][0]:+.2f},{d['H_dir'][1]:+.2f},{d['H_dir'][2]:+.2f}){cmp}")

    ec.write_json(out / "h_modes.json", report)
    print(f"\nWrote {out / 'h_modes.json'}  (harmonic MLIP estimates vs DFT finite differences)")


if __name__ == "__main__":
    main()
