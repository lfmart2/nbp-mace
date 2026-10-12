#!/usr/bin/env python3
"""T1-01 step 5: relax the candidate NEB endpoints (and the top-site intermediate) with the
substrate free (DFT-fixed bottom layers kept fixed), for M0 and M1. MLIP estimates only.

Sites are given as fractional offsets from the DFT r0 H position (bridge B1):
  B1  ( 0.0,  0.0)  Nb-Nb bridge over L4 P   (= DFT r0; the only site with DFT data)
  T   ( 0.0, -0.5)  on top of the surface Nb (shallow intermediate in the step-4 map)
  B2  (-0.5, -0.5)  Nb-Nb bridge over L3 Nb (near-degenerate with B1 in the step-4 map)
H starts nudged 0.05 A off-site: drift back to ~0 means a real minimum (with relaxed substrate).
H positions are NOT wrapped, so B1 -> T -> B2 is a continuous path for the NEB in T1-02.

Usage:
    python 05_relax_endpoints.py               # M0 and M1, fmax 0.005 eV/A
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from ase.constraints import FixAtoms
from ase.io import write
from ase.optimize import BFGS

import endpoint_common as ec

SITES = {"B1": ((0.0, 0.0), 1.10), "T": ((0.0, -0.5), 1.75), "B2": ((-0.5, -0.5), 1.10)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["M0", "M1"], choices=["M0", "M1"])
    ap.add_argument("--sites", nargs="+", default=list(SITES), choices=list(SITES))
    ap.add_argument("--fmax", type=float, default=0.005)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--fixed-from", default="auto")
    ap.add_argument("--nudge", type=float, default=0.05,
                    help="initial lateral push (A) along (1,1)/sqrt2 so a symmetric saddle is not mistaken for a minimum")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = ec.out_dir("endpoints")

    r0 = ec.get_label(ec.load_zscan(), "r0")
    r0.calc = None
    r0.arrays.pop("REF_forces", None)
    r0.info.pop("REF_energy", None)
    hi = ec.h_index(r0)
    cell = r0.cell.array
    top = r0.positions[np.array(r0.get_chemical_symbols()) != "H", 2].max()
    fixed, fixinfo = ec.dft_fixed_mask(r0, args.fixed_from)
    print(f"Fixed atoms: {fixinfo['n_fixed']} from {fixinfo['source']}; fmax {args.fmax} eV/A")

    summary = {"provenance": ec.provenance(), "fixed_atoms": fixinfo, "fmax_eV_A": args.fmax,
               "sites": {k: {"frac_offset_from_r0_H": v[0], "start_height_A": v[1]} for k, v in SITES.items()},
               "models": {}}
    for model in args.models:
        calc = ec.make_calc(model, args.device)
        res = {"checkpoint_sha256": ec.sha256(ec.model_paths()[model])}
        for site in args.sites:
            off, h0 = SITES[site]
            a = r0.copy()
            target = r0.positions[hi, :2] + np.array(off) @ cell[:2, :2]
            push = args.nudge * np.array([1.0, 1.0]) / np.sqrt(2)
            a.positions[hi] = [target[0] + push[0], target[1] + push[1], top + h0]
            a.set_constraint(FixAtoms(mask=fixed))
            a.calc = calc
            t = time.perf_counter()
            opt = BFGS(a, maxstep=0.1, trajectory=str(out / f"{model}_{site}.traj"),
                       logfile=str(out / f"{model}_{site}.log"))
            conv = bool(opt.run(fmax=args.fmax, steps=args.steps))
            e = float(a.get_potential_energy())
            fm = float(np.linalg.norm(a.get_forces(), axis=1).max())
            # measure H relative to the surface Nb (the free block can shift laterally as a whole)
            l1 = ec.substrate_layers(r0)[0]["idx"][0]
            ref_shift = a.positions[l1, :2] - r0.positions[l1, :2]
            drift = ec.min_image_xy(np.r_[a.positions[hi, :2] - target - ref_shift, 0.0], cell)[0, :2]
            sym = np.array(a.get_chemical_symbols())
            top_now = a.positions[sym != "H", 2].max()
            dist = a.get_distances(hi, [i for i in range(len(a)) if i != hi], mic=True)
            r = {"E_eV": e, "converged": conv, "steps": int(opt.get_number_of_steps()), "fmax_final_eV_A": fm,
                 "H_lateral_offset_from_site_rel_surface_A": float(np.linalg.norm(drift)),
                 "H_height_A": float(a.positions[hi, 2] - top_now),
                 "H_site_final": ec.classify_site(a, a.positions[hi, :2])["label"],
                 "H_nearest_A": float(dist.min()), "seconds": time.perf_counter() - t}
            a.info.update({"model": model, "site": site, "mlip_estimate": True, "energy_eV": e})
            write(out / f"{model}_{site}.extxyz", a, format="extxyz")
            res[site] = r
        e_ref = res.get("B1", {}).get("E_eV")
        print(f"\n[{model}]")
        for site in args.sites:
            r = res[site]
            if e_ref is not None:
                r["E_rel_B1_eV"] = r["E_eV"] - e_ref
            print(f"  {site:<3s} conv={r['converged']!s:<5s} steps={r['steps']:<4d} "
                  f"E-E(B1)={r.get('E_rel_B1_eV', float('nan')):+.4f} eV  h={r['H_height_A']:.3f} A  "
                  f"offset from site {r['H_lateral_offset_from_site_rel_surface_A']:.3f} A  nearest {r['H_nearest_A']:.3f} A  "
                  f"-> {r['H_site_final']}")
        summary["models"][model] = res

    ec.write_json(out / "endpoints_summary.json", summary)
    print(f"\nWrote {out}  (MLIP estimates; B2 and T have no DFT reference)")


if __name__ == "__main__":
    main()
