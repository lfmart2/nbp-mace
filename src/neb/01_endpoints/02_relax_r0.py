#!/usr/bin/env python3
"""T1-01 step 2: relax the DFT r0 geometry with M0 and M1 (MLIP estimates).

For each model, three relaxations from the same DFT (SOC-relaxed) r0 geometry:
  slab     clean slab (H removed); free layers as in DFT  -> baseline substrate drift of the model
  h_only   H free, whole substrate frozen at DFT positions -> pure H-site preference of the model
  full     H + free substrate layers (DFT-fixed layers stay fixed) -> candidate NEB endpoint

Usage:
    python 02_relax_r0.py                      # M0 and M1, fmax 0.01 eV/A
    python 02_relax_r0.py --models M1 --fmax 0.005
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np
from ase.constraints import FixAtoms
from ase.io import write
from ase.optimize import BFGS

import endpoint_common as ec


def free_max_force(atoms) -> float:
    f = atoms.get_forces()          # ASE returns constrained forces (fixed atoms -> 0)
    return float(np.linalg.norm(f, axis=1).max())


def relax(atoms, calc, fmax, steps, traj, log):
    atoms.calc = calc
    e0 = atoms.get_potential_energy()
    f0 = free_max_force(atoms)
    t = time.perf_counter()
    opt = BFGS(atoms, maxstep=0.1, trajectory=str(traj), logfile=str(log))
    converged = bool(opt.run(fmax=fmax, steps=steps))
    return {"E_initial_eV": float(e0), "E_final_eV": float(atoms.get_potential_energy()),
            "dE_eV": float(atoms.get_potential_energy() - e0), "fmax_initial_eV_A": f0,
            "fmax_final_eV_A": free_max_force(atoms), "steps": int(opt.get_number_of_steps()),
            "converged": converged, "seconds": time.perf_counter() - t}


def displacement_report(start, end, layers, hi=None) -> dict:
    disp = ec.min_image_xy(end.positions - start.positions, start.cell.array)
    norms = np.linalg.norm(disp, axis=1)
    sub = [i for i in range(len(end)) if i != hi]
    k = sub[int(np.argmax(norms[sub]))]
    layer_of = {i: L["layer"] for L in layers for i in L["idx"]}
    per_layer = {L["layer"]: float(norms[L["idx"]].max()) for L in layers[:6]}
    out = {"max_substrate_disp_A": float(norms[k]), "max_substrate_disp_atom": f"{end[k].symbol}{k}",
           "max_substrate_disp_layer": layer_of.get(k), "max_disp_by_top_layer_A": per_layer}
    if hi is not None:
        d = disp[hi]
        out.update({"H_disp_xyz_A": d, "H_disp_A": float(norms[hi]),
                    "H_lateral_disp_A": float(np.linalg.norm(d[:2])),
                    "H_height_final_A": ec.surface_height(end, end.positions[hi, 2]),
                    "H_height_above_L1_mean_A": float(end.positions[hi, 2]
                                                      - np.mean(end.positions[layers[0]["idx"], 2])),
                    "H_site_final": ec.classify_site(end, end.positions[hi, :2])["label"]})
        dist = end.get_distances(hi, sub, mic=True)
        j = int(np.argmin(dist))
        out["H_nearest_neighbour"] = f"{end[sub[j]].symbol}{sub[j]} {dist[j]:.3f} A"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["M0", "M1"], choices=["M0", "M1"])
    ap.add_argument("--fmax", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--fixed-from", default="auto", help="POSCAR whose selective flags define fixed atoms")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = ec.out_dir("relax_r0")

    r0 = ec.get_label(ec.load_zscan(), "r0")
    r0.calc = None
    hi = ec.h_index(r0)
    layers = ec.substrate_layers(r0)
    fixed, fixinfo = ec.dft_fixed_mask(r0, args.fixed_from)
    free_layers = sorted({L["layer"] for L in layers for i in L["idx"] if not fixed[i]})
    print(f"Fixed atoms from {fixinfo['source']}: {fixinfo['n_fixed']} fixed; free substrate layers {free_layers}")
    if fixinfo["max_match_distance_A"] > 0.5:
        print(f"WARNING: POSCAR/r0 atom matching distance {fixinfo['max_match_distance_A']:.2f} A - check flags")

    dft_fh = r0.arrays.pop("REF_forces")[hi].copy()   # drop DFT labels so they are not carried into outputs
    r0.info.pop("REF_energy", None)
    summary = {"provenance": ec.provenance(), "fixed_atoms": fixinfo, "free_substrate_layers": free_layers,
               "fmax_target_eV_A": args.fmax, "H_height_DFT_r0_A": ec.surface_height(r0, r0.positions[hi, 2]),
               "H_site_DFT_r0": ec.classify_site(r0, r0.positions[hi, :2], layers)["label"],
               "dft_H_force_r0_eV_A": dft_fh, "models": {}}
    rows = []
    for model in args.models:
        calc = ec.make_calc(model, args.device)
        res = {"checkpoint_sha256": ec.sha256(ec.model_paths()[model])}
        probe = r0.copy()
        probe.calc = calc
        fh = probe.get_forces()[hi]
        res["H_force_at_DFT_r0_eV_A"] = fh
        print(f"\n[{model}] force on H at DFT r0: {np.array2string(fh, precision=3)} eV/A "
              f"(DFT {np.array2string(dft_fh, precision=3)})")

        cases = {}
        slab = r0.copy()
        del slab[hi]
        slab_fixed = np.delete(fixed, hi)
        slab.set_constraint(FixAtoms(mask=slab_fixed))
        cases["slab"] = (slab, None)
        a = r0.copy()
        a.set_constraint(FixAtoms(indices=[i for i in range(len(a)) if i != hi]))
        cases["h_only"] = (a, hi)
        a = r0.copy()
        a.set_constraint(FixAtoms(mask=fixed))
        cases["full"] = (a, hi)

        for case, (atoms, h) in cases.items():
            start = atoms.copy()
            stem = out / f"{model}_{case}"
            r = relax(atoms, calc, args.fmax, args.steps, stem.with_suffix(".traj"), stem.with_suffix(".log"))
            lay = ec.substrate_layers(start)
            r.update(displacement_report(start, atoms, lay, h))
            atoms.info.update({"model": model, "case": case, "mlip_estimate": True,
                               "energy_eV": r["E_final_eV"]})
            write(stem.with_suffix(".extxyz"), atoms, format="extxyz")
            res[case] = r
            msg = (f"  {case:<7s} conv={r['converged']!s:<5s} steps={r['steps']:<4d} dE={r['dE_eV']:+.4f} eV "
                   f"fmax {r['fmax_initial_eV_A']:.3f}->{r['fmax_final_eV_A']:.4f}  "
                   f"max sub disp {r['max_substrate_disp_A']:.3f} A (L{r['max_substrate_disp_layer']})")
            if h is not None:
                d = r["H_disp_xyz_A"]
                msg += (f"\n          H disp ({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f}) |d|={r['H_disp_A']:.3f} "
                        f"lateral {r['H_lateral_disp_A']:.3f} A; height {r['H_height_final_A']:.3f} A "
                        f"(DFT r0 {summary['H_height_DFT_r0_A']:.3f}); site {r['H_site_final']}; "
                        f"nearest {r['H_nearest_neighbour']}")
            print(msg)
            rows.append({"model": model, "case": case, **{k: v for k, v in r.items()
                                                           if not isinstance(v, (dict, np.ndarray, list))}})
        summary["models"][model] = res

    ec.write_json(out / "relax_r0_summary.json", summary)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("model", "case"), k))
    with (out / "relax_r0_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}  (all numbers are MLIP estimates)")


if __name__ == "__main__":
    main()
