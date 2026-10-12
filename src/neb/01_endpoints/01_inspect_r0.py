#!/usr/bin/env python3
"""T1-01 step 1: inspect the r0 geometry, the DFT fixed-atom flags, and how much lateral
H motion any existing DFT data covers. No MACE calls; reads only.

Usage (after dot-sourcing your private env.ps1):
    python 01_inspect_r0.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read

import endpoint_common as ec


def h_lateral_summary(frames, label: str) -> dict:
    """Lateral (in-plane) excursion of H in a trajectory, relative to its first frame."""
    hi = ec.h_index(frames[0])
    first = frames[0]
    p0 = first.positions[hi]
    lat = [float(np.linalg.norm(ec.min_image_xy(a.positions[hi] - p0, a.cell.array)[0, :2])) for a in frames]
    dz = [float(a.positions[hi, 2] - p0[2]) for a in frames]
    layers = ec.substrate_layers(frames[-1])
    site_first = ec.classify_site(first, p0[:2])
    site_last = ec.classify_site(frames[-1], frames[-1].positions[hi, :2], layers)
    return {
        "set": label, "n_frames": len(frames), "n_atoms": len(first),
        "cell_ab_A": [float(np.linalg.norm(first.cell[0])), float(np.linalg.norm(first.cell[1]))],
        "top_layer_species": layers[0]["species"], "atoms_per_layer": len(layers[0]["idx"]),
        "H_max_lateral_excursion_A": max(lat), "H_z_range_A": [min(dz), max(dz)],
        "H_site_first": site_first["label"], "H_site_last": site_last["label"],
        "H_height_last_A": ec.surface_height(frames[-1], frames[-1].positions[hi, 2]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=6, help="substrate layers to print")
    args = ap.parse_args()
    out = ec.out_dir("inspect")
    report: dict = {"provenance": ec.provenance(), "zscan": str(ec.zscan_path()),
                    "zscan_sha256": ec.sha256(ec.zscan_path())}

    frames = ec.load_zscan()
    r0 = ec.get_label(frames, "r0")
    hi = ec.h_index(r0)
    cell = r0.cell.array
    sym = np.array(r0.get_chemical_symbols())
    zsub = r0.positions[sym != "H", 2]
    layers = ec.substrate_layers(r0)
    print(f"r0: {len(r0)} atoms, formula {r0.get_chemical_formula()}, H index {hi}")
    print("cell (A):\n" + np.array2string(cell, precision=4))
    print(f"substrate z: {zsub.min():.3f} .. {zsub.max():.3f} A; slab thickness {np.ptp(zsub):.2f} A; "
          f"vacuum gap {cell[2, 2] - np.ptp(zsub):.2f} A (MACE cutoff is ~6 A)")
    print(f"{len(layers)} substrate layers; atoms per layer: {sorted(set(len(L['idx']) for L in layers))}")
    print("top layers (z, species, frac xy of first atom):")
    for L in layers[: args.layers]:
        f = r0.get_scaled_positions()[L["idx"][0]]
        print(f"  L{L['layer']:<2d} z={L['z']:8.3f}  dz_from_top={L['z'] - layers[0]['z']:6.3f}  "
              f"{L['species']:<3s} frac=({f[0]:.3f},{f[1]:.3f})  n={len(L['idx'])}")
    print("bottom layer:", layers[-1]["species"], f"z={layers[-1]['z']:.3f}")

    hpos = r0.positions[hi]
    site = ec.classify_site(r0, hpos[:2], layers)
    fh = r0.get_scaled_positions()[hi]
    print(f"\nH: z={hpos[2]:.3f} A, height above top atom {ec.surface_height(r0, hpos[2]):.3f} A, "
          f"frac xy=({fh[0]:.3f},{fh[1]:.3f})")
    print(f"H site: {site['label']}")
    for n in site["nearest_projected"]:
        print(f"   nearest projected L{n['layer']} {n['species']}: {n['lateral_distance_A']:.3f} A")
    print(f"   distance to nearest L1-L1 bridge: {site['distance_to_L1_bridge_A']:.3f} A")
    d = r0.get_distances(hi, [i for i in range(len(r0)) if i != hi], mic=True)
    others = [i for i in range(len(r0)) if i != hi]
    near = np.argsort(d)[:6]
    print("   nearest neighbours of H:", ", ".join(f"{sym[others[k]]}{others[k]} {d[k]:.3f}" for k in near))
    print(f"   DFT force on H at r0 (eV/A): {np.array2string(r0.arrays['REF_forces'][hi], precision=4)}")

    report["r0"] = {"n_atoms": len(r0), "cell_A": cell, "h_index": hi,
                    "slab_thickness_A": float(np.ptp(zsub)), "vacuum_gap_A": float(cell[2, 2] - np.ptp(zsub)),
                    "layers": [{k: L[k] for k in ("layer", "z", "species")} | {"n": len(L["idx"])} for L in layers],
                    "H_frac": fh, "H_height_A": ec.surface_height(r0, hpos[2]), "H_site": site,
                    "H_neighbours": [(sym[others[k]], int(others[k]), float(d[k])) for k in near]}

    # ---- fixed atoms in every DFT POSCAR we can find
    print("\nSelective-dynamics flags in DFT inputs:")
    report["fixed_sources"] = []
    for label, path in ec.fixed_source_candidates():
        entry = {"source": label, "path": str(path), "exists": path.is_file()}
        if path.is_file():
            ref, flags = ec.parse_poscar_flags(path)
            entry["n_atoms"] = len(ref)
            if flags is None:
                entry["selective"] = False
                print(f"  [{label}] {path}\n     no Selective dynamics block")
            else:
                mapping, worst = ec.match_atoms(ref, r0)
                fixed = [mapping[i] for i, f in enumerate(flags) if not any(f) and mapping[i] >= 0]
                partial = [(int(mapping[i]), f) for i, f in enumerate(flags) if any(f) and not all(f)]
                fixed_layers = sorted({L["layer"] for L in layers for i in L["idx"] if i in fixed})
                free_layers = sorted({L["layer"] for L in layers} - set(fixed_layers))
                hflag = [flags[i] for i, s in enumerate(ref.get_chemical_symbols()) if s == "H"]
                entry.update({"selective": True, "n_fixed": len(fixed), "fixed_layers": fixed_layers,
                              "free_layers": free_layers, "partial": partial, "H_flags": hflag,
                              "max_match_distance_A": worst})
                print(f"  [{label}] {path}\n     {len(ref)} atoms; fully fixed {len(fixed)}; "
                      f"free substrate layers {free_layers} (1 = top); partial {partial}; H flags {hflag}; "
                      f"max position mismatch vs r0 {worst:.3f} A")
        else:
            print(f"  [{label}] not found: {path}")
        report["fixed_sources"].append(entry)

    # ---- lateral coverage of DFT data that involves H
    print("\nLateral H motion covered by existing DFT data (relative to first frame):")
    report["coverage"] = []
    sets = [("z-scan r0-r8 (external test)", frames)]
    data = ec.env_path("NBP_DATA_DIR")
    for name in ("train", "valid", "test_internal", "test_overlap"):
        p = data / "m1" / f"{name}.xyz"
        if p.is_file():
            fr = [a for a in read(p, ":", format="extxyz") if "H" in a.get_chemical_symbols()]
            if fr:
                sets.append((f"M1 {name} (H frames)", fr))
    bench = data / "m0_benchmark.xyz"
    if bench.is_file():
        allf = read(bench, ":", format="extxyz")
        for g in sorted({a.info.get("source_group") for a in allf}):
            fr = [a for a in allf if a.info.get("source_group") == g and "H" in a.get_chemical_symbols()]
            if fr:
                sets.append((f"benchmark {g}", fr))
    for label, fr in sets:
        s = h_lateral_summary(fr, label)
        report["coverage"].append(s)
        print(f"  {label:<38s} n={s['n_frames']:<3d} atoms={s['n_atoms']:<3d} ab={s['cell_ab_A'][0]:.2f}x"
              f"{s['cell_ab_A'][1]:.2f}  top={s['top_layer_species']}  lateral max {s['H_max_lateral_excursion_A']:.3f} A  "
              f"dz {s['H_z_range_A'][0]:+.2f}..{s['H_z_range_A'][1]:+.2f}  site {s['H_site_first']} -> {s['H_site_last']}")

    # ---- ZPE finite-difference run (DFT lateral curvature at the H site)
    zpe = ec.env_path("NBP_ZPE_COLL_H", required=False)
    print("\nCollinear ZPE run (finite differences):", zpe)
    zinfo = {"path": str(zpe) if zpe else None}
    if zpe and zpe.is_dir():
        files = sorted(p.name for p in zpe.iterdir())
        zinfo["files"] = files
        print("  files:", ", ".join(files[:30]))
        for name in ("POSCAR", "CONTCAR"):
            if (zpe / name).is_file():
                ref, flags = ec.parse_poscar_flags(zpe / name)
                mapping, worst = ec.match_atoms(ref, r0)
                free = [] if flags is None else [i for i, f in enumerate(flags) if any(f)]
                zinfo[name] = {"n_atoms": len(ref), "selective": flags is not None,
                               "free_atoms": [(ref[i].symbol, int(i)) for i in free][:20],
                               "n_free": len(free), "max_match_distance_vs_r0_A": worst}
                print(f"  {name}: {len(ref)} atoms; free atoms {len(free)} {zinfo[name]['free_atoms'][:6]}; "
                      f"max mismatch vs r0 {worst:.3f} A")
                break
        outcar = zpe / "OUTCAR"
        if outcar.is_file():
            text = outcar.read_text(encoding="utf-8", errors="replace")
            for key in ("IBRION", "NFREE", "POTIM", "ISPIN", "LSORBIT", "ENCUT"):
                for line in text.splitlines():
                    if key in line and "=" in line:
                        zinfo[key] = line.strip()[:100]
                        break
            zinfo["n_freq_lines"] = sum(1 for line in text.splitlines() if " f  =" in line or " f/i=" in line)
            print("  OUTCAR:", {k: zinfo.get(k) for k in ("IBRION", "NFREE", "POTIM", "ISPIN", "LSORBIT")},
                  f"frequency lines: {zinfo['n_freq_lines']}")
    report["zpe"] = zinfo

    ec.write_json(out / "inspect_r0.json", report)
    print(f"\nWrote {out / 'inspect_r0.json'}")


if __name__ == "__main__":
    main()
