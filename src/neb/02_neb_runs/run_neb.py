#!/usr/bin/env python
r"""
run_neb.py - T1-02: MLIP-driven CI-NEB for H on the Nb-terminated NbP slab (1x1 cell, 1 ML H).

Every energy and barrier printed here is an MLIP PATHWAY ESTIMATE (MACE M0 or M1),
not a DFT-validated number. No DFT data exist off the B1 site.

Path: B1 (Nb-Nb bridge, DFT r0 site) -> T (atop surface Nb) -> B2 (second bridge).
The initial band is two linear segments through T. It is never a straight B1->B2 line,
because that line crosses a ~0.38 eV region.
IDPP is not used: with a 3.36 A cell the minimum-image pairs can flip along the path,
and in this path almost only H moves, so the linear band through T is already a good guess.

Usage (Anaconda PowerShell Prompt, after . your private env.ps1 (RUNBOOK section 8),
inside src\neb\02_neb_runs):
  python run_neb.py --model M1 --nimages 9 --dry-run     # band + single points + cost estimate
  python run_neb.py --model M1 --nimages 9               # NEB to fmax1, then CI-NEB to fmax2
  python run_neb.py --model M1 --nimages 7 --path B1T    # split band B1 -> T (relaxed T endpoint)
  python run_neb.py --model M1 --nimages 7 --path TB2    # split band T -> B2
  python run_neb.py --model M0 --nimages 9

Model checkpoint: --model-path, else NEB_<MODEL>_MODEL, else the env.ps1 variables
MACE_M1_CHECKPOINT / MACE_MP0_SMALL_CHECKPOINT (the same files T1-01 used).
Outputs: run_<date>_<model>_<nimages>[_<path>][_k<k>][_bfgs][_<tag>]\
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
from ase import units
from ase.constraints import FixAtoms, FixCartesian
from ase.geometry import find_mic
from ase.io import read, write
from ase.optimize import BFGS, FIRE

try:
    from ase.mep import NEB, NEBTools
except ImportError:  # ASE < 3.23
    from ase.neb import NEB, NEBTools

HERE = Path(__file__).resolve().parent
DEFAULT_ENDPOINTS = HERE.parent / "01_endpoints" / "2026-10-09_endpoints"
M1_SHA_HEAD, M1_SHA_TAIL = "fa762686", "cf8d"
N_FIXED_EXPECTED = 16
T_HEIGHT_GUESS = 1.75  # H height above the top Nb at T (T1-01 lateral map), Angstrom
FRAMING = "MLIP pathway estimate (MACE), not DFT-validated; 1 ML H, 1x1 cell"


# ------------------------------------------------------------------ logging
class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


# ------------------------------------------------------------------ model
ENV_CHECKPOINT = {"M0": "MACE_MP0_SMALL_CHECKPOINT", "M1": "MACE_M1_CHECKPOINT"}  # set by env.ps1


def resolve_model_path(model, cli_path):
    p = (cli_path or os.environ.get(f"NEB_{model}_MODEL")
         or os.environ.get(ENV_CHECKPOINT[model], "").strip() or None)
    if p and not Path(p).is_file():
        sys.exit(f"Model file not found: {p}")
    return Path(p) if p else None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_calc(model, model_path):
    if model_path is not None:
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=str(model_path), device="cpu",
                              default_dtype="float64")
    if model == "M0":
        from mace.calculators import mace_mp
        return mace_mp(model="small", device="cpu", default_dtype="float64")
    sys.exit("M1 needs --model-path or the env var NEB_M1_MODEL.")


def provenance(model, model_path):
    import ase
    prov = {"date": dt.datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(), "platform": platform.platform(),
            "ase": ase.__version__, "numpy": np.__version__, "model": model,
            "model_path": str(model_path) if model_path else "mace_mp(model='small')"}
    try:
        import torch
        prov["torch"] = torch.__version__
        prov["torch_threads"] = torch.get_num_threads()
    except Exception:
        pass
    try:
        import mace
        prov["mace"] = getattr(mace, "__version__", "unknown")
    except Exception:
        pass
    if model_path is not None:
        prov["model_sha256"] = sha256(model_path)
    return prov


# ------------------------------------------------------------------ structure helpers
def h_index(atoms):
    idx = [a.index for a in atoms if a.symbol == "H"]
    if len(idx) != 1:
        sys.exit(f"Expected exactly one H atom, found {len(idx)}.")
    return idx[0]


def fixed_from_constraints(atoms):
    idx = set()
    for c in atoms.constraints:
        if isinstance(c, FixAtoms):
            idx.update(int(i) for i in c.get_indices())
        elif isinstance(c, FixCartesian):
            mask = np.asarray(c.mask, dtype=bool)
            ind = getattr(c, "index", getattr(c, "a", []))
            if mask.all():
                idx.update(int(i) for i in np.atleast_1d(ind))
            else:
                print("  WARNING: partial FixCartesian found in file; ignored")
    return sorted(idx)


def fixed_indices(atoms, label):
    from_file = fixed_from_constraints(atoms)
    order = np.argsort(atoms.positions[:, 2])
    lowest = sorted(int(i) for i in order[:N_FIXED_EXPECTED])
    if from_file:
        if len(from_file) != N_FIXED_EXPECTED:
            print(f"  WARNING {label}: {len(from_file)} fixed atoms in file, expected {N_FIXED_EXPECTED}")
        if from_file != lowest:
            print(f"  WARNING {label}: fixed atoms in file are not the {N_FIXED_EXPECTED} lowest atoms")
        return from_file, "from file"
    print(f"  NOTE {label}: no constraints stored in the file; fixing the {N_FIXED_EXPECTED} lowest atoms")
    return lowest, f"rebuilt ({N_FIXED_EXPECTED} lowest by z)"


def top_nb(atoms):
    nb = [a.index for a in atoms if a.symbol == "Nb"]
    return max(nb, key=lambda i: atoms.positions[i, 2])


def inplane_dist(atoms, p, q):
    d = np.array([[p[0] - q[0], p[1] - q[1], 0.0]])
    v, _ = find_mic(d, atoms.cell, pbc=[True, True, False])
    return float(np.linalg.norm(v[0, :2]))


def align(ref, atoms, h, h_xy_target):
    """Unwrap the substrate onto ref (minimum image), then move H by in-plane lattice
    vectors so that its xy is as close as possible to h_xy_target (keeps H unwrapped)."""
    out = atoms.copy()
    out.set_constraint()
    pos = out.get_positions()
    sub = np.array([i for i in range(len(out)) if i != h])
    d, _ = find_mic(pos[sub] - ref.positions[sub], ref.cell, ref.pbc)
    pos[sub] = ref.positions[sub] + d
    a1, a2 = ref.cell[0], ref.cell[1]
    raw = pos[h].copy()
    cands = [raw + n1 * a1 + n2 * a2 for n1 in range(-2, 3) for n2 in range(-2, 3)]
    best = min(cands, key=lambda p: np.linalg.norm(p[:2] - h_xy_target))
    pos[h] = best
    out.set_positions(pos, apply_constraint=False)
    return out, float(np.linalg.norm(best[:2] - h_xy_target)), float(np.linalg.norm(best - raw))


def make_image(template, positions, fixed, calc):
    im = template.copy()
    im.set_constraint()
    im.set_positions(positions, apply_constraint=False)
    im.set_constraint(FixAtoms(indices=fixed))
    im.calc = calc
    return im


def fmax_free(atoms):
    return float(np.linalg.norm(atoms.get_forces(), axis=1).max())


def band_positions(start, waypoint, end, n):
    """n images in total; two linear segments that meet at the waypoint (index (n-1)//2)."""
    mid = (n - 1) // 2
    seg1 = [start + (waypoint - start) * f for f in np.linspace(0, 1, mid + 1)]
    seg2 = [waypoint + (end - waypoint) * f for f in np.linspace(0, 1, n - mid)]
    return seg1 + seg2[1:]


def linear_positions(start, end, n):
    return [start + (end - start) * f for f in np.linspace(0, 1, n)]


# ------------------------------------------------------------------ setup
def setup(args):
    ep = Path(args.endpoints_dir)
    m = args.model
    f_b1, f_b2, f_t = ep / f"{m}_B1.extxyz", ep / f"{m}_B2.extxyz", ep / f"{m}_T.extxyz"
    for f in (f_b1, f_b2):
        if not f.is_file():
            sys.exit(f"Missing endpoint file: {f}")
    B1, B2 = read(f_b1), read(f_b2)
    if B1.get_chemical_symbols() != B2.get_chemical_symbols():
        sys.exit("B1 and B2 have different atom order.")
    if not np.allclose(B1.cell.array, B2.cell.array, atol=1e-6):
        sys.exit("B1 and B2 have different cells.")
    h = h_index(B1)
    fix1, src = fixed_indices(B1, "B1")
    fix2, _ = fixed_indices(B2, "B2")
    if fix1 != fix2:
        sys.exit("B1 and B2 fix different atoms.")
    fixed = fix1
    cell = B1.cell.array
    tn = top_nb(B1)

    hB1 = B1.positions[h].copy()
    t_xy = hB1[:2] - 0.5 * cell[1][:2]   # B1 -> T : -a/2 along y (T1-01 notes)
    b2_xy = t_xy - 0.5 * cell[0][:2]     # T -> B2 : -a/2 along x
    B2a, off_b2, shift_b2 = align(B1, B2, h, b2_xy)

    if f_t.is_file():
        Traw = read(f_t)
        T, off_t, _ = align(B1, Traw, h, t_xy)
        t_src, t_from_file = f"relaxed {f_t.name}", True
    else:
        T = B1.copy()
        T.set_constraint()
        pos = 0.5 * (B1.positions + B2a.positions)
        pos[h] = [t_xy[0], t_xy[1], B1.positions[tn, 2] + T_HEIGHT_GUESS]
        T.set_positions(pos, apply_constraint=False)
        off_t, t_src, t_from_file = 0.0, "geometric guess (no relaxed T file)", False

    free = np.array([i for i in range(len(B1)) if i not in fixed and i != h])
    checks = {
        "n_atoms": len(B1), "H_index": h, "top_Nb_index": tn,
        "fixed_atoms": len(fixed), "fixed_source": src,
        "cell_a_A": float(np.linalg.norm(cell[0])), "cell_b_A": float(np.linalg.norm(cell[1])),
        "B2_H_offset_from_expected_A": off_b2,
        "B2_H_lattice_shift_applied_A": shift_b2,
        "T_waypoint_source": t_src,
        "T_H_offset_from_expected_A": off_t,
        "T_xy_to_topNb_A": inplane_dist(B1, np.r_[t_xy, 0.0], B1.positions[tn]),
        "fixed_atoms_max_diff_B1_B2_A": float(np.abs(B2a.positions[fixed] - B1.positions[fixed]).max()),
        "free_substrate_max_disp_B1_B2_A": float(np.linalg.norm(B2a.positions[free] - B1.positions[free], axis=1).max()),
    }
    print("Setup checks:")
    for k, v in checks.items():
        print(f"  {k:34s} {v:.4f}" if isinstance(v, float) else f"  {k:34s} {v}")
    if off_b2 > 0.15:
        print("  WARNING: B2 H is not where the T1-01 notes put it (B1 - a/2 y - a/2 x).")
    if checks["T_xy_to_topNb_A"] > 0.15:
        print("  WARNING: the T waypoint is not above the top Nb. Check the site convention.")
    return dict(B1=B1, B2=B2a, T=T, t_from_file=t_from_file, h=h, top_nb=tn,
                fixed=fixed, checks=checks)


def relaxed_T(S, calc, run_dir):
    if S["t_from_file"]:
        return S["T"]
    print("Relaxing T (no relaxed T file found), fmax 0.005 eV/A ...")
    T = make_image(S["B1"], S["T"].positions, S["fixed"], calc)
    BFGS(T, logfile=str(run_dir / "relax_T.log"), maxstep=0.05).run(fmax=0.005, steps=1000)
    write(run_dir / "T_relaxed.extxyz", T)
    return T


# ------------------------------------------------------------------ H-only finite differences
def h_modes(atoms, h, delta):
    """3x3 Hessian of H only (substrate frozen at this image), central differences,
    as in 03_h_modes.py. Negative frequency = imaginary mode."""
    a = atoms.copy()
    a.set_constraint()
    a.calc = atoms.calc
    p0 = a.get_positions()
    hess = np.zeros((3, 3))
    for i in range(3):
        f = []
        for s in (+1, -1):
            p = p0.copy()
            p[h, i] += s * delta
            a.set_positions(p)
            f.append(a.get_forces()[h].copy())
        hess[i] = -(f[0] - f[1]) / (2 * delta)
    hess = 0.5 * (hess + hess.T)
    w, v = np.linalg.eigh(hess)
    m = atoms.get_masses()[h]
    modes = []
    for k in range(3):
        s2 = abs(w[k]) * units._e / (m * units._amu * 1e-20)
        cm = units._hbar * np.sqrt(s2) / units._e / units.invcm
        vec = v[:, k]
        modes.append({"freq_cm-1": float(np.sign(w[k]) * cm),
                      "k_eV_per_A2": float(w[k]),
                      "dominant_axis": "xyz"[int(np.argmax(np.abs(vec)))],
                      "eigvec_xyz": [float(x) for x in vec]})
    return modes


# ------------------------------------------------------------------ NEB
def residual(neb):
    try:
        return float(neb.get_residual())
    except Exception:
        f = np.asarray(neb.get_forces()).reshape(-1, 3)
        return float(np.linalg.norm(f, axis=1).max())


def run_phase(neb, args, run_dir, name, fmax):
    Opt = FIRE if args.optimizer == "fire" else BFGS
    opt = Opt(neb, logfile=str(run_dir / f"{name}.log"),
              trajectory=str(run_dir / f"{name}.traj"), maxstep=0.1)
    t0 = time.perf_counter()

    def report():
        try:
            e = np.asarray(neb.energies)
            emax = f"{np.max(e[1:-1]) - e[0]:+.4f}"
        except Exception:
            emax = "  n/a "
        print(f"  [{name}] step {opt.nsteps:5d}  max NEB force {residual(neb):.4f} eV/A"
              f"  max(E)-E0 {emax} eV  {(time.perf_counter() - t0) / 60:6.1f} min", flush=True)

    opt.attach(report, interval=25)
    opt.run(fmax=fmax, steps=args.steps)
    r = residual(neb)
    out = {"phase": name, "fmax_target": fmax, "steps": int(opt.nsteps), "final_max_force": r,
           "converged": bool(r <= fmax), "minutes": (time.perf_counter() - t0) / 60}
    print(f"  [{name}] done: {out['steps']} steps, max force {r:.4f} eV/A, "
          f"converged={out['converged']}, {out['minutes']:.1f} min")
    return out


def analyse(images, start_pos, end_pos, S, args, run_dir, e_b1):
    h, tn = S["h"], S["top_nb"]
    n = len(images)
    E = np.array([im.get_potential_energy() for im in images])
    rows = []
    for i, im in enumerate(images):
        p = im.positions[h]
        rows.append({"image": i, "E_rel_first_eV": float(E[i] - E[0]),
                     "E_rel_B1_eV": float(E[i] - e_b1),
                     "H_dxy_to_topNb_A": inplane_dist(im, p, im.positions[tn]),
                     "H_height_above_topNb_A": float(p[2] - im.positions[tn, 2]),
                     "true_fmax_free_eV_per_A": fmax_free(im)})
    interior = list(range(1, n - 1))
    i_max = max(interior, key=lambda i: E[i])
    minima = [i for i in interior if E[i] < E[i - 1] and E[i] < E[i + 1]]
    maxima = [i for i in interior if E[i] > E[i - 1] and E[i] > E[i + 1]]
    t_like = [i for i in minima if rows[i]["H_dxy_to_topNb_A"] < 0.3]

    fit = None
    try:
        try:
            from ase.utils.forcecurve import fit_images  # ASE >= 3.23
            s, Ei, Sf, Ef, _ = fit_images(images)
        except ImportError:
            s, Ei, Sf, Ef, _ = NEBTools(images).get_fit()
        bar_fit, dE_fit = NEBTools(images).get_barrier(fit=True)
        fit = {"s": np.asarray(s), "E": np.asarray(Ei) - Ei[0],
               "Sf": np.asarray(Sf), "Ef": np.asarray(Ef) - Ei[0],
               "barrier_fit_eV": float(bar_fit), "dE_fit_eV": float(dE_fit)}
    except Exception as exc:
        print(f"  NOTE: NEBTools fit failed ({exc}); reporting image energies only")

    print("\nH-only finite-difference modes (substrate frozen at the image):")
    modes = {}
    for i in sorted(set([i_max] + minima)):
        kind = "highest image (CI)" if i == i_max else "interior minimum"
        modes[str(i)] = {"kind": kind, "modes": h_modes(images[i], h, args.fd_delta)}
        txt = ", ".join(f"{md['freq_cm-1']:+.0f} ({md['dominant_axis']})" for md in modes[str(i)]["modes"])
        print(f"  image {i:2d} [{kind}]: {txt} cm-1")
    n_imag = sum(md["freq_cm-1"] < 0 for md in modes[str(i_max)]["modes"])

    endpoints_moved = float(max(np.abs(images[0].positions - start_pos).max(),
                                np.abs(images[-1].positions - end_pos).max()))
    res = {
        "framing": FRAMING,
        "barrier_images_eV": float(E[i_max] - E[0]),
        "barrier_fit_eV": fit["barrier_fit_eV"] if fit else None,
        "reaction_energy_eV": float(E[-1] - E[0]),
        "climbing_image": i_max,
        "interior_maxima": maxima,
        "interior_minima": minima,
        "T_intermediate_found": bool(t_like),
        "saddle_imaginary_H_modes": int(n_imag),
        "endpoint_fmax_eV_per_A": [rows[0]["true_fmax_free_eV_per_A"], rows[-1]["true_fmax_free_eV_per_A"]],
        "endpoint_max_position_change_A": endpoints_moved,
        "h_modes": modes,
        "images": rows,
    }

    with open(run_dir / "profile.csv", "w", encoding="utf-8") as f:
        f.write(",".join(rows[0].keys()) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in r.values()) + "\n")
    write(run_dir / "band_final.extxyz", images)
    plot_profile(images, E, fit, i_max, minima, args, run_dir)
    return res


def plot_profile(images, E, fit, i_max, minima, args, run_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    if fit:
        ax.plot(fit["Sf"], fit["Ef"], "-", color="C0", lw=1.5, label="spline fit")
        x = fit["s"]
    else:
        x = np.arange(len(E))
    ax.plot(x, E - E[0], "o", color="C0", label="images")
    ax.plot(x[i_max], E[i_max] - E[0], "s", color="C3", label="climbing image")
    for i in minima:
        ax.annotate("T?", (x[i], E[i] - E[0]), textcoords="offset points", xytext=(0, -14), ha="center")
    ax.set_xlabel("path length (A)" if fit else "image")
    ax.set_ylabel("E - E(first) (eV)")
    ax.set_title(f"{args.model}, {args.path}, {len(E)} images: MLIP pathway estimate", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(run_dir / "profile.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ main
def run_dir_for(args):
    name = f"run_{dt.date.today().isoformat()}_{args.model}_{args.nimages}"
    if args.path != "full":
        name += f"_{args.path}"
    if args.k != 0.1:
        name += f"_k{args.k:g}"
    if args.optimizer != "fire":
        name += f"_{args.optimizer}"
    if args.tag:
        name += f"_{args.tag}"
    if args.dry_run:
        d = HERE / name.replace("run_", "dryrun_", 1)
        d.mkdir(parents=True, exist_ok=True)
        return d
    d, v = HERE / name, 2
    while d.exists() and any(d.iterdir()):
        d, v = HERE / f"{name}_v{v}", v + 1
    d.mkdir(parents=True, exist_ok=True)
    return d


def dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def main():
    ap = argparse.ArgumentParser(description="MLIP-driven CI-NEB B1 -> T -> B2 (T1-02)")
    ap.add_argument("--model", choices=["M0", "M1"], required=True)
    ap.add_argument("--model-path", help="MACE checkpoint (or env NEB_<MODEL>_MODEL)")
    ap.add_argument("--nimages", type=int, default=9, help="total images including endpoints")
    ap.add_argument("--path", choices=["full", "B1T", "TB2"], default="full")
    ap.add_argument("--k", type=float, default=0.1, help="spring constant, eV/A^2")
    ap.add_argument("--optimizer", choices=["fire", "bfgs"], default="fire")
    ap.add_argument("--fmax1", type=float, default=0.03, help="plain NEB target, eV/A")
    ap.add_argument("--fmax2", type=float, default=0.01, help="climbing-image target, eV/A")
    ap.add_argument("--steps", type=int, default=1500, help="max steps per phase")
    ap.add_argument("--fd-delta", type=float, default=0.01, help="H displacement for modes, A")
    ap.add_argument("--endpoints-dir", default=str(DEFAULT_ENDPOINTS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run", action="store_true", help="build band, single points, cost estimate")
    args = ap.parse_args()
    if args.nimages < (5 if args.path == "full" else 3):
        sys.exit("Too few images.")

    run_dir = run_dir_for(args)
    sys.stdout = Tee(run_dir / "run.log")
    print(f"T1-02 CI-NEB | {FRAMING}")
    print(f"Output folder: {run_dir.name}")

    model_path = resolve_model_path(args.model, args.model_path)
    prov = provenance(args.model, model_path)
    print(f"Model {args.model}: {prov['model_path']}")
    if "model_sha256" in prov:
        sha = prov["model_sha256"]
        print(f"  SHA-256 {sha[:8]}...{sha[-4:]}")
        if args.model == "M1" and not (sha.startswith(M1_SHA_HEAD) and sha.endswith(M1_SHA_TAIL)):
            print("  WARNING: this is NOT the T1-01 M1 checkpoint (fa762686...cf8d).")
    print(f"  ASE {prov['ase']}, torch {prov.get('torch')}, mace {prov.get('mace')}, "
          f"threads {prov.get('torch_threads')}")

    S = setup(args)
    n = args.nimages

    t_load = time.perf_counter()
    calcs = [make_calc(args.model, model_path) for _ in range(n)]
    print(f"Loaded {n} calculators in {time.perf_counter() - t_load:.1f} s")

    if args.path == "full":
        start, end = S["B1"], S["B2"]
        positions = band_positions(start.positions, S["T"].positions, end.positions, n)
    else:
        T = relaxed_T(S, calcs[0], run_dir)
        start, end = (S["B1"], T) if args.path == "B1T" else (T, S["B2"])
        positions = linear_positions(start.positions, end.positions, n)
    start_pos, end_pos = positions[0].copy(), positions[-1].copy()
    images = [make_image(S["B1"], p, S["fixed"], c) for p, c in zip(positions, calcs)]
    write(run_dir / "band_initial.extxyz", images)

    # reference energy of B1 with this model (for split bands)
    b1 = make_image(S["B1"], S["B1"].positions, S["fixed"], calcs[0])
    e_b1 = b1.get_potential_energy()

    # single points on the initial band + timing
    for im in (images[0], images[-1]):
        im.get_forces()
    t0 = time.perf_counter()
    for im in images[1:-1]:
        im.get_forces()
    t_call = (time.perf_counter() - t0) / (n - 2)
    E = np.array([im.get_potential_energy() for im in images])
    print("\nInitial band (single points):")
    print("  img   E-E(B1) eV   H dxy to top Nb (A)   H height (A)   true fmax (eV/A)")
    for i, im in enumerate(images):
        p = im.positions[S["h"]]
        print(f"  {i:3d}   {E[i] - e_b1:+9.4f}   {inplane_dist(im, p, im.positions[S['top_nb']]):10.3f}"
              f"            {p[2] - im.positions[S['top_nb'], 2]:7.3f}        {fmax_free(im):8.4f}")
    ep_fmax = [fmax_free(images[0]), fmax_free(images[-1])]
    print(f"Endpoint fmax with this model: {ep_fmax[0]:.4f}, {ep_fmax[1]:.4f} eV/A "
          f"(expect <= ~0.005 if the model matches the endpoint relaxation)")

    per_step = t_call * (n - 2)
    est = {"seconds_per_force_call": t_call, "seconds_per_neb_step": per_step,
           "phase1_minutes_range(150-600 steps)": [per_step * 150 / 60, per_step * 600 / 60],
           "phase2_minutes_range(100-400 steps)": [per_step * 100 / 60, per_step * 400 / 60]}
    print(f"Cost: {t_call:.2f} s per force call, {per_step:.1f} s per NEB step "
          f"-> plain NEB ~{est['phase1_minutes_range(150-600 steps)'][0]:.0f}-"
          f"{est['phase1_minutes_range(150-600 steps)'][1]:.0f} min, CI ~"
          f"{est['phase2_minutes_range(100-400 steps)'][0]:.0f}-"
          f"{est['phase2_minutes_range(100-400 steps)'][1]:.0f} min (typical step counts, not a guarantee)")

    summary = {"framing": FRAMING, "args": vars(args), "provenance": prov, "setup_checks": S["checks"],
               "E_B1_abs_eV": float(e_b1), "initial_band_E_rel_B1_eV": (E - e_b1).tolist(),
               "endpoint_fmax_eV_per_A": ep_fmax, "cost_estimate": est}
    if args.dry_run:
        dump(summary, run_dir / "summary_dryrun.json")
        print(f"\nDry run done. Files in {run_dir.name}")
        return

    print("\nPhase 1: plain NEB (improved tangent)")
    neb = NEB(images, k=args.k, climb=False, method="improvedtangent")
    ph1 = run_phase(neb, args, run_dir, "phase1_neb", args.fmax1)
    print("Phase 2: climbing image")
    neb.climb = True
    ph2 = run_phase(neb, args, run_dir, "phase2_ci", args.fmax2)

    res = analyse(images, start_pos, end_pos, S, args, run_dir, e_b1)
    summary.update({"phases": [ph1, ph2], "result": res})
    dump(summary, run_dir / "summary.json")

    print("\n=== Result (MLIP pathway estimate, not DFT) ===")
    print(f"  barrier, highest image   : {res['barrier_images_eV']:.4f} eV (image {res['climbing_image']})")
    if res["barrier_fit_eV"] is not None:
        print(f"  barrier, spline fit      : {res['barrier_fit_eV']:.4f} eV")
    print(f"  reaction energy last-first: {res['reaction_energy_eV']:+.4f} eV")
    print(f"  interior minima          : {res['interior_minima']}  (T intermediate: {res['T_intermediate_found']})")
    print(f"  imaginary H modes at CI  : {res['saddle_imaginary_H_modes']} (want 1)")
    print(f"  endpoints moved          : {res['endpoint_max_position_change_A']:.2e} A")
    print(f"  converged                : phase1 {ph1['converged']}, phase2 {ph2['converged']}")
    if res["T_intermediate_found"] and args.path == "full":
        print("  -> T is an intermediate: the CI climbs only the higher saddle. "
              "Run --path B1T and --path TB2 to resolve both saddles.")
    print(f"Files in {run_dir.name}")


if __name__ == "__main__":
    main()
