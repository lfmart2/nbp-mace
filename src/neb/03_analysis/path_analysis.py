"""T1-03: path analysis + figures for the MLIP-driven CI-NEB of H on NbP (B1 -> T -> B2).

Reads the canonical split-band runs in 02_neb_runs (band_final.extxyz with stored energies
and forces), stitches B1->T and T->B2 for each model, and writes

  <out>/path_geometry.csv     per image: path coordinates, E - E(B1), H height, H-Nb distances, |F_H|
  <out>/key_points.csv/.json  B1, saddle 1, T, saddle 2, B2 per model (+ H-only modes/ZPE with --zpe)
  <figdir>/fig_neb_path.png/.pdf      3 panels: energy, H height, H-Nb distances vs H lateral path
  <figdir>/fig_neb_energy.png/.pdf    energy panel only (README-sized)

--zpe recomputes the H-only Hessian (central differences, substrate frozen at the image, as in
03_h_modes.py) at the five key points with MACE (env vars from your private env.ps1) and reports
ZPE-corrected energies. Sensitivity estimate only: M1 H frequencies are 17-61 % too stiff.

All energies are MLIP pathway estimates, not DFT. Outputs contain no absolute local paths.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
from ase import units
from ase.geometry import find_mic
from ase.io import read

H_IDX = 0
TOP_NB_IDX = 1
KEYS = ["B1", "S1", "T", "S2", "B2"]

RUNS = {
    "M1": {"B1T": "run_2026-10-10_M1_9_B1T", "TB2": "run_2026-10-10_M1_9_TB2"},
    "M0": {"B1T": "run_2026-10-10_M0_7_B1T", "TB2": "run_2026-10-10_M0_7_TB2_v2"},
}
ENV_CKPT = {"M1": "MACE_M1_CHECKPOINT", "M0": "MACE_MP0_SMALL_CHECKPOINT"}
LABEL = {"M1": "M1 (fine-tuned)", "M0": "M0 (MACE-MP-0-small)"}
COLOR = {"M1": "#1f77b4", "M0": "#ff7f0e"}
# T1-02 reference values (eV vs B1) to flag accidental input mix-ups
T102 = {"M1": {"S1": 0.252, "T": 0.187, "S2": 0.230, "B2": 0.008},
        "M0": {"S1": 0.295, "T": 0.238, "S2": 0.279, "B2": 0.010}}


# ----------------------------------------------------------------------------- geometry
def nb_image_distances(a, nb):
    """Sorted H-Nb distances over in-plane periodic images (in a 1x1 cell the two bridge Nb
    are the same atom and its image, which mic=True would collapse to one distance)."""
    h = a.positions[H_IDX]
    cell = a.cell.array
    shifts = np.array([i * cell[0] + j * cell[1] for i in (-1, 0, 1) for j in (-1, 0, 1)])
    d = []
    for idx in nb:
        v = find_mic((a.positions[idx] - h)[None, :], a.cell, a.pbc)[0][0]
        d.extend(np.linalg.norm(v + shifts, axis=1))
    return np.sort(d)


def band_rows(images, E_ref, band, s_all0, s_lat0):
    """Per-image geometry/energy rows; path coordinates offset by s_all0 / s_lat0."""
    cell, pbc = images[0].cell, images[0].pbc
    nb = [i for i, s in enumerate(images[0].get_chemical_symbols()) if s == "Nb"]
    rows, s_all, s_lat = [], s_all0, s_lat0
    for k, a in enumerate(images):
        if k:
            p = images[k - 1]
            dR = find_mic(a.positions - p.positions, cell, pbc)[0]
            s_all += float(np.sqrt((dR ** 2).sum()))
            rel = (a.positions[H_IDX] - a.positions[TOP_NB_IDX]) - (p.positions[H_IDX] - p.positions[TOP_NB_IDX])
            rel = find_mic(rel[None, :], cell, pbc)[0][0]
            s_lat += float(np.hypot(rel[0], rel[1]))
        d = nb_image_distances(a, nb)
        rows.append({
            "band": band, "image": k,
            "s_allatom_A": s_all, "s_H_lateral_A": s_lat,
            "E_rel_eV": float(a.get_potential_energy() - E_ref),
            "H_height_above_topNb_A": float(a.positions[H_IDX, 2] - a.positions[TOP_NB_IDX, 2]),
            "d_H_topNb_A": float(a.get_distance(H_IDX, TOP_NB_IDX, mic=True)),
            "d_H_Nb_1st_A": float(d[0]), "d_H_Nb_2nd_A": float(d[1]),
            "F_H_norm_eV_A": float(np.linalg.norm(a.get_forces(apply_constraint=False)[H_IDX])),
        })
    return rows


def energy_fit(images, rows, method="pchip"):
    """Display curve on the H-lateral coordinate. 'pchip': shape-preserving through the images
    (no overshoot). 'force': ASE forcecurve fit on the all-atom path, remapped (can kink)."""
    s_lat = np.array([r["s_H_lateral_A"] for r in rows])
    E = np.array([r["E_rel_eV"] for r in rows])
    if method == "pchip":
        from scipy.interpolate import PchipInterpolator
        x = np.linspace(s_lat[0], s_lat[-1], 40 * (len(s_lat) - 1) + 1)
        return x, PchipInterpolator(s_lat, E)(x), "PCHIP through image energies (guide to the eye)"
    try:
        from ase.utils.forcecurve import fit_images
        ff = fit_images(images)
        x = np.interp(ff.fit_path, ff.path, s_lat)
        y = np.asarray(ff.fit_energies) - np.asarray(ff.energies)[0] + E[0]  # robust to ASE's own shift
        return x, y, "ase forcecurve (energies + tangential forces)"
    except Exception as exc:  # fall back to a shape-preserving spline through the energies only
        from scipy.interpolate import PchipInterpolator
        x = np.linspace(s_lat[0], s_lat[-1], 20 * (len(s_lat) - 1) + 1)
        return x, PchipInterpolator(s_lat, E)(x), f"PCHIP on energies (forcecurve failed: {exc})"


# ----------------------------------------------------------------------------- H-only modes
def load_calc(model):
    var = ENV_CKPT[model]
    path = os.environ.get(var)
    if not path or not Path(path).exists():
        sys.exit(f"--zpe needs {var}; run  . your private env.ps1  first")
    from mace.calculators import MACECalculator
    try:
        calc = MACECalculator(model_paths=path, device="cpu", default_dtype="float64")
    except TypeError:
        calc = MACECalculator(model_path=path, device="cpu", default_dtype="float64")
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return calc, sha


def h_modes(atoms, calc, delta=0.01):
    """3x3 H-only Hessian by central differences; frequencies in cm-1 (negative = imaginary)."""
    a = atoms.copy()
    a.calc = calc
    x0 = a.positions[H_IDX].copy()
    hess = np.zeros((3, 3))
    for j in range(3):
        f = []
        for sgn in (+1, -1):
            pos = a.positions.copy()
            pos[H_IDX] = x0
            pos[H_IDX, j] += sgn * delta
            a.positions = pos
            f.append(a.get_forces(apply_constraint=False)[H_IDX])
        hess[:, j] = -(f[0] - f[1]) / (2 * delta)
    hess = 0.5 * (hess + hess.T)
    w, v = np.linalg.eigh(hess / atoms[H_IDX].mass)  # eV / A^2 / amu
    conv = np.sqrt(units._e / units._amu) * 1e10 / (2 * np.pi * units._c * 100)  # -> cm-1
    nu = np.sign(w) * np.sqrt(np.abs(w)) * conv
    axes = ["xyz"[int(np.argmax(np.abs(v[:, i])))] for i in range(3)]
    zpe = 0.5 * float(np.sum(nu[nu > 0])) * units.invcm
    return nu, axes, zpe


# ----------------------------------------------------------------------------- figures
def site_marks(ax, kp, labels=True):
    ref = "M1" if "M1" in kp else next(iter(kp))
    for name in ("B1", "T", "B2"):
        x = kp[ref][name]["s_H_lateral_A"]
        ax.axvline(x, color="0.75", lw=0.8, ls=":", zorder=0)
        if labels:
            ax.text(x, 1.01, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9, color="0.3")


def plot_energy(ax, data, kp, annotate=True):
    for m in data:
        x, y = data[m]["fit"]
        ax.plot(x, y, color=COLOR[m], lw=1.6, label=LABEL[m])
        ax.plot([r["s_H_lateral_A"] for r in data[m]["rows"]], [r["E_rel_eV"] for r in data[m]["rows"]],
                "o", color=COLOR[m], ms=4)
        if annotate:
            for s in ("S1", "S2"):
                p = kp[m][s]
                ax.annotate(f"{p['E_rel_eV']:.3f}", (p["s_H_lateral_A"], p["E_rel_eV"]),
                            textcoords="offset points", xytext=(0, 5), ha="center",
                            fontsize=8, color=COLOR[m])
    ax.axhline(0, color="0.7", lw=0.6)
    ax.set_ylabel("E − E(B1) (eV)")
    ax.set_ylim(-0.02, max(kp[m]["S1"]["E_rel_eV"] for m in kp) * 1.22)
    site_marks(ax, kp)
    ax.legend(loc="lower center", frameon=False, fontsize=8)


def make_figures(data, kp, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    title = "H on Nb-terminated NbP, 1 ML: MLIP pathway estimate (not DFT)"

    fig, axs = plt.subplots(3, 1, figsize=(6.0, 7.6), sharex=True,
                            gridspec_kw={"height_ratios": [1.4, 1, 1]})
    plot_energy(axs[0], data, kp)
    axs[0].set_title(title, fontsize=10, pad=16)
    for m in data:
        r = data[m]["rows"]
        x = [q["s_H_lateral_A"] for q in r]
        axs[1].plot(x, [q["H_height_above_topNb_A"] for q in r], "o-", color=COLOR[m], ms=3.5, lw=1.2)
        axs[2].plot(x, [q["d_H_Nb_1st_A"] for q in r], "o-", color=COLOR[m], ms=3.5, lw=1.2)
        axs[2].plot(x, [q["d_H_Nb_2nd_A"] for q in r], "s--", color=COLOR[m], ms=3, lw=1.0, mfc="none")
    axs[1].set_ylabel("H height above\nsurface Nb (Å)")
    axs[2].set_ylabel("H–Nb distance (Å)")
    axs[2].plot([], [], "k-", lw=1.2, label="nearest Nb")
    axs[2].plot([], [], "k--", lw=1.0, label="2nd-nearest Nb")
    axs[2].legend(frameon=False, fontsize=8, loc="upper right")
    for ax in axs[1:]:
        site_marks(ax, kp, labels=False)
    axs[2].set_xlabel("H lateral path from B1 (Å)")
    for ax, tag in zip(axs, "abc"):
        ax.text(0.01, 0.97, f"({tag})", transform=ax.transAxes, va="top", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"fig_neb_path.{ext}", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    plot_energy(ax, data, kp)
    ax.set_title(title, fontsize=10, pad=16)
    ax.set_xlabel("H lateral path from B1 (Å)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"fig_neb_energy.{ext}", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    here = Path(__file__).resolve().parent
    stamp = datetime.date.today().isoformat()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=str(here.parent / "02_neb_runs"))
    ap.add_argument("--out", default=str(here / f"{stamp}_path_analysis"))
    ap.add_argument("--figdir", default=str(here.parent / "04_figures" / stamp))
    ap.add_argument("--models", nargs="+", default=["M1", "M0"], choices=list(RUNS))
    ap.add_argument("--fit", choices=["pchip", "force"], default="pchip", help="energy curve in the figures")
    ap.add_argument("--zpe", action="store_true", help="H-only modes + ZPE at key points (needs MACE env vars)")
    ap.add_argument("--delta", type=float, default=0.01, help="finite-difference step for --zpe (Å)")
    args = ap.parse_args()

    runs_dir, out, figdir = Path(args.runs_dir), Path(args.out), Path(args.figdir)
    out.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    data, kp, warnings, prov = {}, {}, [], {"date": stamp, "inputs": {}, "checkpoint_sha256": {}}
    for m in args.models:
        bands = {}
        for b in ("B1T", "TB2"):
            f = runs_dir / RUNS[m][b] / "band_final.extxyz"
            if not f.exists():
                sys.exit(f"missing {f}")
            bands[b] = read(f, ":")
            prov["inputs"][f"{m}_{b}"] = {"run": RUNS[m][b], "file": "band_final.extxyz",
                                          "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                                          "n_images": len(bands[b])}
        E_ref = bands["B1T"][0].get_potential_energy()
        r1 = band_rows(bands["B1T"], E_ref, "B1T", 0.0, 0.0)
        r2 = band_rows(bands["TB2"], E_ref, "TB2", r1[-1]["s_allatom_A"], r1[-1]["s_H_lateral_A"])

        # T must be the same structure at the end of band 1 and the start of band 2
        dE = abs(r1[-1]["E_rel_eV"] - r2[0]["E_rel_eV"])
        dX = float(np.abs(find_mic(bands["B1T"][-1].positions - bands["TB2"][0].positions,
                                   bands["TB2"][0].cell, bands["TB2"][0].pbc)[0]).max())
        if dE > 1e-4 or dX > 1e-3:
            warnings.append(f"{m}: T differs between bands (dE {dE:.2e} eV, max dx {dX:.2e} Å)")

        x1, y1, how = energy_fit(bands["B1T"], r1, args.fit)
        x2, y2, _ = energy_fit(bands["TB2"], r2, args.fit)
        rows = r1 + r2[1:]  # drop duplicate T
        for r in rows:
            r["model"] = m
        data[m] = {"rows": rows, "fit": (np.concatenate([x1, x2[1:]]), np.concatenate([y1, y2[1:]])),
                   "fit_method": how}

        e1 = [r["E_rel_eV"] for r in r1]
        e2 = [r["E_rel_eV"] for r in r2]
        i1, i2 = 1 + int(np.argmax(e1[1:-1])), 1 + int(np.argmax(e2[1:-1]))
        picks = {"B1": ("B1T", 0), "S1": ("B1T", i1), "T": ("B1T", len(r1) - 1),
                 "S2": ("TB2", i2), "B2": ("TB2", len(r2) - 1)}
        kp[m] = {}
        for k, (b, i) in picks.items():
            row = dict((r1 if b == "B1T" else r2)[i])
            row["atoms"] = bands[b][i]
            kp[m][k] = row
        for k, ref in T102[m].items():
            if abs(kp[m][k]["E_rel_eV"] - ref) > 0.002:
                warnings.append(f"{m} {k}: {kp[m][k]['E_rel_eV']:.3f} eV vs T1-02 {ref:.3f} eV")

    if args.zpe:
        for m in args.models:
            calc, sha = load_calc(m)
            prov["checkpoint_sha256"][m] = sha
            for k in KEYS:
                nu, ax, zpe = h_modes(kp[m][k]["atoms"], calc, args.delta)
                n_imag = int((nu < 0).sum())
                expect = 1 if k.startswith("S") else 0
                if n_imag != expect:
                    warnings.append(f"{m} {k}: {n_imag} imaginary H modes (expected {expect})")
                kp[m][k].update({"modes_cm1": [round(float(v), 1) for v in nu], "mode_axes": ax,
                                 "n_imaginary": n_imag, "ZPE_H_eV": zpe})
            z0 = kp[m]["B1"]["ZPE_H_eV"]
            for k in KEYS:
                p = kp[m][k]
                p["E_rel_ZPE_eV"] = p["E_rel_eV"] + p["ZPE_H_eV"] - z0

    # ---- write tables
    cols = ["model", "band", "image", "s_allatom_A", "s_H_lateral_A", "E_rel_eV", "H_height_above_topNb_A",
            "d_H_topNb_A", "d_H_Nb_1st_A", "d_H_Nb_2nd_A", "F_H_norm_eV_A"]
    with open(out / "path_geometry.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for m in data:
            for r in data[m]["rows"]:
                w.writerow({c: (round(v, 5) if isinstance(v, float) else v) for c, v in r.items()})

    kcols = ["model", "point", "band", "image", "s_H_lateral_A", "E_rel_eV", "H_height_above_topNb_A",
             "d_H_topNb_A", "d_H_Nb_1st_A", "d_H_Nb_2nd_A", "F_H_norm_eV_A"]
    if args.zpe:
        kcols += ["modes_cm1", "mode_axes", "n_imaginary", "ZPE_H_eV", "E_rel_ZPE_eV"]
    krows = []
    for m in kp:
        for k in KEYS:
            p = {c: v for c, v in kp[m][k].items() if c != "atoms"}
            p.update({"model": m, "point": k})
            krows.append({c: (round(p[c], 5) if isinstance(p.get(c), float) else p.get(c)) for c in kcols})
    with open(out / "key_points.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=kcols)
        w.writeheader()
        w.writerows(krows)

    import ase, matplotlib
    prov["versions"] = {"python": platform.python_version(), "ase": ase.__version__,
                        "numpy": np.__version__, "matplotlib": matplotlib.__version__}
    if args.zpe:
        import mace
        prov["versions"]["mace"] = getattr(mace, "__version__", "unknown")
    prov["energy_fit"] = {m: data[m]["fit_method"] for m in data}
    prov["note"] = "MLIP pathway estimates (not DFT); 1 ML H, 1x1 cell; ZPE is H-only, sensitivity only"
    with open(out / "key_points.json", "w") as fh:
        json.dump({"provenance": prov, "warnings": warnings, "key_points": krows}, fh, indent=2)

    make_figures(data, kp, figdir)

    # ---- console summary
    print(f"\n{'model':5} {'point':5} {'E_rel':>7} {'h_H':>6} {'d1':>6} {'d2':>6} {'|F_H|':>7}"
          + ("  modes (cm-1)            E+ZPE" if args.zpe else ""))
    for r in krows:
        line = (f"{r['model']:5} {r['point']:5} {r['E_rel_eV']:7.3f} {r['H_height_above_topNb_A']:6.3f} "
                f"{r['d_H_Nb_1st_A']:6.3f} {r['d_H_Nb_2nd_A']:6.3f} {r['F_H_norm_eV_A']:7.4f}")
        if args.zpe:
            md = " ".join(f"{v:7.0f}{a}" for v, a in zip(r["modes_cm1"], r["mode_axes"]))
            line += f"  {md}  {r['E_rel_ZPE_eV']:6.3f}"
        print(line)
    print("\nWARNINGS:\n  " + "\n  ".join(warnings) if warnings else "\nno warnings")
    print(f"\ntables -> {out.name}\\   figures -> {figdir.parent.name}\\{figdir.name}\\")


if __name__ == "__main__":
    main()
