#!/usr/bin/env python3
"""T1-01 step 4: MLIP potential-energy scans of one H over the NbP surface cell.

  z   dense vertical curve at the r0 lateral position, substrate frozen at DFT r0
      (same protocol as the DFT z-scan, so the 9 DFT points are overlaid directly).
  xy  lateral 2D map: H x,y fixed on an n x n grid over the surface cell, H z relaxed,
      substrate frozen at DFT r0 (or --relax-substrate). Finds local minima, labels the
      sites, and estimates grid minimax barriers between them (frozen-substrate estimate).

Everything is an MLIP estimate. The 2D map is NOT validated by any DFT data off the r0 site.

Usage:
    python 04_pes_scan.py z                       # M0 and M1
    python 04_pes_scan.py xy --models M1 --n 12   # ~144 constrained relaxations
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np
from ase.constraints import FixAtoms, FixCartesian
from ase.optimize import BFGS

import endpoint_common as ec

CM1 = 521.4709
M_H = 1.00794


def harmonic_fit(h, e, center, window=0.2):
    sel = np.abs(h - center) <= window
    if sel.sum() < 4:
        return None
    c = np.polyfit(h[sel] - center, e[sel], 3)
    d1, d2 = np.polyder(c), np.polyder(c, 2)
    roots = [r.real for r in np.roots(d1) if abs(r.imag) < 1e-9 and np.polyval(d2, r.real) > 0]
    if not roots:
        return None
    x = min(roots, key=abs)
    k = float(np.polyval(d2, x))
    return {"h_min_A": float(center + x), "k_eV_A2": k, "nu_z_cm1_H_only": float(CM1 * np.sqrt(k / M_H))}


# ------------------------------------------------------------------------------------- z scan
def run_z(args, r0, hi, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = r0.positions[np.array(r0.get_chemical_symbols()) != "H", 2].max()
    heights = np.round(np.arange(args.hmin, args.hmax + 1e-9, args.dh), 4)
    frames = ec.load_zscan()
    dft = sorted(((ec.surface_height(a, a.positions[hi, 2]), float(a.info["REF_energy"]),
                   float(a.arrays["REF_forces"][hi, 2]), a.info["r_label"]) for a in frames))
    dh_ = np.array([d[0] for d in dft])
    de_ = np.array([d[1] for d in dft])
    de_ -= de_.min()
    res = {"provenance": ec.provenance(), "protocol": "substrate frozen at DFT r0; H at r0 x,y",
           "dft": {"h_A": dh_, "E_rel_eV": de_, "Fz_eV_A": [d[2] for d in dft], "labels": [d[3] for d in dft]},
           "models": {}}
    dfit = harmonic_fit(dh_, de_, dh_[np.argmin(de_)], window=0.25)
    res["dft"]["fit"] = dfit
    rows = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(dh_, de_, "ko", label="DFT (collinear SCF)", zorder=5)
    ax2.plot(dh_, [d[2] for d in dft], "ko", label="DFT", zorder=5)
    for model in args.models:
        calc = ec.make_calc(model, args.device)
        a = r0.copy()
        a.calc = calc
        E, Fz = [], []
        for h in heights:
            a.positions[hi, 2] = top + h
            E.append(a.get_potential_energy())
            Fz.append(a.get_forces()[hi, 2])
        E = np.array(E)
        # align to the model's energy at the DFT minimum height, as the repo aligns to minima
        ref = np.interp(dh_[np.argmin(de_)], heights, E)
        Erel = E - E.min()
        fit = harmonic_fit(heights, E, heights[np.argmin(E)])
        Edft_pts = np.interp(dh_, heights, E)
        res["models"][model] = {"h_A": heights, "E_rel_eV": Erel, "Fz_eV_A": Fz, "fit": fit,
                                "E_at_dft_heights_rel_eV": Edft_pts - Edft_pts.min(),
                                "E_at_dft_min_minus_model_min_eV": float(ref - E.min())}
        for h, e, f in zip(heights, Erel, Fz):
            rows.append({"model": model, "height_A": h, "E_rel_eV": e, "H_Fz_eV_A": f})
        ax1.plot(heights, Erel, label=model)
        ax2.plot(heights, Fz, label=model)
        if fit:
            print(f"[{model}] minimum h = {fit['h_min_A']:.3f} A, k = {fit['k_eV_A2']:.2f} eV/A^2, "
                  f"nu_z(H only) = {fit['nu_z_cm1_H_only']:.0f} cm-1")
    if dfit:
        print(f"[DFT] minimum h = {dfit['h_min_A']:.3f} A, k = {dfit['k_eV_A2']:.2f} eV/A^2, "
              f"nu_z(H only) = {dfit['nu_z_cm1_H_only']:.0f} cm-1  (cubic fit to 9 points; coarse)")
    ax1.set(xlabel="H height above top substrate atom (A)", ylabel="E - E_min (eV)",
            title="Frozen-substrate z-scan (MLIP estimate)", ylim=(-0.05, 1.5))
    ax2.axhline(0, color="grey", lw=0.8)
    ax2.set(xlabel="H height (A)", ylabel="H Fz (eV/A)", title="H vertical force", ylim=(-4, 4))
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out / "zscan_dense.png", dpi=160)
    with (out / "zscan_dense.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ec.write_json(out / "zscan_dense.json", res)
    print(f"Wrote {out}")


# ------------------------------------------------------------------------------------ xy map
def local_minima(E):
    n0, n1 = E.shape
    mins = []
    for i in range(n0):
        for j in range(n1):
            nb = [E[(i + di) % n0, (j + dj) % n1] for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
            if E[i, j] <= min(nb) + 1e-6:
                mins.append((i, j))
    # merge plateau duplicates (adjacent equal minima)
    keep = []
    for m in mins:
        if not any(abs(m[0] - k[0]) % E.shape[0] in (0, 1, E.shape[0] - 1)
                   and abs(m[1] - k[1]) % E.shape[1] in (0, 1, E.shape[1] - 1) for k in keep):
            keep.append(m)
    return keep


def minimax_barriers(E, mins):
    """Grid minimax barrier from each minimum to (a) its own periodic image and (b) every other
    minimum, using a 3x3 tiling and 4-neighbour connectivity (frozen-substrate, coarse grid)."""
    n0, n1 = E.shape
    T = np.tile(E, (3, 3))
    N0, N1 = T.shape
    parent = np.arange(N0 * N1)
    active = np.zeros(N0 * N1, bool)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cid = lambda i, j: i * N1 + j  # noqa: E731
    centre = {k: cid(n0 + m[0], n1 + m[1]) for k, m in enumerate(mins)}
    targets = {}
    for a, ma in enumerate(mins):
        imgs = [cid(n0 + ma[0] + di * n0, n1 + ma[1] + dj * n1)
                for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
        targets[(a, "image")] = imgs
        for b, mb in enumerate(mins):
            if b != a:
                targets[(a, b)] = [cid(mb[0] + di * n0, mb[1] + dj * n1) for di in range(3) for dj in range(3)]
    result = {}
    order = np.argsort(T, axis=None)
    for flat in order:
        i, j = divmod(int(flat), N1)
        active[flat] = True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + di, j + dj
            if 0 <= ii < N0 and 0 <= jj < N1 and active[cid(ii, jj)]:
                ra, rb = find(flat), find(cid(ii, jj))
                if ra != rb:
                    parent[ra] = rb
        for key, tg in targets.items():
            if key in result or not active[centre[key[0]]]:
                continue
            r = find(centre[key[0]])
            if any(active[t] and find(t) == r for t in tg):
                result[key] = float(T.flat[flat] - E[mins[key[0]]])
        if len(result) == len(targets):
            break
    return result


def run_xy(args, r0, hi, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = ec.substrate_layers(r0)
    cell = r0.cell.array
    sym = np.array(r0.get_chemical_symbols())
    top = r0.positions[sym != "H", 2].max()
    fixed, fixinfo = ec.dft_fixed_mask(r0, args.fixed_from) if args.relax_substrate else (None, None)
    h0 = r0.get_scaled_positions()[hi][:2]
    n = args.n
    res = {"provenance": ec.provenance(), "grid_n": n, "grid_origin_frac": h0,
           "protocol": ("H x,y fixed, H z relaxed; substrate "
                        + ("free layers relaxed (DFT-fixed kept fixed)" if args.relax_substrate
                           else "frozen at DFT r0")),
           "fixed_atoms": fixinfo if args.relax_substrate else "all substrate", "models": {}}
    for model in args.models:
        calc = ec.make_calc(model, args.device)
        E = np.zeros((n, n))
        Z = np.zeros((n, n))
        rows = []
        t0 = time.perf_counter()
        for i in range(n):
            for j in range(n):
                frac = (h0 + np.array([i, j]) / n) % 1.0
                a = r0.copy()
                xy = frac @ cell[:2, :2]
                a.positions[hi] = [xy[0], xy[1], top + args.start_height]
                cons = [FixCartesian([hi], mask=(True, True, False))]  # True = fixed (ASE >= 3.23)
                if args.relax_substrate:
                    cons.append(FixAtoms(mask=fixed))
                else:
                    cons.append(FixAtoms(indices=[k for k in range(len(a)) if k != hi]))
                a.set_constraint(cons)
                a.calc = calc
                opt = BFGS(a, maxstep=0.1, logfile=None)
                conv = bool(opt.run(fmax=args.fmax, steps=args.steps))
                e = a.get_potential_energy()
                f = a.get_forces(apply_constraint=False)[hi]
                E[i, j] = e
                Z[i, j] = ec.surface_height(a, a.positions[hi, 2])
                rows.append({"model": model, "i": i, "j": j, "frac_x": frac[0], "frac_y": frac[1],
                             "x_A": xy[0], "y_A": xy[1], "E_eV": e, "height_A": Z[i, j],
                             "H_Fx": f[0], "H_Fy": f[1], "converged": conv, "steps": opt.get_number_of_steps()})
            print(f"  [{model}] row {i + 1}/{n} done ({time.perf_counter() - t0:.0f} s)", flush=True)
        Erel = E - E.min()
        for r in rows:
            r["E_rel_eV"] = r["E_eV"] - E.min()
        mins = local_minima(Erel)
        sites = []
        for k, (i, j) in enumerate(mins):
            frac = (h0 + np.array([i, j]) / n) % 1.0
            xy = frac @ cell[:2, :2]
            cl = ec.classify_site(r0, xy, layers)
            sites.append({"id": k, "i": i, "j": j, "frac": frac, "xy_A": xy, "E_rel_eV": float(Erel[i, j]),
                          "height_A": float(Z[i, j]), "site": cl["label"],
                          "nearest_projected": cl["nearest_projected"],
                          "is_r0_site": bool(i == 0 and j == 0)})
        bars = minimax_barriers(Erel, mins)
        res["models"][model] = {"E_min_eV": float(E.min()), "E_max_rel_eV": float(Erel.max()),
                                "n_not_converged": int(sum(not r["converged"] for r in rows)),
                                "minima": sites,
                                "grid_minimax_barriers_eV": {f"{a}->{b}": v for (a, b), v in bars.items()},
                                "r0_site_E_rel_eV": float(Erel[0, 0]), "r0_site_height_A": float(Z[0, 0])}
        print(f"\n[{model}] {len(mins)} local minima (E relative to map minimum; frozen substrate):")
        for s in sites:
            print(f"   min{s['id']}: E={s['E_rel_eV']:.3f} eV  h={s['height_A']:.2f} A  "
                  f"frac=({s['frac'][0]:.2f},{s['frac'][1]:.2f})  {s['site']}"
                  f"{'  <- r0 site' if s['is_r0_site'] else ''}")
        print(f"   r0 lateral site: E={Erel[0, 0]:.3f} eV, h={Z[0, 0]:.2f} A; map max {Erel.max():.2f} eV")
        for (a, b), v in sorted(bars.items(), key=lambda kv: str(kv[0])):
            print(f"   grid minimax barrier min{a} -> {'own periodic image' if b == 'image' else f'min{b}'}: {v:.3f} eV")

        with (out / f"xy_map_{model}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        # plot over a 2x2 tiling
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        for ax, data, lab, cmap in ((axes[0], Erel, "E - E_min (eV)", "viridis"),
                                    (axes[1], Z, "H height above top atom (A)", "magma")):
            xs, ys, vs = [], [], []
            for ti in range(-1, 3):
                for tj in range(-1, 3):
                    for i in range(n):
                        for j in range(n):
                            fr = h0 + np.array([i + ti * n, j + tj * n]) / n
                            p = fr @ cell[:2, :2]
                            xs.append(p[0]); ys.append(p[1]); vs.append(data[i % n, j % n])  # noqa: E702
            tcf = ax.tricontourf(xs, ys, vs, levels=30, cmap=cmap)
            fig.colorbar(tcf, ax=ax, label=lab)
            for L, mk in zip(layers[:3], ("o", "s", "^")):
                for k in L["idx"]:
                    for ti in range(-1, 3):
                        for tj in range(-1, 3):
                            p = r0.positions[k, :2] + ti * cell[0, :2] + tj * cell[1, :2]
                            ax.plot(*p, mk, mfc="none", mec="w", ms=9 - 2 * (L["layer"] - 1), mew=1.5)
                ax.plot([], [], mk, mfc="none", mec="k", label=f"L{L['layer']} {L['species']}")
            for s in sites:
                for ti in (0, 1):
                    for tj in (0, 1):
                        p = s["xy_A"] + ti * cell[0, :2] + tj * cell[1, :2]
                        ax.plot(*p, "r*", ms=11)
                        ax.annotate(f"min{s['id']}", p, color="r", xytext=(4, 4), textcoords="offset points")
            p = r0.positions[hi, :2]
            ax.plot(*p, "cx", ms=12, mew=2, label="DFT r0 H")
            ax.set(xlim=(0, 2 * cell[0, 0]),
                   ylim=(0, 2 * cell[1, 1]), aspect="equal", xlabel="x (A)", ylabel="y (A)")
            ax.legend(loc="upper right", fontsize=7)
        fig.suptitle(f"{model}: H lateral map, {res['protocol']} — MLIP estimate")
        fig.tight_layout()
        fig.savefig(out / f"xy_map_{model}.png", dpi=160)
        plt.close(fig)
    ec.write_json(out / "xy_map_summary.json", res)
    print(f"\nWrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["z", "xy"])
    ap.add_argument("--models", nargs="+", default=["M0", "M1"], choices=["M0", "M1"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hmin", type=float, default=0.55)
    ap.add_argument("--hmax", type=float, default=2.6)
    ap.add_argument("--dh", type=float, default=0.02)
    ap.add_argument("--n", type=int, default=12, help="xy grid points per cell edge")
    ap.add_argument("--start-height", type=float, default=1.8, help="initial H height for each xy point (A)")
    ap.add_argument("--fmax", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--relax-substrate", action="store_true")
    ap.add_argument("--fixed-from", default="auto")
    args = ap.parse_args()

    r0 = ec.get_label(ec.load_zscan(), "r0")
    r0.calc = None
    hi = ec.h_index(r0)
    out = ec.out_dir(f"pes_{args.mode}" + ("_relaxsub" if args.relax_substrate else ""))
    (run_z if args.mode == "z" else run_xy)(args, r0, hi, out)


if __name__ == "__main__":
    main()
