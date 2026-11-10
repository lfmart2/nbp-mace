"""T3-01: exact tracer diffusion coefficient of one H on the periodic B1/T/B2 site network.

Reads rates.json (from rates.py) and, for every model, rate variant and temperature, builds the
Bloch-transformed master-equation generator M(k) of the periodic network (3 sites per cell, events
and displacements from the rates.json lattice block). The slowest eigenvalue behaves as
lambda_0(k) = -sum_ab D_ab k_a k_b near k = 0, so D_xx, D_yy (and D_xy) follow from its curvature.

Three independent evaluations are compared at every point:
  1. second-order perturbation theory for lambda_0(k) (main result; exact for this network),
  2. finite differences of the numerical eigenvalue at small k (Richardson-extrapolated),
  3. closed form for this network (series "resistor" chain): D_x = (a^2/2) p_T k(T->B2),
     D_y = (a^2/2) p_T k(T->B1), with p_T the Boltzmann population of T.
Also checked: stationary vector of M(0) = Boltzmann populations from rates.json; zero drift; D_xy = 0;
the rates recomputed from the stored hop parameters reproduce the rates.json table.

Writes to <out>/:
  exact_D.csv                 model, variant, T, D_x, D_y (cm^2/s), D_x/D_y, p_T
  exact_D.json                provenance (rates.json SHA-256 + its upstream provenance), Arrhenius fits
                              (Ea_eff, D0 per axis/model/variant), checks
  arrhenius_quicklook.png     quick-look Arrhenius plot (not a publication figure)

These are MLIP-based harmonic-TST estimates for a lattice model parameterised with 1 ML collective
barriers; not DFT-validated values. Outputs contain no absolute local paths. Needs NumPy/SciPy/Matplotlib.
"""
import argparse
import csv
import datetime
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rates as R  # noqa: E402  (same folder: rate formulas, constants, helpers)

AXES = {"x": np.array([1.0, 0.0]), "y": np.array([0.0, 1.0])}
# T3-00 scope section 3, preliminary M1 classical numbers (rounded) and M0 comparison
SCOPE = {"D300_x": 2.0e-6, "D300_y": 0.9e-6, "Ea_x": 0.23, "Ea_y": 0.25, "D0": 1e-2,
         "M0_Ea_x": 0.28, "M0_Ea_y": 0.29, "M0_slowdown": (5.0, 6.0), "aniso300": 2.3, "window": (150.0, 500.0)}


# ----------------------------------------------------------------------------- master equation
def generator(events, idx, rates_by_hop, kvec):
    """M(k)[to, from] = sum_events w exp(-i k.d); diagonal = -total escape rate. d in units of a."""
    n = len(idx)
    M = np.zeros((n, n), complex)
    for e in events:
        i, j = idx[e["from"]], idx[e["to"]]
        w = rates_by_hop[e["hop"]]
        M[j, i] += w * np.exp(-1j * float(np.dot(kvec, e["d_over_a"])))
        M[i, i] -= w
    return M


def derivatives(events, idx, rates_by_hop, n_hat):
    """dM/dk and (1/2) d^2M/dk^2 along unit vector n_hat at k = 0."""
    n = len(idx)
    A = np.zeros((n, n), complex)
    B = np.zeros((n, n), complex)
    for e in events:
        i, j = idx[e["from"]], idx[e["to"]]
        w, dn = rates_by_hop[e["hop"]], float(np.dot(n_hat, e["d_over_a"]))
        A[j, i] += -1j * dn * w
        B[j, i] += -0.5 * dn * dn * w
    return A, B


def stationary(M0):
    """Right null vector of M(0), normalised to sum 1."""
    w, v = np.linalg.eig(M0.real)
    p = np.abs(v[:, np.argmin(abs(w))].real)
    return p / p.sum()


def D_perturbation(events, idx, rates_by_hop, n_hat):
    """-lambda_2 along n_hat (units a^2 * rate). Rayleigh-Schroedinger to second order, non-Hermitian M."""
    M0 = generator(events, idx, rates_by_hop, np.zeros(2))
    Rv = stationary(M0).astype(complex)
    L = np.ones(len(idx))                     # left null vector (probability conservation)
    A, B = derivatives(events, idx, rates_by_hop, n_hat)
    lam1 = L @ A @ Rv
    rhs = -(A @ Rv - lam1 * Rv)
    R1 = np.linalg.lstsq(M0, rhs, rcond=None)[0]
    R1 -= (L @ R1) * Rv
    lam2 = L @ B @ Rv + L @ A @ R1
    return float(-lam2.real), complex(lam1), Rv.real


def D_finite_difference(events, idx, rates_by_hop, n_hat, h=0.05):
    """Curvature of the numerical slowest eigenvalue, Richardson h -> h/2."""
    def f(hh):
        lam = np.linalg.eigvals(generator(events, idx, rates_by_hop, hh * n_hat))
        return -lam[np.argmax(lam.real)].real / hh ** 2
    return (4.0 * f(h / 2) - f(h)) / 3.0


def exact_D(events, idx, rates_by_hop, a_cm):
    """D_x, D_y, D_xy (cm^2/s) and the diagnostics for one rate set. Rates rescaled to O(1) for precision."""
    s = max(rates_by_hop.values())
    rs = {h: k / s for h, k in rates_by_hop.items()}
    out, diag = {}, {}
    for ax, n_hat in AXES.items():
        d, lam1, p = D_perturbation(events, idx, rs, n_hat)
        out[ax] = d * s * a_cm ** 2
        diag[f"fd_rel_{ax}"] = abs(D_finite_difference(events, idx, rs, n_hat) / d - 1.0)
        diag[f"drift_{ax}"] = abs(lam1) * s
    d_diag, _, _ = D_perturbation(events, idx, rs, np.array([1.0, 1.0]) / np.sqrt(2.0))
    out["xy"] = (d_diag * s * a_cm ** 2) - 0.5 * (out["x"] + out["y"])
    diag["p"] = p
    return out, diag


# ----------------------------------------------------------------------------- fits
def arrhenius(T, D):
    """ln D = ln D0 - Ea/(kB T); returns Ea (eV), D0 (cm^2/s), max |residual| in ln D."""
    x = 1.0 / (R.KB_EV * np.asarray(T))
    slope, icpt = np.polyfit(x, np.log(D), 1)
    res = np.log(D) - (slope * x + icpt)
    return {"Ea_eff_eV": float(-slope), "D0_cm2_s": float(np.exp(icpt)), "max_abs_resid_lnD": float(abs(res).max())}


def local_Ea(fun, T, dT=1.0):
    """Tangent activation energy -d lnD / d(1/kT) at T."""
    x1, x2 = 1 / (R.KB_EV * (T - dT)), 1 / (R.KB_EV * (T + dT))
    return float(-(np.log(fun(T + dT)) - np.log(fun(T - dT))) / (x2 - x1))


# ----------------------------------------------------------------------------- plot
def quicklook(table, temps, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"M1": "#1f77b4", "M0": "#ff7f0e"}
    vstyle = {"classical": ("-", "classical (main)"), "quantum_harmonic": ("--", "quantum harmonic"),
              "classical_wigner": (":", "classical + Wigner"), "quantum_wigner": ("-.", "quantum + Wigner"),
              "fixed_1e13": ((0, (1, 3)), r"fixed $10^{13}$ s$^{-1}$")}
    x = 1000.0 / np.asarray(temps)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    ax = axs[0]
    for m in table:
        ax.semilogy(x, table[m]["classical"]["x"], "-", color=col[m], lw=2, label=f"{m} $D_x$")
        ax.semilogy(x, table[m]["classical"]["y"], "--", color=col[m], lw=2, label=f"{m} $D_y$")
    ax.set_title("(a) classical Vineyard hTST, M1 vs M0", fontsize=10)
    ax = axs[1]
    for v, (ls, lab) in vstyle.items():
        ax.semilogy(x, table["M1"][v]["x"], linestyle=ls, color="#1f77b4", lw=1.6, label=lab)
    ax.set_title("(b) M1 $D_x$: rate-theory variants", fontsize=10)
    for ax in axs:
        ax.set_xlabel("1000 / T (K$^{-1}$)")
        ax.set_ylabel("D (cm$^2$ s$^{-1}$)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, frameon=False)
        top = ax.secondary_xaxis("top", functions=(lambda v: 1000.0 / np.maximum(v, 1e-9),
                                                   lambda v: 1000.0 / np.maximum(v, 1e-9)))
        top.set_xticks([600, 400, 300, 250, 200])
        top.set_xlabel("T (K)")
    fig.suptitle("Quick look: exact lattice D(T) for H on NbP, MLIP-based hTST (1 ML barriers; not DFT-validated)",
                 fontsize=9)
    fig.savefig(png, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    here = Path(__file__).resolve().parent
    stamp = datetime.date.today().isoformat()
    root, root_src = R.portfolio_root(here)
    found = sorted(here.glob("*_rates/rates.json"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rates", default=str(found[-1]) if found else None,
                    help="rates.json (default: newest <date>_rates/rates.json next to this script)")
    ap.add_argument("--out", default=str(here / f"{stamp}_exact_D"))
    ap.add_argument("--fit-tmin", type=float, default=200.0)
    ap.add_argument("--fit-tmax", type=float, default=600.0)
    args = ap.parse_args()
    if not args.rates or not Path(args.rates).exists():
        sys.exit("rates.json not found; run  python rates.py  first")

    rpath, out = Path(args.rates), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(rpath) as fh:
        rd = json.load(fh)
    temps = np.array(rd["T_K"])
    lat = rd["lattice"]
    a_cm = lat["a_A"] * 1e-8
    events = lat["events"]
    idx = {p: n for n, p in enumerate(R.MINIMA)}
    warnings, checks = list(rd.get("warnings", [])), {}

    def rates_at(m, v, T):
        return {h: R.rate(hp, T, v) for h, hp in rd["models"][m]["hops"].items()}

    def D_at(m, v, T):
        return exact_D(events, idx, rates_at(m, v, T), a_cm)[0]

    # ---- rates recomputed from stored parameters must equal the table
    dev = 0.0
    for m, md in rd["models"].items():
        for v in rd["variants"]:
            for h, ks in md["rates_s-1"][v].items():
                rc = np.array([R.rate(md["hops"][h], T, v) for T in temps])
                dev = max(dev, float(np.max(abs(rc / np.array(ks) - 1))))
    checks["rates_recomputed_vs_table_max_rel_dev"] = dev
    if dev > 1e-12:
        warnings.append(f"rates recomputed from parameters differ from rates.json table ({dev:.1e})")

    # ---- D(T) for every model / variant
    table, rows = {}, []
    worst = {"fd_rel": 0.0, "closed_form_rel": 0.0, "stationary_vs_boltzmann_rel": 0.0,
             "drift_over_rate": 0.0, "Dxy_over_D": 0.0}
    for m, md in rd["models"].items():
        table[m] = {}
        for v in rd["variants"]:
            Dx, Dy = [], []
            for n, T in enumerate(temps):
                kr = {h: md["rates_s-1"][v][h][n] for h in md["hops"]}
                D, dg = exact_D(events, idx, kr, a_cm)
                pB = np.array([md["populations"][v][p][n] for p in R.MINIMA])
                pT = pB[idx["T"]]
                cf = {"x": 0.5 * a_cm ** 2 * pT * kr["T->B2"], "y": 0.5 * a_cm ** 2 * pT * kr["T->B1"]}
                worst["fd_rel"] = max(worst["fd_rel"], dg["fd_rel_x"], dg["fd_rel_y"])
                worst["closed_form_rel"] = max(worst["closed_form_rel"],
                                               *(abs(D[a] / cf[a] - 1) for a in ("x", "y")))
                worst["stationary_vs_boltzmann_rel"] = max(worst["stationary_vs_boltzmann_rel"],
                                                           float(np.max(abs(dg["p"] / pB - 1))))
                worst["drift_over_rate"] = max(worst["drift_over_rate"],
                                               max(dg["drift_x"], dg["drift_y"]) / max(kr.values()))
                worst["Dxy_over_D"] = max(worst["Dxy_over_D"], abs(D["xy"]) / min(D["x"], D["y"]))
                Dx.append(D["x"]); Dy.append(D["y"])
                rows.append([m, v, f"{T:g}", f"{D['x']:.6e}", f"{D['y']:.6e}", f"{D['x'] / D['y']:.5f}", f"{pT:.6e}"])
            table[m][v] = {"x": Dx, "y": Dy}
    checks["three_way_and_consistency"] = worst
    tol = {"fd_rel": 1e-5, "closed_form_rel": 1e-9, "stationary_vs_boltzmann_rel": 1e-9,
           "drift_over_rate": 1e-12, "Dxy_over_D": 1e-9}
    for k, t in tol.items():
        if worst[k] > t:
            warnings.append(f"check {k} = {worst[k]:.1e} exceeds {t:.0e}")
    checks["tolerances"] = tol

    # ---- Arrhenius fits (main window) + local Ea at 300 K
    win = (temps >= args.fit_tmin - 1e-9) & (temps <= args.fit_tmax + 1e-9)
    fits = {}
    for m in table:
        fits[m] = {}
        for v in table[m]:
            fits[m][v] = {}
            for ax in ("x", "y"):
                f = arrhenius(temps[win], np.array(table[m][v][ax])[win])
                f["Ea_local_300K_eV"] = local_Ea(lambda T: D_at(m, v, T)[ax], 300.0)
                f["D_300K_cm2_s"] = float(np.interp(300.0, temps, table[m][v][ax]))
                fits[m][v][ax] = f

    # ---- T3-00 scope section 3 check (M1 classical), incl. the scope's own 150-500 K fit window
    tw = np.arange(SCOPE["window"][0], SCOPE["window"][1] + 1e-9, 10.0)
    scope_fit = {m: {ax: arrhenius(tw, np.array([D_at(m, "classical", T)[ax] for T in tw]))
                     for ax in ("x", "y")} for m in table}
    c = fits["M1"]["classical"]
    m0 = fits.get("M0", {}).get("classical")
    sc = {"D300_x": [c["x"]["D_300K_cm2_s"], SCOPE["D300_x"]], "D300_y": [c["y"]["D_300K_cm2_s"], SCOPE["D300_y"]],
          "Ea_x_200-600": [c["x"]["Ea_eff_eV"], SCOPE["Ea_x"]], "Ea_y_200-600": [c["y"]["Ea_eff_eV"], SCOPE["Ea_y"]],
          "Ea_x_150-500": [scope_fit["M1"]["x"]["Ea_eff_eV"], SCOPE["Ea_x"]],
          "Ea_y_150-500": [scope_fit["M1"]["y"]["Ea_eff_eV"], SCOPE["Ea_y"]],
          "D0_x_150-500": [scope_fit["M1"]["x"]["D0_cm2_s"], SCOPE["D0"]],
          "D0_y_150-500": [scope_fit["M1"]["y"]["D0_cm2_s"], SCOPE["D0"]],
          "anisotropy_300K": [c["x"]["D_300K_cm2_s"] / c["y"]["D_300K_cm2_s"], SCOPE["aniso300"]]}
    ok = (abs(sc["D300_x"][0] / sc["D300_x"][1] - 1) <= 0.05 and abs(sc["D300_y"][0] / sc["D300_y"][1] - 1) <= 0.06
          and abs(sc["Ea_x_150-500"][0] - SCOPE["Ea_x"]) <= 0.005 and abs(sc["Ea_y_150-500"][0] - SCOPE["Ea_y"]) <= 0.005
          and abs(np.log10(sc["D0_x_150-500"][0] / SCOPE["D0"])) <= 0.5
          and abs(sc["anisotropy_300K"][0] / SCOPE["aniso300"] - 1) <= 0.05)
    if m0:
        slow = [c[a]["D_300K_cm2_s"] / m0[a]["D_300K_cm2_s"] for a in ("x", "y")]
        sc["M0_slowdown_300K_x_y"] = [slow, list(SCOPE["M0_slowdown"])]
        sc["M0_Ea_x_150-500"] = [scope_fit["M0"]["x"]["Ea_eff_eV"], SCOPE["M0_Ea_x"]]
        sc["M0_Ea_y_150-500"] = [scope_fit["M0"]["y"]["Ea_eff_eV"], SCOPE["M0_Ea_y"]]
    checks["scope_section3"] = {"values_[this_run, scope]": sc, "M1_reproduced": bool(ok),
                                "tolerance": "D(300 K) 5-6 %, Ea 5 meV, D0 within x3, anisotropy 5 % (scope values rounded); "
                                             "150-500 K window evaluated only to compare with the scope (below 200 K is "
                                             "outside the stated validity range)"}
    if not ok:
        warnings.append("M1 classical D does not reproduce T3-00 scope section 3 within tolerance")

    # ---- write
    with open(out / "exact_D.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "variant", "T_K", "D_x_cm2_s", "D_y_cm2_s", "Dx_over_Dy", "p_T"])
        w.writerows(rows)
    import matplotlib, scipy
    prov = {"date": stamp, "script": "src/kmc/01_rates/exact_diffusion.py", "root": root_src,
            "input": {"file": R.portable(rpath, root), "sha256": R.sha256(rpath)},
            "upstream": rd["provenance"],
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
            "method": ("curvature at k=0 of the slowest eigenvalue of the Bloch master-equation generator "
                       "(2nd-order perturbation theory), cross-checked by finite differences and the closed form "
                       "D_x = a^2/2 p_T k(T->B2), D_y = a^2/2 p_T k(T->B1)"),
            "fit_window_K": [args.fit_tmin, args.fit_tmax],
            "note": ("MLIP-based harmonic-TST estimates; lattice model parameterised with 1 ML collective barriers "
                     "(single walker, no H-H interactions); not DFT-validated.")}
    with open(out / "exact_D.json", "w") as fh:
        json.dump({"provenance": prov, "warnings": warnings, "checks": checks, "fits": fits,
                   "fits_scope_window_150-500K_classical": scope_fit, "T_K": temps.tolist(), "D_cm2_s": table},
                  fh, indent=2)
    quicklook(table, temps, out / "arrhenius_quicklook.png")

    # ---- console
    print(f"\nexact D (cm^2/s), Arrhenius fit {args.fit_tmin:g}-{args.fit_tmax:g} K")
    print(f"{'model':5} {'variant':17} {'Dx(300)':>10} {'Dy(300)':>10} {'Dx/Dy':>6} "
          f"{'Ea_x':>7} {'Ea_y':>7} {'D0_x':>9} {'D0_y':>9}")
    for m in fits:
        for v, f in fits[m].items():
            print(f"{m:5} {v:17} {f['x']['D_300K_cm2_s']:10.3e} {f['y']['D_300K_cm2_s']:10.3e} "
                  f"{f['x']['D_300K_cm2_s'] / f['y']['D_300K_cm2_s']:6.2f} {f['x']['Ea_eff_eV']:7.4f} "
                  f"{f['y']['Ea_eff_eV']:7.4f} {f['x']['D0_cm2_s']:9.2e} {f['y']['D0_cm2_s']:9.2e}")
    print("\nchecks (max over all models/variants/T):")
    for k, val in worst.items():
        print(f"  {k:30} {val:.1e}   (tol {tol[k]:.0e})")
    print(f"\nscope section 3 (M1 classical) reproduced: {ok}")
    for k, (a_, b_) in sc.items():
        print(f"  {k:24} this run {a_}   scope {b_}" if isinstance(a_, list) else
              f"  {k:24} this run {a_:.4g}   scope {b_:.4g}")
    print("\nWARNINGS:\n  " + "\n  ".join(warnings) if warnings else "\nno warnings")
    print(f"\noutputs -> {out.name}\\exact_D.csv, exact_D.json, arrhenius_quicklook.png")


if __name__ == "__main__":
    main()
