"""T3-03 (1/3): sensitivity band of the exact lattice D(T) for one H on the Nb-terminated NbP slab.

For every model (M1, M0) and rate variant, D_x(T) and D_y(T) are computed exactly from the Bloch
master-equation generator (01_rates/exact_diffusion.exact_D) on the T3-01 temperature grid
(200-600 K, 10 K steps), with Arrhenius fits (Ea_eff, D0) over 200-600 K and the local Ea at 300 K.

Variants (T3-00 scope section 6):
  classical, quantum_harmonic, classical_wigner, quantum_wigner, fixed_1e13   as in rates.json (T3-01)
  classical_saddles-0.05 / +0.05   every saddle (S1, S2) shifted by -/+0.05 eV, minima fixed
                                   (forward and reverse barriers move together: detailed balance holds)
  fixed_1e13_saddles-0.05 / +0.05  the same with nu = 1e13 s^-1
  classical_direct          direct B1<->B2 hop added: 4 diagonal events per bridge, d = (+-a/2, +-a/2).
                            Saddle = maximum of the frozen-substrate T1-01 xy map along the straight
                            B1->B2 line (an upper estimate), nu = 1e13 s^-1, reverse rate from detailed
                            balance with the classical site weights.
  classical_direct_upper    same route with the saddle lowered by 0.05 eV and nu = Vineyard prefactor of
                            B1->T: a deliberately generous bound on how much the route could matter.
Bands (min/max over the members, per model, axis and T):
  barrier            classical, classical_saddles+-0.05                  (shaded band in the figure)
  barrier_prefactor  barrier + fixed_1e13 with saddles 0, +-0.05          (requested envelope)
  all_variants       every variant above (incl. quantum and direct route)
Checks: detailed balance for every hop pair (incl. the direct route) at every T; Bloch D vs finite
differences of the eigenvalue; stationary vector = Boltzmann populations; zero drift and D_xy; Bloch =
closed form (02_kmc/kmc.closed_form_D) wherever the star topology holds; the five T3-01 variants
reproduce exact_D.json.

Writes <out>/ (default 03_results/<date>_sensitivity/):
  D_variants.csv   model, variant, family, T, D_x, D_y, D_x/D_y, site populations
  fits.csv         per model x variant: D(300 K), D_x/D_y, Ea_eff, D0, local Ea(300 K), factors vs main
  band.csv         per model x band x T: main, lower and upper envelope (and which variant) for x and y
  results.json     provenance, direct-route parameters, variant specs, D tables, fits, bands, drivers,
                   direct-route effect, checks

All numbers are MLIP-based harmonic-TST estimates for a lattice model parameterised with 1 ML collective
barriers (not DFT-validated). The +-0.05 eV band is a judgment-based range, not a statistical error bar.
Needs NumPy/SciPy (via 01_rates/rates.py). Setup and run order: RUNBOOK section 9.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRACK = HERE.parent
for _sub in ("01_rates", "02_kmc"):
    _p = str(TRACK / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
import rates as R  # noqa: E402  (rate formulas, constants, site weights, portable paths)
import exact_diffusion as X  # noqa: E402  (Bloch solver, Arrhenius fit)
import kmc as K  # noqa: E402  (closed form, input discovery)

MODELS = ["M1", "M0"]
MAIN = "classical"
DELTA = 0.05                      # eV, judgment-based saddle uncertainty (scope section 6)
GRID_N = 12                       # T1-01 xy map is 12 x 12
XY_MAP_REL = "artifacts/neb/01_endpoints/pes_xy/xy_map_{model}.csv"  # committed Track 1 map
IDX = {p: n for n, p in enumerate(R.MINIMA)}          # B1, T, B2 (exact_diffusion order)
REPORT_T = (200.0, 300.0, 400.0, 500.0, 600.0)
TOL = {"fd_rel": 1e-5, "stationary_vs_boltzmann_rel": 1e-9, "drift_over_rate": 1e-12,
       "Dxy_over_D": 1e-9, "detailed_balance_rel": 1e-10, "closed_form_rel": 1e-9,
       "reproduce_exact_D_json_rel": 1e-9, "rates_vs_table_rel": 1e-12}


def _v(base, delta=0.0, direct=None, family="", label=""):
    return {"base": base, "delta_saddle_eV": delta, "direct": direct, "family": family, "label": label}


VARIANTS = {
    "classical": _v("classical", family="main", label="classical Vineyard hTST (main)"),
    "quantum_harmonic": _v("quantum_harmonic", family="quantum", label="quantum harmonic hTST (ZPE)"),
    "classical_wigner": _v("classical_wigner", family="quantum", label="classical x Wigner"),
    "quantum_wigner": _v("quantum_wigner", family="quantum", label="quantum harmonic x Wigner"),
    "fixed_1e13": _v("fixed_1e13", family="prefactor", label="nu = 1e13 s^-1"),
    "classical_saddles-0.05": _v("classical", -DELTA, family="barrier", label="classical, saddles -0.05 eV"),
    "classical_saddles+0.05": _v("classical", +DELTA, family="barrier", label="classical, saddles +0.05 eV"),
    "fixed_1e13_saddles-0.05": _v("fixed_1e13", -DELTA, family="barrier_x_prefactor",
                                  label="nu = 1e13, saddles -0.05 eV"),
    "fixed_1e13_saddles+0.05": _v("fixed_1e13", +DELTA, family="barrier_x_prefactor",
                                  label="nu = 1e13, saddles +0.05 eV"),
    "classical_direct": _v("classical", direct="nominal", family="network",
                           label="classical + direct B1<->B2 (grid saddle, nu 1e13)"),
    "classical_direct_upper": _v("classical", direct="upper", family="network",
                                 label="classical + direct B1<->B2 (grid saddle - 0.05 eV, Vineyard nu)"),
}
BASE_VARIANTS = ["classical", "quantum_harmonic", "classical_wigner", "quantum_wigner", "fixed_1e13"]
BANDS = {
    "barrier": ["classical", "classical_saddles-0.05", "classical_saddles+0.05"],
    "barrier_prefactor": ["classical", "classical_saddles-0.05", "classical_saddles+0.05",
                          "fixed_1e13", "fixed_1e13_saddles-0.05", "fixed_1e13_saddles+0.05"],
    "all_variants": list(VARIANTS),
}
# Direct route: B1 at (0, 1/2) and B2 at (1/2, 0) in the cell; each bridge has 4 diagonal neighbours of
# the other kind at (+-a/2, +-a/2) (distance a/sqrt 2 = 2.37 A).
DIRECT_EVENTS = ([{"from": "B1", "to": "B2", "hop": "B1->B2", "d_over_a": [sx * 0.5, sy * 0.5]}
                  for sx in (1, -1) for sy in (1, -1)]
                 + [{"from": "B2", "to": "B1", "hop": "B2->B1", "d_over_a": [sx * 0.5, sy * 0.5]}
                    for sx in (1, -1) for sy in (1, -1)])
DIRECT_PAIR = ("B1->B2", "B2->B1")
HOP_ENDS = {h: (f, t) for h, (f, t, _s) in R.HOPS.items()}
HOP_ENDS.update({"B1->B2": ("B1", "B2"), "B2->B1": ("B2", "B1")})


# ----------------------------------------------------------------------------- inputs
def direct_route(root, models=MODELS):
    """Straight-line B1 -> B2 maximum of the T1-01 frozen-substrate map (H z relaxed), per model."""
    out = {}
    for m in models:
        p = Path(root) / XY_MAP_REL.format(model=m)
        if not p.exists():
            raise FileNotFoundError(f"T1-01 xy map not found: {R.portable(p, root)}")
        with open(p) as fh:
            E = {(int(r["i"]), int(r["j"])): float(r["E_rel_eV"]) for r in csv.DictReader(fh)}
        n, h = GRID_N, GRID_N // 2
        i0, j0 = min(E, key=E.get)                     # B1 = global minimum of the map (DFT r0 site)
        lines = {"+": [E[((i0 + s) % n, (j0 + s) % n)] for s in range(h + 1)],
                 "-": [E[((i0 + s) % n, (j0 - s) % n)] for s in range(h + 1)]}
        if abs(E[(i0, j0)]) > 1e-9 or max(lines["+"][-1], lines["-"][-1]) > 0.03:
            raise ValueError(f"{m}: unexpected map layout (B1 not at E=0 or B2 not a low point)")
        out[m] = {"file": R.portable(p, root), "sha256": R.sha256(p), "B1_grid_ij": [i0, j0],
                  "B2_grid_E_eV": lines["+"][-1], "line_profiles_eV": lines,
                  "E_S3_grid_eV": float(min(max(v) for v in lines.values())),
                  "protocol": "frozen DFT substrate, H z relaxed, 0.28 A grid; straight-line maximum = upper "
                              "estimate of the direct-route saddle (T1-01: freeing the substrate lowered the "
                              "S1 grid estimate by ~0.03 eV)"}
    return out


# ----------------------------------------------------------------------------- rates for a variant
def rate_set(md, spec, T, direct=None):
    """{hop: k (s^-1)} for one model block of rates.json, variant spec and temperature."""
    v, dE = spec["base"], spec["delta_saddle_eV"]
    k = {}
    for h, hp in md["hops"].items():
        q = dict(hp)
        q["Ea_eV"] = hp["Ea_eV"] + dE                  # saddle shift: both directions of the pair move
        k[h] = R.rate(q, T, v)
    if spec["direct"]:
        if direct is None:
            raise ValueError("direct-route parameters missing")
        sites = md["sites"]
        e3 = direct["E_S3_grid_eV"] + dE - (DELTA if spec["direct"] == "upper" else 0.0)
        nu = R.NU_FIXED if spec["direct"] == "nominal" else md["hops"]["B1->T"]["nu_vineyard_s-1"]
        kf = nu * np.exp(-(e3 - sites["B1"]["E_eV"]) / (R.KB_EV * T))
        kb = kf * np.exp(R.ln_site_weight(sites["B1"], T, v) - R.ln_site_weight(sites["B2"], T, v))
        k["B1->B2"], k["B2->B1"] = float(kf), float(kb)
    return k


def events_for(spec, base_events):
    return list(base_events) + (DIRECT_EVENTS if spec["direct"] else [])


def exact_point(md, spec, T, base_events, a_A, direct=None):
    """Exact D (cm^2/s), Boltzmann populations, rates and per-point checks for one variant at T."""
    k = rate_set(md, spec, T, direct)
    ev = events_for(spec, base_events)
    D, dg = X.exact_D(ev, IDX, k, a_A * 1e-8)
    pop = R.populations(md["sites"], T, spec["base"])
    pB = np.array([pop[p] for p in R.MINIMA])
    chk = {"fd_rel": max(dg["fd_rel_x"], dg["fd_rel_y"]),
           "stationary_vs_boltzmann_rel": float(np.max(abs(dg["p"] / pB - 1))),
           "drift_over_rate": max(dg["drift_x"], dg["drift_y"]) / max(k.values()),
           "Dxy_over_D": abs(D["xy"]) / min(D["x"], D["y"])}
    pairs = list(R.PAIRS) + ([DIRECT_PAIR] if spec["direct"] else [])
    db = 0.0
    for f, r in pairs:
        i, j = HOP_ENDS[f]
        lhs = np.log(k[f]) + R.ln_site_weight(md["sites"][i], T, spec["base"])
        rhs = np.log(k[r]) + R.ln_site_weight(md["sites"][j], T, spec["base"])
        db = max(db, abs(float(np.expm1(lhs - rhs))))
    chk["detailed_balance_rel"] = db
    if not spec["direct"]:
        cf = K.closed_form_D(k, a_A)
        chk["closed_form_rel"] = max(abs(D["x"] / cf[0] - 1), abs(D["y"] / cf[1] - 1))
    return {"D_x": float(D["x"]), "D_y": float(D["y"]), "D_xy": float(D["xy"]), "pop": pop, "k": k,
            "checks": chk}


# ----------------------------------------------------------------------------- analysis
def compute(rd, direct, temps, fit_window=(200.0, 600.0)):
    a_A, ev = rd["lattice"]["a_A"], rd["lattice"]["events"]
    table, worst = {}, {k: 0.0 for k in TOL if k not in ("reproduce_exact_D_json_rel", "rates_vs_table_rel")}
    fits = {}
    win = (temps >= fit_window[0] - 1e-9) & (temps <= fit_window[1] + 1e-9)
    for m in MODELS:
        md = rd["models"][m]
        table[m], fits[m] = {}, {}
        for name, spec in VARIANTS.items():
            pts = [exact_point(md, spec, T, ev, a_A, direct.get(m)) for T in temps]
            for p in pts:
                for c, val in p["checks"].items():
                    worst[c] = max(worst[c], val)
            table[m][name] = {"x": [p["D_x"] for p in pts], "y": [p["D_y"] for p in pts],
                              **{f"p_{s}": [p["pop"][s] for p in pts] for s in R.MINIMA}}
            fits[m][name] = {}
            for ax in ("x", "y"):
                f = X.arrhenius(temps[win], np.array(table[m][name][ax])[win])
                f["Ea_local_300K_eV"] = X.local_Ea(
                    lambda T, _ax=ax, _s=spec: exact_point(md, _s, T, ev, a_A, direct.get(m))[f"D_{_ax}"], 300.0)
                f["D_300K_cm2_s"] = float(np.interp(300.0, temps, table[m][name][ax]))
                fits[m][name][ax] = f
            fits[m][name]["Dx_over_Dy_300K"] = fits[m][name]["x"]["D_300K_cm2_s"] / fits[m][name]["y"]["D_300K_cm2_s"]
    return table, fits, worst


def at(temps, arr, T):
    return float(np.asarray(arr)[int(np.argmin(abs(temps - T)))])


def bands(table, fits, temps):
    out = {}
    for m in table:
        out[m] = {}
        for b, members in BANDS.items():
            out[m][b] = {"members": members}
            for ax in ("x", "y"):
                arr = np.array([table[m][v][ax] for v in members])
                lo, hi = arr.min(0), arr.max(0)
                main = np.array(table[m][MAIN][ax])
                ea = [fits[m][v][ax]["Ea_eff_eV"] for v in members]
                d0 = [fits[m][v][ax]["D0_cm2_s"] for v in members]
                out[m][b][ax] = {
                    "lower": lo.tolist(), "upper": hi.tolist(),
                    "lower_variant": [members[i] for i in arr.argmin(0)],
                    "upper_variant": [members[i] for i in arr.argmax(0)],
                    "at_T": {f"{T:g}": {"main": at(temps, main, T), "lower": at(temps, lo, T), "upper": at(temps, hi, T),
                                        "factor_down": at(temps, main / lo, T), "factor_up": at(temps, hi / main, T),
                                        "span_upper_over_lower": at(temps, hi / lo, T)} for T in REPORT_T},
                    "over_200_600K": {"max_factor_down": float((main / lo).max()), "max_factor_up": float((hi / main).max()),
                                      "min_D": float(lo.min()), "max_D": float(hi.max()),
                                      "Ea_eff_range_eV": [min(ea), max(ea)], "D0_range_cm2_s": [min(d0), max(d0)]},
                }
    return out


def drivers(table, temps):
    """D(variant)/D(main) per model, axis and report temperature, plus M0/M1 (classical)."""
    out = {}
    for m in table:
        out[m] = {v: {ax: {f"{T:g}": at(temps, np.array(table[m][v][ax]) / np.array(table[m][MAIN][ax]), T)
                           for T in REPORT_T} for ax in ("x", "y")} for v in VARIANTS}
    if "M0" in table and "M1" in table:
        out["M0_over_M1_classical"] = {ax: {f"{T:g}": at(temps, np.array(table["M0"][MAIN][ax])
                                                          / np.array(table["M1"][MAIN][ax]), T)
                                            for T in REPORT_T} for ax in ("x", "y")}
    return out


def direct_effect(table, temps):
    out = {}
    for m in table:
        out[m] = {}
        for v in ("classical_direct", "classical_direct_upper"):
            r = {}
            for ax in ("x", "y"):
                rel = np.array(table[m][v][ax]) / np.array(table[m][MAIN][ax]) - 1.0
                i = int(np.argmax(rel))
                r[ax] = {"max_rel_change": float(rel[i]), "at_T_K": float(temps[i]),
                         "rel_change_at_T": {f"{T:g}": at(temps, rel, T) for T in REPORT_T}}
            mx = max(r["x"]["max_rel_change"], r["y"]["max_rel_change"])
            r["verdict"] = ("negligible (< 1 %)" if mx < 0.01 else "small (1-10 %)" if mx < 0.10
                            else "significant (> 10 %)")
            out[m][v] = r
    return out


# ----------------------------------------------------------------------------- main
def main(argv=None):
    stamp = datetime.date.today().isoformat()
    root, root_src = R.portfolio_root(HERE)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rates", help="rates.json (default: newest 01_rates/<date>_rates/rates.json)")
    ap.add_argument("--exact", help="exact_D.json (default: newest 01_rates/<date>_exact_D/exact_D.json)")
    ap.add_argument("--out", default=str(HERE / f"{stamp}_sensitivity"))
    ap.add_argument("--fit-tmin", type=float, default=200.0)
    ap.add_argument("--fit-tmax", type=float, default=600.0)
    args = ap.parse_args(argv)

    r_def, e_def = K.default_inputs()
    rpath = Path(args.rates) if args.rates else r_def
    epath = Path(args.exact) if args.exact else e_def
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rd = json.loads(rpath.read_text())
    ed = json.loads(epath.read_text())
    temps = np.asarray(rd["T_K"], float)
    direct = direct_route(root)
    warnings = list(rd.get("warnings", [])) + list(ed.get("warnings", []))

    table, fits, worst = compute(rd, direct, temps, (args.fit_tmin, args.fit_tmax))

    # ---- reproduce T3-01: rates vs table, and D of the five base variants vs exact_D.json
    dev_k, dev_D = 0.0, 0.0
    for m in MODELS:
        md = rd["models"][m]
        for v in BASE_VARIANTS:
            for n, T in enumerate(temps):
                k = rate_set(md, VARIANTS[v], T)
                dev_k = max(dev_k, max(abs(k[h] / md["rates_s-1"][v][h][n] - 1) for h in R.HOPS))
            for ax in ("x", "y"):
                dev_D = max(dev_D, float(np.max(abs(np.array(table[m][v][ax]) / np.array(ed["D_cm2_s"][m][v][ax]) - 1))))
    worst["rates_vs_table_rel"] = dev_k
    worst["reproduce_exact_D_json_rel"] = dev_D
    for c, val in worst.items():
        if val > TOL[c]:
            warnings.append(f"check {c} = {val:.1e} exceeds {TOL[c]:.0e}")

    bd = bands(table, fits, temps)
    drv = drivers(table, temps)
    deff = direct_effect(table, temps)

    # ---- write CSV tables
    with open(out / "D_variants.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "variant", "family", "T_K", "D_x_cm2_s", "D_y_cm2_s", "Dx_over_Dy", "p_B1", "p_T", "p_B2"])
        for m in MODELS:
            for v, spec in VARIANTS.items():
                t = table[m][v]
                for n, T in enumerate(temps):
                    w.writerow([m, v, spec["family"], f"{T:g}", f"{t['x'][n]:.6e}", f"{t['y'][n]:.6e}",
                                f"{t['x'][n] / t['y'][n]:.5f}", f"{t['p_B1'][n]:.6e}", f"{t['p_T'][n]:.6e}",
                                f"{t['p_B2'][n]:.6e}"])
    with open(out / "fits.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "variant", "family", "D_x_300K", "D_y_300K", "Dx_over_Dy_300K", "Ea_eff_x_eV", "Ea_eff_y_eV",
                    "D0_x_cm2_s", "D0_y_cm2_s", "Ea_local_300K_x_eV", "Ea_local_300K_y_eV", "max_resid_lnD_x",
                    "max_resid_lnD_y"] + [f"factor_vs_main_{ax}_{T:g}K" for ax in ("x", "y") for T in (200, 300, 600)])
        for m in MODELS:
            for v, spec in VARIANTS.items():
                f = fits[m][v]
                w.writerow([m, v, spec["family"], f"{f['x']['D_300K_cm2_s']:.4e}", f"{f['y']['D_300K_cm2_s']:.4e}",
                            f"{f['Dx_over_Dy_300K']:.4f}", f"{f['x']['Ea_eff_eV']:.4f}", f"{f['y']['Ea_eff_eV']:.4f}",
                            f"{f['x']['D0_cm2_s']:.4e}", f"{f['y']['D0_cm2_s']:.4e}",
                            f"{f['x']['Ea_local_300K_eV']:.4f}", f"{f['y']['Ea_local_300K_eV']:.4f}",
                            f"{f['x']['max_abs_resid_lnD']:.4f}", f"{f['y']['max_abs_resid_lnD']:.4f}"]
                           + [f"{drv[m][v][ax][f'{T:g}']:.4f}" for ax in ("x", "y") for T in (200, 300, 600)])
    with open(out / "band.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "band", "T_K", "D_x_main", "D_x_lower", "D_x_upper", "D_x_lower_variant", "D_x_upper_variant",
                    "D_y_main", "D_y_lower", "D_y_upper", "D_y_lower_variant", "D_y_upper_variant"])
        for m in MODELS:
            for b in BANDS:
                for n, T in enumerate(temps):
                    row = [m, b, f"{T:g}"]
                    for ax in ("x", "y"):
                        q = bd[m][b][ax]
                        row += [f"{table[m][MAIN][ax][n]:.6e}", f"{q['lower'][n]:.6e}", f"{q['upper'][n]:.6e}",
                                q["lower_variant"][n], q["upper_variant"][n]]
                    w.writerow(row)

    # ---- JSON
    import scipy
    import matplotlib
    prov = {"date": stamp, "script": "src/kmc/03_results/sensitivity.py", "root": root_src,
            "input": {"rates": {"file": R.portable(rpath, root), "sha256": R.sha256(rpath)},
                      "exact_D": {"file": R.portable(epath, root), "sha256": R.sha256(epath)},
                      "xy_maps": {m: {"file": d["file"], "sha256": d["sha256"]} for m, d in direct.items()}},
            "upstream": rd["provenance"],
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                         "matplotlib": matplotlib.__version__},
            "fit_window_K": [args.fit_tmin, args.fit_tmax], "delta_saddle_eV": DELTA,
            "note": ("MLIP-based harmonic-TST estimates; lattice model parameterised with 1 ML collective barriers "
                     "(single walker, no H-H interactions); not DFT-validated. The +-0.05 eV band is a judgment-based "
                     "range, not a statistical error bar. Valid only above ~150-200 K (tunnelling crossover 85-101 K).")}
    doc = {"provenance": prov, "warnings": warnings,
           "checks": {"max_over_all_points": worst, "tolerances": TOL},
           "variants": VARIANTS, "bands_def": BANDS, "direct_route": direct, "direct_events": DIRECT_EVENTS,
           "T_K": temps.tolist(), "D_cm2_s": table, "fits": fits, "bands": bd, "drivers": drv,
           "direct_route_effect": deff}
    (out / "results.json").write_text(json.dumps(doc, indent=1))

    # ---- console
    print(f"rates : {R.portable(rpath, root)}\nexact : {R.portable(epath, root)}")
    for m, d in direct.items():
        print(f"direct B1->B2 straight-line grid maximum, {m}: {d['E_S3_grid_eV']:.3f} eV")
    print(f"\nexact D (cm^2/s), Arrhenius {args.fit_tmin:g}-{args.fit_tmax:g} K")
    print(f"{'model':5} {'variant':25} {'Dx(300)':>10} {'Dy(300)':>10} {'Dx/Dy':>6} {'Ea_x':>7} {'Ea_y':>7} {'D0_x':>9} {'D0_y':>9}")
    for m in MODELS:
        for v in VARIANTS:
            f = fits[m][v]
            print(f"{m:5} {v:25} {f['x']['D_300K_cm2_s']:10.3e} {f['y']['D_300K_cm2_s']:10.3e} {f['Dx_over_Dy_300K']:6.2f} "
                  f"{f['x']['Ea_eff_eV']:7.4f} {f['y']['Ea_eff_eV']:7.4f} {f['x']['D0_cm2_s']:9.2e} {f['y']['D0_cm2_s']:9.2e}")
    print("\nband at 300 K: D_lower - D_main - D_upper (cm^2/s)  [factor down / up]")
    for m in MODELS:
        for b in BANDS:
            for ax in ("x", "y"):
                q = bd[m][b][ax]["at_T"]["300"]
                print(f"  {m} {b:18} D_{ax}: {q['lower']:.2e} - {q['main']:.2e} - {q['upper']:.2e}   "
                      f"[/{q['factor_down']:.1f}, x{q['factor_up']:.1f}]")
    print("\ndirect B1<->B2 route, max relative change of D over 200-600 K:")
    for m in MODELS:
        for v, r in deff[m].items():
            print(f"  {m} {v:24} x {100 * r['x']['max_rel_change']:7.3f} %  y {100 * r['y']['max_rel_change']:7.3f} %  "
                  f"(at {r['y']['at_T_K']:g} K) -> {r['verdict']}")
    print("\nchecks (max over all models/variants/T):")
    for c, val in worst.items():
        print(f"  {c:30} {val:.1e}   (tol {TOL[c]:.0e})")
    print("\nWARNINGS:\n  " + "\n  ".join(warnings) if warnings else "\nno warnings")
    print(f"\noutputs -> {R.portable(out, root)}: D_variants.csv, fits.csv, band.csv, results.json")
    return doc


if __name__ == "__main__":
    main()
