"""T3-01: harmonic-TST rate catalogue for H on NbP (B1 <-> T <-> B2) from the MLIP NEB key points.

Reads the T1-03 key points (energies + H-only modes at B1, S1, T, S2, B2 for M1 and M0) and writes

  <out>/rates.csv    long table: model, variant, hop, T, Ea, effective prefactor, Wigner kappa, k
  <out>/rates.json   provenance, lattice/event table (for the T3-02 kMC), per-hop parameters,
                     k(T) per variant, Boltzmann site populations, and all checks

Hops (both models): B1->T and T->B1 via S1 (y), T->B2 and B2->T via S2 (x).
Rate variants (scope section 6):
  classical         Vineyard hTST, nu = prod(nu_min, 3 modes) / prod(nu_saddle, 2 real modes)  [MAIN]
  quantum_harmonic  hTST with quantum harmonic partition functions (ZPE included; -> classical at high T)
  classical_wigner  classical x Wigner factor kappa = 1 + (h nu_imag / kT)^2 / 24
  quantum_wigner    quantum_harmonic x kappa (upper-end quantum estimate)
  fixed_1e13        k = 1e13 s^-1 x exp(-Ea/kT)

Checks: hop/saddle topology and imaginary-mode axes, ZPE recomputed from the modes vs key_points.json,
detailed balance k_ij Z_i = k_ji Z_j for every hop pair, variant and temperature (Z_i from the minimum's
own modes, independent of the saddles), and the M1 classical numbers of T3-00 scope section 3.

All rates are MLIP-based harmonic-TST estimates (H-only Hessian, substrate frozen, 1 ML collective
H motion in a 1x1 cell), not DFT-validated values. Outputs contain no absolute local paths.
Needs only NumPy/SciPy (ASE version is recorded if importable). Setup and run order: RUNBOOK section 9.
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
from scipy import constants as sc

# ----------------------------------------------------------------------------- constants
KB_EV = sc.physical_constants["Boltzmann constant in eV/K"][0]   # eV/K
H_EV = sc.physical_constants["Planck constant in eV/Hz"][0]      # eV s
C_CM = sc.c * 100.0                                              # cm/s
HC_EV_CM = H_EV * C_CM                                           # eV cm  (1 cm^-1 -> eV)
NU_FIXED = 1.0e13                                                # s^-1

KEY_POINTS_REL = "artifacts/neb/03_analysis/key_points.json"  # committed Track 1 key points
ENV_CKPT = {"M1": "MACE_M1_CHECKPOINT", "M0": "MACE_MP0_SMALL_CHECKPOINT"}
MODELS = ["M1", "M0"]
MINIMA = ["B1", "T", "B2"]
SADDLES = {"S1": {"between": ("B1", "T"), "axis": "y"},
           "S2": {"between": ("T", "B2"), "axis": "x"}}
# hop label -> (from, to, saddle)
HOPS = {"B1->T": ("B1", "T", "S1"), "T->B1": ("T", "B1", "S1"),
        "T->B2": ("T", "B2", "S2"), "B2->T": ("B2", "T", "S2")}
PAIRS = [("B1->T", "T->B1"), ("T->B2", "B2->T")]
VARIANTS = ["classical", "quantum_harmonic", "classical_wigner", "quantum_wigner", "fixed_1e13"]
MAIN_VARIANT = "classical"

# Square surface cell, a = 3.358 A (T1 slab). Fractional site positions in the cell and the
# hop events (displacement in units of a). T has 2 B1 neighbours along +-y and 2 B2 along +-x;
# B1 connects only to T (+-y), B2 only to T (+-x). Direct B1<->B2 and hollow routes excluded.
A_LATTICE = 3.358
SITE_POS = {"T": (0.0, 0.0), "B1": (0.0, 0.5), "B2": (0.5, 0.0)}
EVENTS = [  # (from, to, hop, (dx, dy) / a)
    ("B1", "T", "B1->T", (0.0, -0.5)), ("B1", "T", "B1->T", (0.0, +0.5)),
    ("T", "B1", "T->B1", (0.0, +0.5)), ("T", "B1", "T->B1", (0.0, -0.5)),
    ("T", "B2", "T->B2", (+0.5, 0.0)), ("T", "B2", "T->B2", (-0.5, 0.0)),
    ("B2", "T", "B2->T", (-0.5, 0.0)), ("B2", "T", "B2->T", (+0.5, 0.0)),
]

# T3-00 scope section 3 (M1, classical), as printed there (rounded): Ea (eV), nu (s^-1), k(300 K) (s^-1)
SCOPE_M1 = {"B1->T": (0.253, 4.8e13, 2.8e9), "T->B1": (0.066, 5.5e12, 4.4e11),
            "T->B2": (0.044, 5.4e12, 9.9e11), "B2->T": (0.223, 4.5e13, 8.0e9)}


# ----------------------------------------------------------------------------- helpers
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def portfolio_root(here):
    """Repository root: this script lives in <repo>/src/kmc/<step>/. Recorded paths are relative to it."""
    return here.parents[2], "repo-relative (nbp-mace)"


def portable(path, root):
    """Path relative to the portfolio root (forward slashes), else local://<name>. Never absolute."""
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return f"local://{Path(path).name}"


def ln_q_quantum(nu_cm, T):
    """ln of the quantum harmonic partition function per mode, energy zero at the classical minimum."""
    x = HC_EV_CM * np.asarray(nu_cm, float) / (KB_EV * T)
    return float(np.sum(-0.5 * x - np.log1p(-np.exp(-x))))


def ln_q_classical(nu_cm, T):
    x = HC_EV_CM * np.asarray(nu_cm, float) / (KB_EV * T)
    return float(np.sum(-np.log(x)))


def zpe_ev(nu_cm):
    return 0.5 * HC_EV_CM * float(np.sum(nu_cm))


# ----------------------------------------------------------------------------- input
def load_key_points(path):
    with open(path) as fh:
        d = json.load(fh)
    kp = {}
    for r in d["key_points"]:
        kp.setdefault(r["model"], {})[r["point"]] = r
    return d, kp


def check_topology(kp_model, model, warnings):
    """Minima: 3 real modes; saddles: exactly one imaginary mode along the expected hop axis."""
    for p in MINIMA:
        r = kp_model[p]
        if r["n_imaginary"] != 0 or min(r["modes_cm1"]) <= 0:
            warnings.append(f"{model} {p}: minimum has an imaginary mode")
    for s, info in SADDLES.items():
        r = kp_model[s]
        neg = [(v, a) for v, a in zip(r["modes_cm1"], r["mode_axes"]) if v < 0]
        if r["n_imaginary"] != 1 or len(neg) != 1:
            warnings.append(f"{model} {s}: expected exactly one imaginary mode, got {len(neg)}")
        elif neg[0][1] != info["axis"]:
            warnings.append(f"{model} {s}: imaginary mode along {neg[0][1]}, expected {info['axis']}")


def site_params(kp_model):
    return {p: {"E_eV": kp_model[p]["E_rel_eV"], "modes_cm1": list(kp_model[p]["modes_cm1"]),
                "ZPE_eV": zpe_ev(kp_model[p]["modes_cm1"])} for p in MINIMA}


def hop_params(kp_model):
    """Barrier, real/imaginary modes, Vineyard prefactor and ZPE-corrected barrier for each hop."""
    out = {}
    for hop, (i, j, s) in HOPS.items():
        mi = np.array(kp_model[i]["modes_cm1"], float)
        ms_all = np.array(kp_model[s]["modes_cm1"], float)
        ms = ms_all[ms_all > 0]
        nu_imag = float(-ms_all[ms_all < 0][0])
        Ea = kp_model[s]["E_rel_eV"] - kp_model[i]["E_rel_eV"]
        nu_v = C_CM * float(np.prod(mi) / np.prod(ms))
        out[hop] = {"from": i, "to": j, "saddle": s, "Ea_eV": Ea,
                    "modes_min_cm1": mi.tolist(), "modes_saddle_real_cm1": ms.tolist(),
                    "nu_imag_cm1": nu_imag, "nu_vineyard_s-1": nu_v,
                    "Ea_ZPE_eV": Ea + zpe_ev(ms) - zpe_ev(mi),
                    "T_crossover_K": HC_EV_CM * nu_imag / (2 * np.pi * KB_EV)}
    return out


# ----------------------------------------------------------------------------- rates
def wigner_kappa(nu_imag_cm, T):
    u = HC_EV_CM * nu_imag_cm / (KB_EV * T)
    return 1.0 + u * u / 24.0


def rate(hp, T, variant):
    """Rate constant (s^-1) of one hop at temperature T (K)."""
    boltz = np.exp(-hp["Ea_eV"] / (KB_EV * T))
    if variant == "fixed_1e13":
        return NU_FIXED * boltz
    if variant.startswith("classical"):
        k = hp["nu_vineyard_s-1"] * boltz
    elif variant.startswith("quantum"):
        lnq = ln_q_quantum(hp["modes_saddle_real_cm1"], T) - ln_q_quantum(hp["modes_min_cm1"], T)
        k = KB_EV * T / H_EV * np.exp(lnq) * boltz
    else:
        raise ValueError(variant)
    if variant.endswith("wigner"):
        k *= wigner_kappa(hp["nu_imag_cm1"], T)
    return float(k)


def ln_site_weight(sp, T, variant):
    """ln Z_i of a minimum (same reference for all sites) for the statistics implied by `variant`."""
    lnz = -sp["E_eV"] / (KB_EV * T)
    if variant == "fixed_1e13":
        return lnz
    if variant.startswith("classical"):
        return lnz + ln_q_classical(sp["modes_cm1"], T)
    return lnz + ln_q_quantum(sp["modes_cm1"], T)


def populations(sites, T, variant):
    """Boltzmann site populations (one B1, one T, one B2 per cell)."""
    lz = np.array([ln_site_weight(sites[p], T, variant) for p in MINIMA])
    w = np.exp(lz - lz.max())
    return dict(zip(MINIMA, (w / w.sum()).tolist()))


def detailed_balance(hops, sites, temps, variant):
    """max |k_ij Z_i / (k_ji Z_j) - 1| per hop pair over the temperature grid."""
    res = {}
    for f, r in PAIRS:
        i, j = hops[f]["from"], hops[f]["to"]
        dev = 0.0
        for T in temps:
            lhs = np.log(rate(hops[f], T, variant)) + ln_site_weight(sites[i], T, variant)
            rhs = np.log(rate(hops[r], T, variant)) + ln_site_weight(sites[j], T, variant)
            dev = max(dev, abs(np.expm1(lhs - rhs)))
        res[f"{f} / {r}"] = dev
    return res


# ----------------------------------------------------------------------------- main
def main():
    here = Path(__file__).resolve().parent
    stamp = datetime.date.today().isoformat()
    root, root_src = portfolio_root(here)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key-points", default=str(root / KEY_POINTS_REL))
    ap.add_argument("--out", default=str(here / f"{stamp}_rates"))
    ap.add_argument("--tmin", type=float, default=200.0)
    ap.add_argument("--tmax", type=float, default=600.0)
    ap.add_argument("--dt", type=float, default=10.0)
    ap.add_argument("--a", type=float, default=A_LATTICE, help="surface lattice constant (A)")
    ap.add_argument("--skip-ckpt-hash", action="store_true",
                    help="do not re-hash the checkpoints from the env vars (use the SHA in key_points.json)")
    args = ap.parse_args()

    kp_path, out = Path(args.key_points), Path(args.out)
    if not kp_path.exists():
        sys.exit(f"key_points.json not found: {portable(kp_path, root)}  (see RUNBOOK section 9)")
    out.mkdir(parents=True, exist_ok=True)
    temps = np.round(np.arange(args.tmin, args.tmax + 0.5 * args.dt, args.dt), 6)
    if 300.0 not in temps:
        temps = np.sort(np.append(temps, 300.0))

    kp_doc, kp = load_key_points(kp_path)
    warnings, checks = [], {}

    # ---- provenance
    up = kp_doc.get("provenance", {})
    ckpt = {}
    for m in MODELS:
        recorded = up.get("checkpoint_sha256", {}).get(m)
        p = os.environ.get(ENV_CKPT[m])
        if args.skip_ckpt_hash or not p or not Path(p).exists():
            ckpt[m] = {"sha256": recorded, "source": "key_points.json provenance (not re-hashed)"}
            continue
        h = sha256(p)
        ckpt[m] = {"sha256": h, "source": f"re-hashed from env:{ENV_CKPT[m]}",
                   "matches_key_points": h == recorded}
        if h != recorded:
            warnings.append(f"{m} checkpoint SHA-256 differs from the one recorded in key_points.json")
    import scipy
    versions = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    try:
        import ase
        versions["ase"] = ase.__version__
    except ImportError:
        pass
    prov = {"date": stamp, "script": "src/kmc/01_rates/rates.py", "root": root_src,
            "input": {"file": portable(kp_path, root), "sha256": sha256(kp_path),
                      "upstream_date": up.get("date"), "upstream_note": up.get("note")},
            "checkpoint_sha256": ckpt, "versions": versions,
            "constants": {"kB_eV_K": KB_EV, "h_eV_s": H_EV, "c_cm_s": C_CM, "source": "scipy.constants (CODATA)"},
            "note": ("MLIP-based harmonic-TST estimates (H-only Hessian, substrate frozen; 1 ML H, 1x1 cell, "
                     "collective H motion); not DFT-validated. Classical Vineyard hTST is the main result.")}

    # ---- parameters + checks per model
    models = {}
    for m in MODELS:
        if m not in kp:
            warnings.append(f"model {m} missing in key_points.json")
            continue
        check_topology(kp[m], m, warnings)
        sites, hops = site_params(kp[m]), hop_params(kp[m])
        zdev = {p: abs(sites[p]["ZPE_eV"] - kp[m][p]["ZPE_H_eV"]) for p in MINIMA}
        zdev.update({s: abs(zpe_ev([v for v in kp[m][s]["modes_cm1"] if v > 0]) - kp[m][s]["ZPE_H_eV"])
                     for s in SADDLES})
        if max(zdev.values()) > 5e-5:
            warnings.append(f"{m}: ZPE from modes differs from key_points.json by {max(zdev.values()):.1e} eV")
        db = {v: detailed_balance(hops, sites, temps, v) for v in VARIANTS}
        db_max = max(max(d.values()) for d in db.values())
        if db_max > 1e-10:
            warnings.append(f"{m}: detailed balance violated (max rel. dev. {db_max:.1e})")
        models[m] = {"sites": sites, "hops": hops,
                     "rates_s-1": {v: {h: [rate(hops[h], T, v) for T in temps] for h in HOPS} for v in VARIANTS},
                     "wigner_kappa": {s: [wigner_kappa(hops[h]["nu_imag_cm1"], T) for T in temps]
                                      for s, h in (("S1", "B1->T"), ("S2", "T->B2"))},
                     "populations": {v: {p: [populations(sites, T, v)[p] for T in temps] for p in MINIMA}
                                     for v in VARIANTS}}
        checks[f"{m}_zpe_max_abs_dev_eV"] = max(zdev.values())
        checks[f"{m}_detailed_balance_max_rel_dev"] = {v: db[v] for v in VARIANTS}

    # ---- T3-00 scope section 3 (M1 classical)
    if "M1" in models:
        sc_rows = {}
        for h, (Ea_s, nu_s, k_s) in SCOPE_M1.items():
            hp = models["M1"]["hops"][h]
            k300 = rate(hp, 300.0, "classical")
            sc_rows[h] = {"Ea_eV": [round(hp["Ea_eV"], 5), Ea_s], "nu_s-1": [hp["nu_vineyard_s-1"], nu_s],
                          "k300_s-1": [k300, k_s],
                          "ok": bool(abs(hp["Ea_eV"] - Ea_s) <= 1e-3 and abs(hp["nu_vineyard_s-1"] / nu_s - 1) <= 0.03
                                     and abs(k300 / k_s - 1) <= 0.05)}
        checks["scope_section3_M1_classical"] = {
            "rows": sc_rows, "all_ok": all(r["ok"] for r in sc_rows.values()),
            "tolerance": "Ea 1 meV, nu 3 %, k(300 K) 5 % (scope values are rounded to 2-3 digits)"}
        if not checks["scope_section3_M1_classical"]["all_ok"]:
            warnings.append("M1 classical rates differ from T3-00 scope section 3 beyond rounding tolerance")

    lattice = {"type": "square", "a_A": args.a, "site_frac_pos": SITE_POS,
               "events": [{"from": f, "to": t, "hop": h, "d_over_a": list(d)} for f, t, h, d in EVENTS],
               "note": "one T, B1, B2 per cell; target cell = from_pos + d - to_pos (integer offset)"}
    doc = {"provenance": prov, "warnings": warnings, "checks": checks, "lattice": lattice,
           "hops": {h: {"from": i, "to": j, "saddle": s} for h, (i, j, s) in HOPS.items()},
           "variants": VARIANTS, "main_variant": MAIN_VARIANT, "T_K": temps.tolist(), "models": models}
    with open(out / "rates.json", "w") as fh:
        json.dump(doc, fh, indent=2)

    with open(out / "rates.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "variant", "hop", "T_K", "Ea_eV", "prefactor_eff_s-1", "kappa_wigner", "k_s-1"])
        for m, md in models.items():
            for v in VARIANTS:
                for h, hp in md["hops"].items():
                    for n, T in enumerate(temps):
                        k = md["rates_s-1"][v][h][n]
                        kap = wigner_kappa(hp["nu_imag_cm1"], T) if v.endswith("wigner") else 1.0
                        w.writerow([m, v, h, f"{T:g}", f"{hp['Ea_eV']:.5f}",
                                    f"{k / np.exp(-hp['Ea_eV'] / (KB_EV * T)):.6e}", f"{kap:.6f}", f"{k:.6e}"])

    # ---- console summary
    i300 = int(np.argmin(abs(temps - 300.0)))
    for m, md in models.items():
        print(f"\n{m}  hop     Ea(eV)  Ea+dZPE  nu_V(s-1)   T_c(K)  k(300 K) classical / quantum / +Wigner / 1e13")
        for h, hp in md["hops"].items():
            ks = [md["rates_s-1"][v][h][i300] for v in ("classical", "quantum_harmonic", "classical_wigner", "fixed_1e13")]
            print(f"    {h:6} {hp['Ea_eV']:7.4f} {hp['Ea_ZPE_eV']:8.4f}  {hp['nu_vineyard_s-1']:9.3e}  {hp['T_crossover_K']:6.1f}  "
                  + "  ".join(f"{k:9.3e}" for k in ks))
        pop = {p: md["populations"]["classical"][p][i300] for p in MINIMA}
        print("    populations 300 K (classical): " + "  ".join(f"{p} {v:.4e}" for p, v in pop.items()))
        print("    detailed balance max rel. dev.: "
              + f"{max(max(d.values()) for d in checks[f'{m}_detailed_balance_max_rel_dev'].values()):.1e}")
    if "scope_section3_M1_classical" in checks:
        print(f"\nscope section 3 (M1 classical) reproduced: {checks['scope_section3_M1_classical']['all_ok']}")
    print("\nWARNINGS:\n  " + "\n  ".join(warnings) if warnings else "\nno warnings")
    print(f"\ntables -> {out.name}\\rates.csv, rates.json   ({len(temps)} temperatures, {temps[0]:g}-{temps[-1]:g} K)")


if __name__ == "__main__":
    main()
