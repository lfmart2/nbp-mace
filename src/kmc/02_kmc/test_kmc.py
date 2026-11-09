"""T3-02 validation of kmc.py against the exact lattice solution (T3-00 scope section 5).

Five checks, each with an explicit statistical tolerance (|z| < Z_TOL unless stated):
  1. Boltzmann populations: time-weighted kMC site occupations vs rates.json populations.
  2. Residence times: exponential with mean 1/sum(k) per site (mean and 2nd-moment z-tests, KS p > 1e-3).
  3. Exact D: kMC D_x, D_y vs exact_D.json (and the closed form) at 200, 300, 400, 600 K (M1 classical),
     plus M0 classical and M1 quantum_harmonic at 300 K.
  4. Coarse-grained (T eliminated) = full: D and occupations agree; reports the speed-up.
  5. Limits: equal saddles (k(T->B1) = k(T->B2), unequal bridges) give isotropic D; removing the S2 events
     gives D_x = 0 exactly.

Run:  python test_kmc.py            (plain asserts; prints a table, writes <date>_tests/test_report.json/.csv)
      python -m pytest test_kmc.py  (also works if pytest is ever installed; not needed)
Runtime ~1-2 min on a laptop CPU. Seeds are fixed, so results are reproducible bit for bit.
The tests validate the engine; the physical numbers remain MLIP-based harmonic-TST estimates with 1 ML barriers.
"""
from __future__ import annotations

import csv
import datetime
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kmc  # noqa: E402

Z_TOL = 4.0
KS_P_MIN = 1e-3
N_WALKERS = 20000
N_EVENTS = 1500            # full-model events per walker
BASE_SEED = kmc.DEFAULT_SEED

_BOOK = None
_CACHE = {}
ROWS = []                  # (check, case, quantity, kmc, reference, z, tolerance, passed)


def book():
    global _BOOK
    if _BOOK is None:
        r, e = kmc.default_inputs()
        _BOOK = kmc.RateBook(r, e)
    return _BOOK


def seed_for(key):
    return int(np.random.SeedSequence([BASE_SEED, zlib.crc32(key.encode())]).generate_state(1)[0])


def run(model, variant, T, mode, k=None, tag=""):
    """Cached kMC run. k overrides the rate set (limit tests); populations then come from the rates."""
    key = f"{model}|{variant}|{T:g}|{mode}|{tag}"
    if key not in _CACHE:
        b = book()
        if k is None:
            k = b.rates(model, variant, T)
            p = b.populations(model, variant, T)
        else:
            p = kmc.tree_populations(k)
        k_ref = b.rates(model, variant, T)
        t_end = kmc.t_end_for(k_ref, b.populations(model, variant, T), N_EVENTS)  # same time base
        res = kmc.run_case(k, b.a_A, p, mode, N_WALKERS, t_end, seed_for(key), b.events)
        res["_k"], res["_p"] = k, p
        _CACHE[key] = res
    return _CACHE[key]


def record(check, case, qty, val, ref, z, tol, passed):
    ROWS.append({"check": check, "case": case, "quantity": qty, "kmc": val, "reference": ref,
                 "z": z, "tolerance": tol, "passed": bool(passed)})
    return passed


def zcheck(check, case, qty, val, ref, se, tol=Z_TOL):
    z = (val - ref) / se if se > 0 else (0.0 if val == ref else np.inf)
    ok = abs(z) < tol
    record(check, case, qty, val, ref, float(z), f"|z|<{tol:g}", ok)
    return ok


# ----------------------------------------------------------------------------- 1
def test_1_boltzmann_populations():
    ok = True
    for model, variant, T in [("M1", "classical", 300), ("M1", "classical", 600),
                              ("M1", "quantum_harmonic", 300), ("M0", "classical", 300)]:
        res = run(model, variant, T, "full")
        pop = book().populations(model, variant, T)
        for s in kmc.SITES:
            ok &= zcheck(1, f"{model} {variant} {T} K", f"occ {s}", res["occupation"][s], pop[s],
                         max(res["occupation_se"][s], 1e-15))
    assert ok, "kMC occupations differ from Boltzmann populations"


# ----------------------------------------------------------------------------- 2
def test_2_residence_times():
    ok = True
    for model, variant, T, mode in [("M1", "classical", 300, "full"), ("M1", "classical", 200, "full"),
                                    ("M1", "classical", 300, "cg")]:
        res = run(model, variant, T, mode)
        for s, r in res["residence"].items():
            case = f"{model} {variant} {T} K {mode} {s}"
            ok &= record(2, case, "mean residence (s)", r["mean_s"], r["mean_theory_s"], r["z_mean"],
                         f"|z|<{Z_TOL:g}", abs(r["z_mean"]) < Z_TOL)
            ok &= record(2, case, "<t^2>/(2 mean^2)", r["m2_over_theory"], 1.0, r["z_m2"],
                         f"|z|<{Z_TOL:g}", abs(r["z_m2"]) < Z_TOL)
            ok &= record(2, case, f"KS p (n={r['ks_n']})", r["ks_D"], 0.0, r["ks_p"],
                         f"p>{KS_P_MIN:g}", r["ks_p"] > KS_P_MIN)
    assert ok, "residence times are not exponential with mean 1/sum(k)"


# ----------------------------------------------------------------------------- 3
CASES_D = [("M1", "classical", 200), ("M1", "classical", 300), ("M1", "classical", 400),
           ("M1", "classical", 600), ("M0", "classical", 300), ("M1", "quantum_harmonic", 300)]


def test_3_exact_D():
    ok = True
    b = book()
    for model, variant, T in CASES_D:
        res = run(model, variant, T, "full")
        ex = b.exact_D(model, variant, T)
        cf = kmc.closed_form_D(res["_k"], b.a_A)
        for j, ax in enumerate("xy"):
            rel = abs(cf[j] / ex[j] - 1.0)       # the two references must agree first
            ok &= record(3, f"{model} {variant} {T} K", f"closed form vs exact_D.json D_{ax}", cf[j], ex[j],
                         rel, "rel<1e-9", rel < 1e-9)
            ok &= zcheck(3, f"{model} {variant} {T} K", f"D_{ax} (cm2/s)", res[f"D_{ax}"], ex[j], res[f"se_{ax}"])
    assert ok, "kMC D disagrees with the exact lattice D"


# ----------------------------------------------------------------------------- 4
def test_4_coarse_grained_equals_full():
    ok = True
    b = book()
    speed = []
    for model, variant, T in [("M1", "classical", 200), ("M1", "classical", 300), ("M1", "classical", 600)]:
        full = run(model, variant, T, "full")
        cg = run(model, variant, T, "cg")
        ex = b.exact_D(model, variant, T)
        case = f"{model} {variant} {T} K"
        for j, ax in enumerate("xy"):
            ok &= zcheck(4, case, f"cg D_{ax} vs full", cg[f"D_{ax}"], full[f"D_{ax}"],
                         float(np.hypot(cg[f"se_{ax}"], full[f"se_{ax}"])))
            ok &= zcheck(4, case, f"cg D_{ax} vs exact", cg[f"D_{ax}"], ex[j], cg[f"se_{ax}"])
        pop = b.populations(model, variant, T)
        for s in kmc.SITES:
            ok &= zcheck(4, case, f"cg occ {s}", cg["occupation"][s], pop[s], max(cg["occupation_se"][s], 1e-15))
        er = full["events_per_walker"] / cg["events_per_walker"]
        wr = full["wall_s"] / cg["wall_s"]
        speed.append((T, er, wr))
        ok &= record(4, case, "speed-up (events, wall)", er, wr, None, "events ratio > 1.5", er > 1.5)
    test_4_coarse_grained_equals_full.speed = speed
    assert ok, "coarse-grained model differs from the full model"


# ----------------------------------------------------------------------------- 5
def test_5_limits():
    ok = True
    b = book()
    base = b.rates("M1", "classical", 300)
    # (a) equal saddles: k(T->B1) = k(T->B2); bridge rates left unequal (B1 and B2 still differ)
    k_eq = dict(base)
    k_eq["T->B1"] = base["T->B2"]
    k_eq["B1->T"] = base["B1->T"] * base["T->B2"] / base["T->B1"]   # keeps the B1 site energy
    cf = kmc.closed_form_D(k_eq, b.a_A)
    ok &= record(5, "equal saddles", "closed form D_x/D_y", cf[0] / cf[1], 1.0, None, "== 1 (1e-12)",
                 abs(cf[0] / cf[1] - 1) < 1e-12)
    try:                                            # independent Bloch solution from T3-01 (needs SciPy)
        sys.path.insert(0, str(HERE.parent / "01_rates"))
        import exact_diffusion as ED
        idx = {s: i for i, s in enumerate(["B1", "T", "B2"])}
        bloch, _ = ED.exact_D(b.events, idx, k_eq, b.a_A * 1e-8)
        for j, ax in enumerate("xy"):
            rel = abs(bloch[ax] / cf[j] - 1)
            ok &= record(5, "equal saddles", f"Bloch vs closed form D_{ax}", bloch[ax], cf[j], rel,
                         "rel<1e-9", rel < 1e-9)
    except ImportError as exc:
        record(5, "equal saddles", f"Bloch cross-check skipped ({exc.name} missing)", None, None, None, "-", True)
    res = run("M1", "classical", 300, "full", k=k_eq, tag="equal_saddles")
    d = res["D_x"] - res["D_y"]
    ok &= record(5, "equal saddles", "D_x - D_y (cm2/s)", d, 0.0, d / res["se_diff"], f"|z|<{Z_TOL:g}",
                 abs(d / res["se_diff"]) < Z_TOL)
    for j, ax in enumerate("xy"):
        ok &= zcheck(5, "equal saddles", f"D_{ax} vs closed form", res[f"D_{ax}"], cf[j], res[f"se_{ax}"])
    # (b) remove S2 events: no B2 access, no x motion
    k_no = dict(base)
    k_no["T->B2"] = 0.0
    k_no["B2->T"] = 0.0
    cf = kmc.closed_form_D(k_no, b.a_A)
    res = run("M1", "classical", 300, "full", k=k_no, tag="no_S2")
    ok &= record(5, "no S2", "D_x (cm2/s)", res["D_x"], 0.0, None, "== 0 exactly", res["D_x"] == 0.0)
    ok &= record(5, "no S2", "occ B2", res["occupation"]["B2"], 0.0, None, "== 0 exactly",
                 res["occupation"]["B2"] == 0.0)
    ok &= zcheck(5, "no S2", "D_y vs closed form", res["D_y"], cf[1], res["se_y"])
    assert ok, "limit tests failed"


# ----------------------------------------------------------------------------- runner
TESTS = [test_1_boltzmann_populations, test_2_residence_times, test_3_exact_D,
         test_4_coarse_grained_equals_full, test_5_limits]


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def main():
    t0 = time.perf_counter()
    failed = []
    for t in TESTS:
        t1 = time.perf_counter()
        try:
            t()
            status = "PASS"
        except AssertionError as exc:
            status = f"FAIL ({exc})"
            failed.append(t.__name__)
        print(f"{t.__name__:38s} {status}  [{time.perf_counter() - t1:.1f} s]")
    print()
    print(f"{'chk':3s} {'case':34s} {'quantity':36s} {'kMC':>11s} {'ref':>11s} {'z / stat':>9s}  tol")
    for r in ROWS:
        flag = "" if r["passed"] else "  <-- FAIL"
        print(f"{r['check']:<3d} {r['case']:34s} {r['quantity']:36s} {_fmt(r['kmc']):>11s} "
              f"{_fmt(r['reference']):>11s} {_fmt(r['z']):>9s}  {r['tolerance']}{flag}")
    sp = getattr(test_4_coarse_grained_equals_full, "speed", [])
    for T, er, wr in sp:
        print(f"cg speed-up at {T} K: {er:.2f}x fewer events, {wr:.2f}x wall time")
    total = time.perf_counter() - t0
    print(f"\n{len(TESTS) - len(failed)}/{len(TESTS)} checks passed in {total:.0f} s")

    root, root_src = kmc.portfolio_root()
    b = book()
    out = HERE / f"{datetime.date.today().isoformat()}_tests"
    out.mkdir(exist_ok=True)
    with open(out / "test_report.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ROWS[0].keys()))
        w.writeheader()
        w.writerows(ROWS)
    rep = {"provenance": {"date": datetime.date.today().isoformat(), "script": "src/kmc/02_kmc/test_kmc.py",
                          "root": root_src,
                          "input": {"rates": {"file": kmc.portable(b.rates_path, root), "sha256": kmc.sha256(b.rates_path)},
                                    "exact_D": {"file": kmc.portable(b.exact_path, root),
                                                "sha256": kmc.sha256(b.exact_path)}},
                          "versions": kmc.versions(), "base_seed": BASE_SEED, "n_walkers": N_WALKERS,
                          "n_events_per_walker": N_EVENTS, "z_tol": Z_TOL, "ks_p_min": KS_P_MIN,
                          "note": "validates the engine against the exact lattice solution; physical numbers are "
                                  "MLIP-based harmonic-TST estimates with 1 ML barriers, not DFT-validated"},
           "passed": not failed, "failed": failed, "runtime_s": total,
           "cg_speedup": [{"T_K": T, "event_ratio": er, "wall_ratio": wr} for T, er, wr in sp],
           "rows": ROWS}
    (out / "test_report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(f"report -> {kmc.portable(out, root)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
