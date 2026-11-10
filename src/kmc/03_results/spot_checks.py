"""T3-03 (2/3): kMC spot checks of every sensitivity variant against the exact lattice D.

For each model, variant (sensitivity.VARIANTS, incl. the direct B1<->B2 network) and temperature, the
validated T3-02 BKL engine (02_kmc/kmc.py, full three-site network) is run with the variant's rates and
event table, and D_x, D_y and the time-weighted site occupations are compared with the exact Bloch
solution and the Boltzmann populations: z = (kMC - exact) / SE, SE from 32 independent walker batches.
Pass criterion |z| < 4 for every D and occupation (as in T3-02). The direct-route network breaks the
star topology, so its reference is the Bloch solver only and the coarse-grained model is not used.

Also saves the MSD(t) of the M1 classical 300 K run for the publication figure (b).

Writes <out>/ (default 03_results/<date>_kmc_spot/):
  spot_D.csv                     one row per case: D, SE, exact D, z, occupations, populations, z_occ, timing
  spot_summary.json              provenance (input SHA-256, seed, versions), cases, max |z|, pass flag
  msd_300K_M1_classical.json     t, MSD_x, MSD_y (A^2), exact D, kMC D +- SE

Physical numbers are MLIP-based harmonic-TST estimates for a lattice model parameterised with 1 ML
collective barriers, not DFT-validated values. Exit code 0 = all spot checks pass.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import sensitivity as S  # noqa: E402

R, K = S.R, S.K
DEFAULT_SEED = 20261110


def run_one(rd, direct, model, variant, T, n_walkers, n_events, seed, n_samples, n_batches):
    spec = S.VARIANTS[variant]
    md = rd["models"][model]
    a_A, base_ev = rd["lattice"]["a_A"], rd["lattice"]["events"]
    ex = S.exact_point(md, spec, T, base_ev, a_A, direct.get(model))
    k, pop = ex["k"], ex["pop"]
    ev = S.events_for(spec, base_ev)
    net = K.build_full(ev, k)
    ev_rate = sum(pop[s] * net["Rtot"][i] for i, s in enumerate(K.SITES))   # stationary events per second
    t_end = n_events / ev_rate
    res = K.run_case(k, a_A, pop, "full", n_walkers, t_end, seed, ev, n_samples, n_batches)
    K.compare(res, (ex["D_x"], ex["D_y"]), "exact")
    zocc = {}
    for s in K.SITES:
        se = res["occupation_se"][s]
        d = res["occupation"][s] - pop[s]
        zocc[s] = float(d / se) if se > 0 else (0.0 if abs(d) < 1e-15 else float("inf"))
    return ex, res, zocc


def main(argv=None):
    stamp = datetime.date.today().isoformat()
    root, root_src = R.portfolio_root(HERE)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rates", help="rates.json (default: newest 01_rates/<date>_rates/rates.json)")
    ap.add_argument("--models", nargs="+", default=list(S.MODELS))
    ap.add_argument("--variants", nargs="+", default=list(S.VARIANTS), choices=list(S.VARIANTS))
    ap.add_argument("--temps", nargs="+", type=float, default=[200.0, 300.0, 600.0])
    ap.add_argument("--n-walkers", type=int, default=20000)
    ap.add_argument("--n-events", type=float, default=2000, help="full-model events per walker (sets t_end)")
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--n-batches", type=int, default=32)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--z-tol", type=float, default=4.0)
    ap.add_argument("--out", default=str(HERE / f"{stamp}_kmc_spot"))
    args = ap.parse_args(argv)

    rpath = Path(args.rates) if args.rates else K.default_inputs()[0]
    rd = json.loads(rpath.read_text())
    direct = S.direct_route(root, args.models)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"rates : {R.portable(rpath, root)}\nout   : {R.portable(out, root)}")

    combos = [(m, v, T) for m in args.models for v in args.variants for T in args.temps]
    seeds = np.random.SeedSequence(args.seed).spawn(len(combos))
    rows, cases, msd_doc = [], [], None
    t_start = time.perf_counter()
    for (m, v, T), ss in zip(combos, seeds):
        seed = int(ss.generate_state(1)[0])
        ex, res, zocc = run_one(rd, direct, m, v, T, args.n_walkers, args.n_events, seed,
                                args.n_samples, args.n_batches)
        occ = res["occupation"]
        zmax_occ = max(abs(z) for z in zocc.values())
        row = {"model": m, "variant": v, "family": S.VARIANTS[v]["family"], "T_K": T,
               "D_x_cm2_s": res["D_x"], "se_x": res["se_x"], "exact_D_x": ex["D_x"], "z_x": res["z_exact_x"],
               "D_y_cm2_s": res["D_y"], "se_y": res["se_y"], "exact_D_y": ex["D_y"], "z_y": res["z_exact_y"],
               "occ_B1": occ["B1"], "occ_T": occ["T"], "occ_B2": occ["B2"],
               "pop_B1": ex["pop"]["B1"], "pop_T": ex["pop"]["T"], "pop_B2": ex["pop"]["B2"],
               "z_occ_B1": zocc["B1"], "z_occ_T": zocc["T"], "z_occ_B2": zocc["B2"],
               "events_per_walker": res["events_per_walker"], "wall_s": res["wall_s"],
               "n_walkers": res["n_walkers"], "t_end_s": res["t_end_s"], "seed": seed}
        rows.append(row)
        cases.append({**{kk: row[kk] for kk in ("model", "variant", "family", "T_K", "seed")},
                      "rates_s-1": ex["k"], "populations_boltzmann": ex["pop"],
                      "exact_D": [ex["D_x"], ex["D_y"]],
                      "kmc": {kk: vv for kk, vv in res.items() if not kk.startswith("_")
                              and kk not in ("msd_A2", "t_samples_s", "residence")},
                      "z_occupation": zocc})
        print(f"{m} {v:24s} {T:5.0f} K  D_x {res['D_x']:.3e}±{res['se_x']:.1e} (z {res['z_exact_x']:+.2f})  "
              f"D_y {res['D_y']:.3e}±{res['se_y']:.1e} (z {res['z_exact_y']:+.2f})  max|z_occ| {zmax_occ:.2f}  "
              f"{res['wall_s']:.1f} s")
        if m == "M1" and v == "classical" and abs(T - 300.0) < 1e-6:
            msd_doc = {"model": m, "variant": v, "T_K": T, "seed": seed, "n_walkers": res["n_walkers"],
                       "t_end_s": res["t_end_s"], "fit_from_fraction": 0.2, "t_s": res["t_samples_s"],
                       "msd_A2": res["msd_A2"], "D_kmc_cm2_s": [res["D_x"], res["D_y"]],
                       "se_cm2_s": [res["se_x"], res["se_y"]], "D_exact_cm2_s": [ex["D_x"], ex["D_y"]],
                       "z": [res["z_exact_x"], res["z_exact_y"]]}
    wall = time.perf_counter() - t_start

    with open(out / "spot_D.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    zD = max(max(abs(r["z_x"]), abs(r["z_y"])) for r in rows)
    zO = max(max(abs(r["z_occ_B1"]), abs(r["z_occ_T"]), abs(r["z_occ_B2"])) for r in rows)
    zall = [r[c] for r in rows for c in ("z_x", "z_y")]
    passed = bool(zD < args.z_tol and zO < args.z_tol)
    summary = {
        "provenance": {"date": stamp, "script": "src/kmc/03_results/spot_checks.py", "root": root_src,
                       "input": {"rates": {"file": R.portable(rpath, root), "sha256": R.sha256(rpath)},
                                 "xy_maps": {m: {"file": d["file"], "sha256": d["sha256"]} for m, d in direct.items()}},
                       "engine": "src/kmc/02_kmc/kmc.py (full network, BKL)", "versions": K.versions(),
                       "seed": args.seed, "seed_note": "per-case seeds spawned from numpy SeedSequence(seed) in loop order",
                       "n_walkers": args.n_walkers, "n_events_target": args.n_events, "n_samples": args.n_samples,
                       "n_batches": args.n_batches, "z_tol": args.z_tol,
                       "note": ("Engine check against the exact lattice solution for every variant; physical numbers "
                                "are MLIP-based harmonic-TST estimates for a lattice model with 1 ML collective "
                                "barriers, not DFT-validated values.")},
        "n_cases": len(rows), "wall_s_total": wall, "max_abs_z_D": zD, "max_abs_z_occupation": zO,
        "z_D_mean": float(np.mean(zall)), "z_D_std": float(np.std(zall, ddof=1)) if len(zall) > 1 else None,
        "passed": passed, "cases": cases}
    (out / "spot_summary.json").write_text(json.dumps(summary, indent=1))
    if msd_doc:
        (out / "msd_300K_M1_classical.json").write_text(json.dumps(msd_doc, indent=1))
    print(f"\n{len(rows)} cases, {wall / 60:.1f} min;  max |z| D {zD:.2f}, occupations {zO:.2f};  "
          f"z(D) mean {summary['z_D_mean']:+.2f}, sd {summary['z_D_std']:.2f}  ->  {'PASS' if passed else 'FAIL'}")
    print(f"outputs -> {R.portable(out, root)}: spot_D.csv, spot_summary.json"
          + (", msd_300K_M1_classical.json" if msd_doc else ""))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
