#!/usr/bin/env python
r"""
compare_neb.py - T1-02: stitch the split CI-NEB bands (B1 -> T, T -> B2) for each model
and compare M1 with M0. Everything here is an MLIP PATHWAY ESTIMATE, not DFT.

Run inside src\neb\02_neb_runs after run_neb.py:
  python compare_neb.py
Uses only finished runs (folders with summary.json), default settings (k 0.1, FIRE, no tag),
and for each model the largest image count that exists for BOTH split paths.
Output: compare_<date>\  (neb_profile_M1_M0.png, neb_comparison.csv/.json, sensitivity.csv)
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
from ase.io import read

try:
    from ase.utils.forcecurve import fit_images
except ImportError:
    fit_images = None

HERE = Path(__file__).resolve().parent
FRAMING = "MLIP pathway estimate (MACE), not DFT-validated; 1 ML H, 1x1 cell"


def finished_runs():
    runs = {}
    for d in sorted(HERE.glob("run_*")):
        f = d / "summary.json"
        if not f.is_file():
            continue
        s = json.loads(f.read_text(encoding="utf-8"))
        a = s["args"]
        if a["k"] != 0.1 or a["optimizer"] != "fire" or a.get("tag"):
            continue
        key = (a["model"], a["path"], a["nimages"])
        runs.setdefault(key, (d, s))  # first finished copy; _v2 repeats are identical
    return runs


def band_curve(d, s):
    """Image path coordinate, image energies vs B1, and the spline fit (if available)."""
    rows = s["result"]["images"]
    E = np.array([r["E_rel_B1_eV"] for r in rows])
    imgs = read(d / "band_final.extxyz", index=":")
    x = np.r_[0.0, np.cumsum([np.linalg.norm(imgs[i].positions - imgs[i - 1].positions)
                              for i in range(1, len(imgs))])]
    xf, ef = x, E
    if fit_images is not None:
        try:
            fit = fit_images(imgs)
            x = np.asarray(fit.path)
            xf = np.asarray(fit.fit_path)
            ef = np.asarray(fit.fit_energies) - fit.energies[0] + E[0]
        except Exception as exc:
            print(f"  NOTE: spline fit failed for {d.name} ({exc}); plotting images only")
    return x, E, xf, ef


def key_points(sB, sT):
    r1, r2 = sB["result"], sT["result"]
    i1, i2 = r1["climbing_image"], r2["climbing_image"]
    imag = lambda r, i: min(m["freq_cm-1"] for m in r["h_modes"][str(i)]["modes"])
    axis = lambda r, i: min(r["h_modes"][str(i)]["modes"], key=lambda m: m["freq_cm-1"])["dominant_axis"]
    T = r1["images"][-1]["E_rel_B1_eV"]
    return {
        "saddle1_B1T_eV": r1["images"][i1]["E_rel_B1_eV"],
        "T_eV": T,
        "saddle2_TB2_eV": r2["images"][i2]["E_rel_B1_eV"],
        "B2_eV": r2["images"][-1]["E_rel_B1_eV"],
        "T_well_depth_to_B1_side_eV": r1["images"][i1]["E_rel_B1_eV"] - T,
        "T_well_depth_to_B2_side_eV": r2["images"][i2]["E_rel_B1_eV"] - T,
        "saddle1_imag_cm-1": imag(r1, i1), "saddle1_imag_axis": axis(r1, i1),
        "saddle2_imag_cm-1": imag(r2, i2), "saddle2_imag_axis": axis(r2, i2),
        "converged_all": all(p["converged"] for p in sB["phases"] + sT["phases"]),
    }


def main():
    runs = finished_runs()
    out = HERE / f"compare_{dt.date.today().isoformat()}"
    out.mkdir(exist_ok=True)

    chosen, table = {}, {}
    for model in ("M1", "M0"):
        ns = sorted({n for (m, p, n) in runs if m == model and p == "B1T"}
                    & {n for (m, p, n) in runs if m == model and p == "TB2"})
        if not ns:
            print(f"{model}: no finished B1T + TB2 pair, skipped")
            continue
        n = ns[-1]
        chosen[model] = (n, runs[(model, "B1T", n)], runs[(model, "TB2", n)])
        table[model] = {"nimages": n,
                        "runs": [runs[(model, "B1T", n)][0].name, runs[(model, "TB2", n)][0].name],
                        **key_points(runs[(model, "B1T", n)][1], runs[(model, "TB2", n)][1])}

    # sensitivity: every finished image count
    sens = []
    for (m, p, n), (d, s) in sorted(runs.items()):
        r = s["result"]
        sens.append({"model": m, "path": p, "nimages": n,
                     "barrier_images_eV": r["barrier_images_eV"], "barrier_fit_eV": r["barrier_fit_eV"],
                     "reaction_energy_eV": r["reaction_energy_eV"],
                     "imag_cm-1": min(md["freq_cm-1"] for md in r["h_modes"][str(r["climbing_image"])]["modes"]),
                     "run": d.name})

    # print
    print(f"T1-02 comparison | {FRAMING}")
    keys = ["saddle1_B1T_eV", "T_eV", "saddle2_TB2_eV", "B2_eV",
            "T_well_depth_to_B1_side_eV", "T_well_depth_to_B2_side_eV",
            "saddle1_imag_cm-1", "saddle2_imag_cm-1", "converged_all"]
    models = list(table)
    print(f"  {'quantity (vs B1)':30s}" + "".join(f"{m + ' (' + str(table[m]['nimages']) + ' img)':>16s}" for m in models))
    for k in keys:
        vals = [table[m][k] for m in models]
        print(f"  {k:30s}" + "".join(f"{v:>16.4f}" if isinstance(v, float) else f"{str(v):>16s}" for v in vals))
    print("\nImage-count sensitivity:")
    for r in sens:
        print(f"  {r['model']} {r['path']:3s} n={r['nimages']:2d}  barrier {r['barrier_images_eV']:.4f} eV"
              f"  fit {r['barrier_fit_eV']:.4f}  imag {r['imag_cm-1']:+.0f} cm-1")

    # files
    with open(out / "neb_comparison.json", "w", encoding="utf-8") as f:
        json.dump({"framing": FRAMING, "models": table, "sensitivity": sens,
                   "provenance": {m: chosen[m][1][1]["provenance"] for m in chosen}}, f, indent=2)
    with open(out / "neb_comparison.csv", "w", encoding="utf-8") as f:
        f.write("quantity," + ",".join(models) + "\n")
        for k in ["nimages"] + keys:
            f.write(k + "," + ",".join(str(table[m][k]) for m in models) + "\n")
    with open(out / "sensitivity.csv", "w", encoding="utf-8") as f:
        f.write(",".join(sens[0].keys()) + "\n")
        for r in sens:
            f.write(",".join(str(v) for v in r.values()) + "\n")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    colors = {"M1": "C0", "M0": "C1"}
    for model, (n, (dB, sB), (dT, sT)) in chosen.items():
        x1, e1, xf1, ef1 = band_curve(dB, sB)
        x2, e2, xf2, ef2 = band_curve(dT, sT)
        off = x1[-1]
        c = colors[model]
        label = "M1 (fine-tuned)" if model == "M1" else "M0 (MACE-MP-0-small)"
        ax.plot(np.r_[xf1, xf2 + off], np.r_[ef1, ef2], "-", color=c, lw=1.5, label=label)
        ax.plot(np.r_[x1, x2[1:] + off], np.r_[e1, e2[1:]], "o", color=c, ms=4)
        if model == "M1":
            tops = {"B1": (x1[0], e1[0]), "T": (x1[-1], e1[-1]), "B2": (x2[-1] + off, e2[-1])}
            for name, (xx, yy) in tops.items():
                ax.annotate(name, (xx, yy), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9)
    ax.axhline(0, color="0.7", lw=0.6)
    ax.set_xlabel("path length (Å, all atoms)")
    ax.set_ylabel("E − E(B1) (eV)")
    ax.set_title("H on Nb-terminated NbP, 1 ML: MLIP pathway estimate (not DFT)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "neb_profile_M1_M0.png", dpi=250)
    print(f"\nFiles in {out.name}")


if __name__ == "__main__":
    main()
