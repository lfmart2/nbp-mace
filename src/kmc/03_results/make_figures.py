"""T3-03 (3/3): publication figures for the kMC / exact-D results of H on the Nb-terminated NbP slab.

  fig_kmc_arrhenius   (a) Arrhenius D_x, D_y for M1 and M0 (exact lattice D, classical Vineyard hTST), the
                      +-0.05 eV saddle band shaded, nu = 1e13 s^-1 as dashed lines, kMC points with error bars
                      (T3-02 production run, classical; T3-03 spot checks at the band edges)
  fig_kmc_msd         (b) MSD(t) at 300 K, M1 classical, x and y, with the exact 2Dt lines
  fig_kmc_occupation  (c) kMC site occupations (full and coarse-grained) vs Boltzmann populations vs T
  fig_kmc_network     (d) network schematic: T, B1, B2, hops S1 (y) and S2 (x), optional direct B1<->B2 route
Each as PNG (300 dpi) and PDF, plus CAPTIONS.md and figures_provenance.json (input SHA-256, versions).

Inputs (defaults = newest date-stamped folders next to the scripts):
  03_results/<date>_sensitivity/results.json    (sensitivity.py)
  03_results/<date>_kmc_spot/spot_summary.json + msd_300K_M1_classical.json   (spot_checks.py)
  02_kmc/<date>_kmc/kmc_summary.json            (T3-02 production run: classical D and occupations)
  01_rates/<date>_rates/rates.json              (site parameters for the Boltzmann curves, key-point energies)

All curves and points are MLIP-based harmonic-TST estimates for a lattice model parameterised with 1 ML
collective barriers; not DFT-validated; valid only above ~150-200 K. Needs NumPy/SciPy/Matplotlib.
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import sensitivity as S  # noqa: E402

R, K = S.R, S.K
COLOR = {"M1": "#1f77b4", "M0": "#ff7f0e"}
LABEL = {"M1": "M1 (fine-tuned)", "M0": "M0 (MACE-MP-0-small)"}
SITE_COLOR = {"B1": "#2ca02c", "T": "#d62728", "B2": "#9467bd"}
NOTE = ("MLIP-based harmonic-TST estimates; lattice model with 1 ML collective barriers; "
        "not DFT-validated; valid only above ~150–200 K")
TITLE = "H on Nb-terminated NbP (1 ML barriers): lattice kMC / exact D, MLIP-based hTST (not DFT)"


def newest(folder, suffix, fname):
    return K.newest(folder, suffix, fname)


def top_T_axis(ax):
    top = ax.secondary_xaxis("top", functions=(lambda v: 1000.0 / np.maximum(v, 1e-9),
                                               lambda v: 1000.0 / np.maximum(v, 1e-9)))
    top.set_xticks([600, 500, 400, 300, 250, 200])
    top.set_xlabel("T (K)", fontsize=9)


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)


# ----------------------------------------------------------------------------- (a) Arrhenius
def fig_arrhenius(plt, sens, spot, kmc, out):
    T = np.asarray(sens["T_K"])
    x = 1000.0 / T
    tab = sens["D_cm2_s"]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 5.1), sharey=True)
    for ax, comp in zip(axs, "xy"):
        for m in ("M1", "M0"):
            c = COLOR[m]
            lo, hi = tab[m]["classical_saddles+0.05"][comp], tab[m]["classical_saddles-0.05"][comp]
            ax.fill_between(x, lo, hi, color=c, alpha=0.16, lw=0)
            ax.semilogy(x, tab[m]["classical"][comp], "-", color=c, lw=1.8)
            ax.semilogy(x, tab[m]["fixed_1e13"][comp], "--", color=c, lw=1.0)
            pts = [(cs["T_K"], cs["runs"]["full"][f"D_{comp}"], cs["runs"]["full"][f"se_{comp}"])
                   for cs in kmc["cases"] if cs["model"] == m and cs["variant"] == "classical" and "full" in cs["runs"]]
            if pts:
                tt, dd, ee = map(np.array, zip(*pts))
                ax.errorbar(1000.0 / tt, dd, yerr=ee, fmt="o", color=c, ms=4.5, capsize=2, lw=1, zorder=5)
            for v, mk in (("classical_saddles-0.05", "^"), ("classical_saddles+0.05", "v")):
                sp = [(cs["T_K"], cs["kmc"][f"D_{comp}"], cs["kmc"][f"se_{comp}"]) for cs in spot["cases"]
                      if cs["model"] == m and cs["variant"] == v]
                if sp:
                    tt, dd, ee = map(np.array, zip(*sp))
                    ax.errorbar(1000.0 / tt, dd, yerr=ee, fmt=mk, color=c, mfc="white", ms=5, capsize=2, lw=1, zorder=5)
        ax.set_xlabel("1000 / T (K$^{-1}$)")
        ax.grid(alpha=0.3, which="both", lw=0.5)
        top_T_axis(ax)
        ax.text(0.02, 0.03, f"({'a' if comp == 'x' else 'b'})  $D_{comp}$", transform=ax.transAxes, fontsize=10)
    axs[0].set_ylabel("D (cm$^2$ s$^{-1}$)")
    h = [plt.Line2D([], [], color=COLOR["M1"], lw=1.8, label=LABEL["M1"]),
         plt.Line2D([], [], color=COLOR["M0"], lw=1.8, label=LABEL["M0"]),
         plt.matplotlib.patches.Patch(color="0.5", alpha=0.25, label="saddles ±0.05 eV (judgment-based band)"),
         plt.Line2D([], [], color="0.3", ls="--", lw=1.0, label=r"prefactor fixed at $10^{13}$ s$^{-1}$"),
         plt.Line2D([], [], color="0.3", marker="o", ls="none", label="kMC, classical (T3-02)"),
         plt.Line2D([], [], color="0.3", marker="^", mfc="white", ls="none", label="kMC, saddles −0.05 eV"),
         plt.Line2D([], [], color="0.3", marker="v", mfc="white", ls="none", label="kMC, saddles +0.05 eV")]
    fig.suptitle(TITLE, fontsize=10)
    fig.text(0.5, 0.005, NOTE, ha="center", va="bottom", fontsize=7, color="0.35")
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.legend(handles=h, fontsize=7.5, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.03))
    save(fig, out, "fig_kmc_arrhenius")
    plt.close(fig)


# ----------------------------------------------------------------------------- (b) MSD
def fig_msd(plt, msd, out):
    t = np.asarray(msd["t_s"]) * 1e9
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    for j, (comp, c) in enumerate((("x", "#1f77b4"), ("y", "#d62728"))):
        Dk, se, De = msd["D_kmc_cm2_s"][j], msd["se_cm2_s"][j], msd["D_exact_cm2_s"][j]
        ax.plot(t, msd["msd_A2"][comp], "o", ms=3.5, color=c,
                label=f"kMC ⟨Δ{comp}²⟩: $D_{comp}$ = ({Dk * 1e6:.3f} ± {se * 1e6:.3f})×10$^{{-6}}$ cm²/s")
        ax.plot(t, 2 * De * 1e16 * t * 1e-9, "-", color=c, lw=1.1,
                label=f"exact 2$D_{comp}$t: $D_{comp}$ = {De * 1e6:.3f}×10$^{{-6}}$ cm²/s")
    ax.axvspan(0, msd["fit_from_fraction"] * msd["t_end_s"] * 1e9, color="0.92", zorder=0)
    ax.text(0.01, 0.97, "excluded from\nslope fit", transform=ax.transAxes, va="top", fontsize=7, color="0.45")
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("MSD (Å$^2$)")
    ax.set_xlim(0, t[-1] * 1.02)
    ax.set_ylim(0, None)
    ax.set_title(f"M1 classical, {msd['T_K']:g} K: {msd['n_walkers']} walkers, BKL kMC vs exact lattice D", fontsize=10)
    ax.grid(alpha=0.3, lw=0.5)
    fig.text(0.5, 0.005, NOTE, ha="center", va="bottom", fontsize=6.5, color="0.35")
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    h_, l_ = ax.get_legend_handles_labels()
    order = [0, 2, 1, 3]
    fig.legend([h_[i] for i in order], [l_[i] for i in order], fontsize=7.5, frameon=False, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.03))
    save(fig, out, "fig_kmc_msd")
    plt.close(fig)


# ----------------------------------------------------------------------------- (c) occupations
def fig_occupation(plt, rd, kmc, out):
    Tf = np.arange(200.0, 600.0 + 1e-9, 5.0)
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.6), sharey=True)
    for ax, m in zip(axs, ("M1", "M0")):
        sites = rd["models"][m]["sites"]
        pops = [R.populations(sites, T, "classical") for T in Tf]
        for s in ("B1", "B2", "T"):
            ax.semilogy(Tf, [p[s] for p in pops], "-", color=SITE_COLOR[s], lw=1.4, label=f"{s} Boltzmann")
            for mode, mk, dT in (("full", "o", 0.0), ("cg", "s", 6.0)):
                pts = [(cs["T_K"], cs["runs"][mode]["occupation"][s], cs["runs"][mode]["occupation_se"][s])
                       for cs in kmc["cases"] if cs["model"] == m and cs["variant"] == "classical" and mode in cs["runs"]]
                if pts:
                    tt, oo, ee = map(np.array, zip(*pts))
                    ax.errorbar(tt + dT, oo, yerr=ee, fmt=mk, color=SITE_COLOR[s], mfc=SITE_COLOR[s] if mode == "full" else "white",
                                ms=4.5, capsize=2, lw=1, zorder=5)
        ax.set_xlabel("T (K)")
        ax.set_title(f"({'a' if m == 'M1' else 'b'}) {LABEL[m]}, classical hTST", fontsize=10)
        ax.grid(alpha=0.3, which="both", lw=0.5)
    axs[0].set_ylabel("site occupation")
    h = [plt.Line2D([], [], color=SITE_COLOR[s], lw=1.4, label=f"{s}: Boltzmann") for s in ("B1", "B2", "T")]
    h += [plt.Line2D([], [], color="0.3", marker="o", ls="none", label="kMC, full network"),
          plt.Line2D([], [], color="0.3", marker="s", mfc="white", ls="none", label="kMC, coarse-grained (T eliminated)")]
    fig.suptitle("kMC time-weighted site occupations vs Boltzmann populations (H on NbP, 1 ML barriers)", fontsize=10)
    fig.text(0.5, 0.005, NOTE, ha="center", va="bottom", fontsize=7, color="0.35")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.legend(handles=h, fontsize=7.5, frameon=False, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.035))
    save(fig, out, "fig_kmc_occupation")
    plt.close(fig)


# ----------------------------------------------------------------------------- (d) network
def key_energies(rd, sens):
    e = {}
    for m in ("M1", "M0"):
        md = rd["models"][m]
        st, hp = md["sites"], md["hops"]
        e[m] = {"B1": st["B1"]["E_eV"], "T": st["T"]["E_eV"], "B2": st["B2"]["E_eV"],
                "S1": st["B1"]["E_eV"] + hp["B1->T"]["Ea_eV"], "S2": st["T"]["E_eV"] + hp["T->B2"]["Ea_eV"],
                "S3": sens["direct_route"][m]["E_S3_grid_eV"]}
    return e


def fig_network(plt, rd, sens, out):
    E = key_energies(rd, sens)
    a = rd["lattice"]["a_A"]
    fig = plt.figure(figsize=(8.6, 4.6))
    ax = fig.add_axes([0.02, 0.06, 0.52, 0.86])
    lo, hi = -0.5, 1.5
    c1, c2 = "#8c564b", "#17becf"
    Ts = [(i, j) for i in range(-1, 3) for j in range(-1, 3)]
    for i, j in Ts:                                   # S1 (y) and S2 (x) hops
        for dy in (-0.5, 0.5):
            ax.plot([i, i], [j, j + dy], "-", color=c1, lw=2.2, zorder=1)
            ax.plot(i, j + dy / 2, "x", color=c1, ms=5, mew=1.2, zorder=2)
        for dx in (-0.5, 0.5):
            ax.plot([i, i + dx], [j, j], "-", color=c2, lw=2.2, zorder=1)
            ax.plot(i + dx / 2, j, "x", color=c2, ms=5, mew=1.2, zorder=2)
    b1 = (1.0, 0.5)                                   # optional direct route around one B1
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            ax.plot([b1[0], b1[0] + sx], [b1[1], b1[1] + sy], ":", color="0.45", lw=1.4, zorder=1)
    for i, j in Ts:
        ax.plot(i, j, "o", ms=13, color=SITE_COLOR["T"], mec="k", mew=0.6, zorder=3)
        ax.plot(i, j + 0.5, "s", ms=10, color=SITE_COLOR["B1"], mec="k", mew=0.6, zorder=3)
        ax.plot(i + 0.5, j, "D", ms=9, color=SITE_COLOR["B2"], mec="k", mew=0.6, zorder=3)
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], "--", color="0.2", lw=1.0, zorder=0)
    ax.annotate("", xy=(1, -0.3), xytext=(0, -0.3), arrowprops=dict(arrowstyle="<->", color="0.2", lw=1))
    ax.text(0.5, -0.38, f"a = {a:.3f} Å", ha="center", va="top", fontsize=8.5)
    ax.text(0.05, 0.27, "S1 (y)", color=c1, fontsize=8.5, rotation=90, va="center")
    ax.text(0.25, 0.05, "S2 (x)", color=c2, fontsize=8.5, ha="center")
    ax.text(1.27, 0.78, "direct\n(optional)", color="0.35", fontsize=7.5, ha="left")
    ax.annotate("", xy=(-0.42, 0.25), xytext=(-0.42, -0.25), arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.annotate("", xy=(-0.17, -0.42), xytext=(-0.42, -0.42), arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.text(-0.45, 0.27, "y", fontsize=8)
    ax.text(-0.15, -0.44, "x", fontsize=8, va="center")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.axis("off")
    h = [plt.Line2D([], [], marker="o", ls="none", ms=9, color=SITE_COLOR["T"], mec="k", label="T (atop Nb)"),
         plt.Line2D([], [], marker="s", ls="none", ms=8, color=SITE_COLOR["B1"], mec="k", label="B1 (Nb–Nb bridge, DFT r0)"),
         plt.Line2D([], [], marker="D", ls="none", ms=7, color=SITE_COLOR["B2"], mec="k", label="B2 (Nb–Nb bridge)"),
         plt.Line2D([], [], color=c1, lw=2.2, label="B1 ↔ T via S1 (along y)"),
         plt.Line2D([], [], color=c2, lw=2.2, label="T ↔ B2 via S2 (along x)"),
         plt.Line2D([], [], color="0.45", ls=":", lw=1.4, label="direct B1 ↔ B2 (sensitivity only)"),
         plt.Line2D([], [], color="0.2", ls="--", lw=1.0, label="surface unit cell")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.56, 0.93), fontsize=8, frameon=False)
    rows = [("B1", "B1"), ("B2", "B2"), ("T", "T"), ("S1 (B1↔T)", "S1"), ("S2 (T↔B2)", "S2"), ("direct, grid ≤", "S3")]
    txt = "Energies vs B1 (eV), M1 / M0\n" + "\n".join(f"  {lab:15s} {E['M1'][k]:6.3f} / {E['M0'][k]:6.3f}" for lab, k in rows)
    txt += f"\nhop length a/2 = {a / 2:.3f} Å (direct: {a / np.sqrt(2):.2f} Å)"
    fig.text(0.57, 0.30, txt, fontsize=8, family="monospace", va="top")
    fig.suptitle("Site network for H on Nb-terminated NbP (one T, B1, B2 per cell; 1 ML barriers)", fontsize=10)
    fig.text(0.5, 0.005, NOTE, ha="center", va="bottom", fontsize=7, color="0.35")
    save(fig, out, "fig_kmc_network")
    plt.close(fig)


# ----------------------------------------------------------------------------- captions
def captions(sens, spot, kmc, msd, rd):
    f = sens["fits"]
    b = sens["bands"]
    m1 = f["M1"]["classical"]
    bx = b["M1"]["barrier"]["x"]["at_T"]["300"]
    by = b["M1"]["barrier"]["y"]["at_T"]["300"]
    E = key_energies(rd, sens)
    hedge = ("All values are MLIP-based (MACE M1 fine-tune and M0 = MACE-MP-0-small) harmonic-TST estimates for a "
             "lattice model parameterised with 1 ML collective barriers (1×1 cell, all periodic H move together); they "
             "are not measured or DFT-validated values, and the model is valid only above ~150–200 K (tunnelling "
             "crossover 85–101 K).")
    return f"""# T3-03 figure captions (H surface diffusion on Nb-terminated NbP)

**fig_kmc_arrhenius.** Tracer diffusion coefficients (a) D_x and (b) D_y of one H on the Nb-terminated NbP slab
versus 1000/T, 200–600 K. Lines: exact solution of the lattice model (Bloch master equation) with classical Vineyard
harmonic-TST rates for M1 (blue) and M0 (orange). Shaded: all saddles shifted by ±0.05 eV with the minima fixed, a
judgment-based range for the MLIP barrier error, not a statistical error bar (factor ≈{bx['factor_up']:.0f} each way at 300 K).
Dashed: prefactor fixed at 10¹³ s⁻¹ instead of Vineyard. Filled circles: kMC (BKL, 20 000 walkers) with classical rates
from T3-02; open triangles: kMC spot checks at the band edges. kMC error bars are standard errors from 32 independent
walker batches and are mostly smaller than the markers. M1, 300 K: D_x = {m1['x']['D_300K_cm2_s']:.2e} cm²/s
(band {bx['lower']:.2e}–{bx['upper']:.2e}), D_y = {m1['y']['D_300K_cm2_s']:.2e} cm²/s (band {by['lower']:.2e}–{by['upper']:.2e});
E_a,eff = {m1['x']['Ea_eff_eV']:.3f} / {m1['y']['Ea_eff_eV']:.3f} eV (x / y, 200–600 K fit). {hedge}

**fig_kmc_msd.** Mean-square displacement of H along x and y from BKL kMC (M1, classical rates, {msd['T_K']:g} K,
{msd['n_walkers']} independent walkers, unwrapped coordinates) with the exact lines 2D_x t and 2D_y t. D from the slope over
t ≥ 0.2 t_end (shaded region excluded). The anisotropy D_x/D_y ≈ {m1['Dx_over_Dy_300K']:.1f} follows from the lower saddle S2 on the
x route. This validates the kMC engine against the exact lattice solution only, not the rates themselves. {hedge}

**fig_kmc_occupation.** Time-weighted kMC site occupations (filled circles: full B1/T/B2 network; open squares:
coarse-grained model with T eliminated, shifted by +6 K for visibility) versus Boltzmann populations from the same
harmonic partition functions (lines), (a) M1 and (b) M0, classical rates. Agreement within statistical error confirms
detailed balance of the event table; T stays a minority site (≤ ~11 % at 600 K for M1). {hedge}

**fig_kmc_network.** Site network used in the lattice model: per surface cell (a = {rd['lattice']['a_A']:.3f} Å) one atop site T
and two Nb–Nb bridges B1 (the DFT-anchored r0 site) and B2. B1 ↔ T hops cross saddle S1 along y, T ↔ B2 hops cross S2 along
x (both a/2 long). The dotted direct B1 ↔ B2 route (sensitivity analysis only) uses the straight-line maximum of the
frozen-substrate T1-01 map, an upper estimate of its saddle ({E['M1']['S3']:.3f} / {E['M0']['S3']:.3f} eV for M1 / M0). Energies are
CI-NEB key points relative to B1 (M1 / M0), classical, without ZPE. {hedge}
"""


# ----------------------------------------------------------------------------- main
def main(argv=None):
    stamp = datetime.date.today().isoformat()
    root, root_src = R.portfolio_root(HERE)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sens", help="results.json (default: newest 03_results/<date>_sensitivity/)")
    ap.add_argument("--spot", help="spot_summary.json (default: newest 03_results/<date>_kmc_spot/)")
    ap.add_argument("--kmc", help="kmc_summary.json (default: newest 02_kmc/<date>_kmc/)")
    ap.add_argument("--rates", help="rates.json (default: newest 01_rates/<date>_rates/)")
    ap.add_argument("--out", default=str(HERE.parent / "04_figures" / stamp))
    args = ap.parse_args(argv)
    sp = Path(args.sens) if args.sens else newest(HERE, "_sensitivity", "results.json")
    pp = Path(args.spot) if args.spot else newest(HERE, "_kmc_spot", "spot_summary.json")
    kp = Path(args.kmc) if args.kmc else newest(HERE.parent / "02_kmc", "_kmc", "kmc_summary.json")
    rp = Path(args.rates) if args.rates else K.default_inputs()[0]
    mp = pp.parent / "msd_300K_M1_classical.json"
    sens, spot, kmc, rd = (json.loads(p.read_text()) for p in (sp, pp, kp, rp))
    msd = json.loads(mp.read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches  # noqa: F401
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "pdf.fonttype": 42})
    fig_arrhenius(plt, sens, spot, kmc, out)
    fig_msd(plt, msd, out)
    fig_occupation(plt, rd, kmc, out)
    fig_network(plt, rd, sens, out)
    (out / "CAPTIONS.md").write_text(captions(sens, spot, kmc, msd, rd), encoding="utf-8")
    import scipy
    prov = {"date": stamp, "script": "src/kmc/03_results/make_figures.py", "root": root_src,
            "inputs": {n: {"file": R.portable(p, root), "sha256": R.sha256(p)}
                       for n, p in (("results", sp), ("spot_summary", pp), ("msd", mp), ("kmc_summary", kp), ("rates", rp))},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                         "matplotlib": matplotlib.__version__},
            "note": NOTE}
    (out / "figures_provenance.json").write_text(json.dumps(prov, indent=1))
    for n, p in (("results", sp), ("spot", pp), ("kmc", kp), ("rates", rp)):
        print(f"{n:8}: {R.portable(p, root)}")
    print(f"figures -> {R.portable(out, root)}: fig_kmc_arrhenius, fig_kmc_msd, fig_kmc_occupation, fig_kmc_network "
          "(.png/.pdf), CAPTIONS.md, figures_provenance.json")


if __name__ == "__main__":
    main()
