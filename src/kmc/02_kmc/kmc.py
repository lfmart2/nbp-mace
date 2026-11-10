"""T3-02: rejection-free kinetic Monte Carlo of one H walker on the periodic B1/T/B2 site network.

Algorithm: BKL / Gillespie residence-time method. From site s with total escape rate R_s, the walker
waits dt = -ln(u1)/R_s and then takes event e with probability k_e/R_s. Every step is an accepted move
(no rejections). Many independent walkers are advanced together (vectorised NumPy); each keeps its own
clock and unwrapped 2D position, so displacements are never folded back into the cell.

Two kinetic models are run from the same rates:
  full    the three-site network from rates.json (lattice.events; 8 events per cell).
  cg      coarse-grained: T eliminated. From a bridge the walker leaves to a neighbouring T and then goes
          on with the splitting probabilities P(T->B1), P(T->B2). Returns to the starting bridge are
          lumped out, so only productive bridge-to-bridge events remain:
              B1 -> B2 at (+-a/2, +-a/2): rate k(B1->T) P_B2/2 each   (4 events)
              B1 -> B1 at (0, +-a):       rate k(B1->T) P_B1/2 each   (2 events)
              B2 -> B1 at (+-a/2, +-a/2): rate k(B2->T) P_B1/2 each
              B2 -> B2 at (+-a, 0):       rate k(B2->T) P_B2/2 each
          with P_Bi = k(T->Bi)/[k(T->B1)+k(T->B2)]. The mean time spent on T (1/R_T per visit, 1/f visits
          per productive event, f = 1 - P_B/2 the productive fraction of bridge exits) is added to the
          clock after each event, so the long-time D is exact, not shifted by the T population.

Outputs (main): D_x, D_y from the slope of MSD(t) (late-time window), with standard errors from
independent batches of walkers; time-weighted site occupations; per-site residence-time statistics
(mean, 2nd moment, KS distance against the exponential with mean 1/R_s); event counts and wall time.
Reference values: exact D from exact_D.json (T3-01) and the closed form D_x = (a^2/2) p_T k(T->B2),
D_y = (a^2/2) p_T k(T->B1).

Writes to <out>/ (default 02_kmc/<date>_kmc/):
  kmc_D.csv               one row per (model, variant, T, mode): D, SE, exact, z, occupations, timing
  kmc_residence.csv       residence-time statistics per case and site
  kmc_summary.json        provenance (rates.json / exact_D.json SHA-256, seed, n_walkers, versions), cases,
                          coarse-grained vs full speed-up
  msd_300K_M1_classical.png   quick-look MSD(t), x and y, with the exact 2Dt lines

All physical numbers are MLIP-based harmonic-TST estimates for a lattice model parameterised with 1 ML
collective barriers (not DFT-validated). The engine itself is validated against the exact lattice solution
(test_kmc.py). Needs NumPy and Matplotlib; SciPy only for temperatures off the rates.json grid (via rates.py).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SITES = ["T", "B1", "B2"]                     # site index order used everywhere in this module
HOPS = ["B1->T", "T->B1", "T->B2", "B2->T"]
DEFAULT_SEED = 20261109


# ----------------------------------------------------------------------------- paths / provenance
def portfolio_root():
    """Repository root: this script lives in <repo>/src/kmc/02_kmc/. Recorded paths are relative to it."""
    return HERE.parents[2], "repo-relative (nbp-mace)"


def portable(path, root):
    """Path relative to the portfolio root (forward slashes), else local://<name>. Never absolute."""
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return f"local://{Path(path).name}"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def newest(pattern_dir, suffix, fname):
    """Newest <date><suffix>/<fname> under pattern_dir (dated folders sort lexically)."""
    cands = sorted(p for p in Path(pattern_dir).glob(f"*{suffix}") if (p / fname).exists())
    if not cands:
        raise FileNotFoundError(f"no *{suffix}/{fname} under {pattern_dir}")
    return cands[-1] / fname


def default_inputs():
    root, _ = portfolio_root()
    rates_dir = HERE.parent / "01_rates"
    if not rates_dir.exists():
        rates_dir = HERE.parent / "01_rates"
    return newest(rates_dir, "_rates", "rates.json"), newest(rates_dir, "_exact_D", "exact_D.json")


def versions():
    v = {"python": platform.python_version(), "numpy": np.__version__}
    for mod in ("matplotlib", "scipy"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            v[mod] = None
    return v


# ----------------------------------------------------------------------------- rates
class RateBook:
    """Access to rates.json (and optionally exact_D.json) for any model, variant and temperature."""

    def __init__(self, rates_path, exact_path=None):
        self.rates_path = Path(rates_path)
        self.R = json.loads(self.rates_path.read_text())
        self.exact_path = Path(exact_path) if exact_path else None
        self.E = json.loads(self.exact_path.read_text()) if self.exact_path else None
        self.T_grid = np.asarray(self.R["T_K"], float)
        self.a_A = float(self.R["lattice"]["a_A"])
        self.events = self.R["lattice"]["events"]
        self.kB = float(self.R["provenance"]["constants"]["kB_eV_K"])

    def _grid_index(self, T):
        i = int(np.argmin(abs(self.T_grid - T)))
        return i if abs(self.T_grid[i] - T) < 1e-6 else None

    def rates(self, model, variant, T):
        """{hop: k (s^-1)}. Grid temperatures from the table; others recomputed with rates.py (needs SciPy)."""
        i = self._grid_index(T)
        if i is not None:
            tab = self.R["models"][model]["rates_s-1"][variant]
            return {h: float(tab[h][i]) for h in HOPS}
        R = _import_rates_module()
        hp = self.R["models"][model]["hops"]
        return {h: R.rate(hp[h], float(T), variant) for h in HOPS}

    def populations(self, model, variant, T):
        i = self._grid_index(T)
        if i is not None:
            tab = self.R["models"][model]["populations"][variant]
            return {s: float(tab[s][i]) for s in SITES}
        R = _import_rates_module()
        return R.populations(self.R["models"][model]["sites"], float(T), variant)

    def exact_D(self, model, variant, T):
        """(D_x, D_y) in cm^2/s from exact_D.json at a grid temperature, else None."""
        if self.E is None:
            return None
        i = self._grid_index(T)
        if i is None or abs(np.asarray(self.E["T_K"])[i] - T) > 1e-6:
            return None
        d = self.E["D_cm2_s"][model][variant]
        return float(d["x"][i]), float(d["y"][i])


def _import_rates_module():
    sys.path.insert(0, str(HERE.parent / "01_rates"))
    import rates as R  # noqa: E402
    return R


def tree_populations(k):
    """Stationary populations of the star network B1-T-B2 from the rates themselves (detailed balance holds
    on any tree). A bridge whose in- or out-rate is zero is unreachable and gets weight 0."""
    w = {"T": 1.0}
    for b in ("B1", "B2"):
        kin, kout = k[f"T->{b}"], k[f"{b}->T"]
        w[b] = kin / kout if (kin > 0 and kout > 0) else 0.0
    z = sum(w.values())
    return {s: w[s] / z for s in SITES}


def closed_form_D(k, a_A):
    """Exact D for this network: D_x = (a^2/2) p_T k(T->B2), D_y = (a^2/2) p_T k(T->B1); cm^2/s."""
    p = tree_populations(k)
    a2 = (a_A * 1e-8) ** 2
    return 0.5 * a2 * p["T"] * k["T->B2"], 0.5 * a2 * p["T"] * k["T->B1"]


# ----------------------------------------------------------------------------- networks
def _pack(site_events, tau_extra=None, extra_name=None):
    """site_events: {site: [(to, (dx, dy)/a, rate), ...]}. Zero-rate events are dropped. Returns arrays."""
    n = len(SITES)
    clean = {s: [e for e in site_events.get(s, []) if e[2] > 0] for s in SITES}
    kmax = max(1, max(len(v) for v in clean.values()))
    Rtot = np.zeros(n)
    cum = np.full((n, kmax), 2.0)                 # padding never selected (u < 1)
    to = np.tile(np.arange(n)[:, None], (1, kmax))
    disp = np.zeros((n, kmax, 2))
    rates = np.zeros((n, kmax))
    for i, s in enumerate(SITES):
        ev = clean[s]
        if not ev:
            continue
        r = np.array([e[2] for e in ev], float)
        Rtot[i] = r.sum()
        c = np.cumsum(r) / Rtot[i]
        c[-1] = 1.0
        cum[i, :len(ev)] = c
        to[i, :len(ev)] = [SITES.index(e[0]) for e in ev]
        disp[i, :len(ev)] = [e[1] for e in ev]
        rates[i, :len(ev)] = r
    return {"Rtot": Rtot, "cum": cum, "to": to, "disp": disp, "rates": rates,
            "tau_extra": np.zeros(n) if tau_extra is None else np.asarray(tau_extra, float),
            "extra_name": extra_name}


def build_full(events, k):
    """Full three-site network from the rates.json event table."""
    se = {}
    for e in events:
        se.setdefault(e["from"], []).append((e["to"], tuple(e["d_over_a"]), k[e["hop"]]))
    net = _pack(se)
    net["mode"] = "full"
    return net


def cg_parameters(k):
    """Splitting probabilities from T, productive fractions and mean T dwell per productive event."""
    kt = k["T->B1"] + k["T->B2"]
    if kt <= 0:
        raise ValueError("T has no exits")
    P = {"B1": k["T->B1"] / kt, "B2": k["T->B2"] / kt}
    tau_T = 1.0 / (2.0 * kt)                     # mean residence on T (two neighbours of each kind)
    f = {b: 1.0 - P[b] / 2.0 for b in ("B1", "B2")}   # productive fraction of exits from bridge b
    return P, tau_T, f


def build_cg(k):
    """Coarse-grained network on the bridges only (T eliminated); see module docstring."""
    P, tau_T, f = cg_parameters(k)
    h = 0.5
    se = {
        "B1": [("B2", (sx * h, sy * h), k["B1->T"] * P["B2"] / 2) for sx in (1, -1) for sy in (1, -1)]
        + [("B1", (0.0, sy * 1.0), k["B1->T"] * P["B1"] / 2) for sy in (1, -1)],
        "B2": [("B1", (sx * h, sy * h), k["B2->T"] * P["B1"] / 2) for sx in (1, -1) for sy in (1, -1)]
        + [("B2", (sx * 1.0, 0.0), k["B2->T"] * P["B2"] / 2) for sx in (1, -1)],
    }
    tau = np.zeros(len(SITES))
    tau[SITES.index("B1")] = tau_T / f["B1"]
    tau[SITES.index("B2")] = tau_T / f["B2"]
    net = _pack(se, tau_extra=tau, extra_name="T")
    net["mode"] = "cg"
    return net


# ----------------------------------------------------------------------------- engine
def simulate(net, p0, n_walkers, t_end, n_samples, rng, reservoir=50_000):
    """Advance n_walkers independent walkers to t_end with the BKL algorithm.

    p0: initial site probabilities (length 3, SITES order). Positions are recorded at n_samples equally
    spaced times in (0, t_end] (position held during each wait, i.e. the pre-jump position).
    Occupation time is accumulated per walker and site; the T dwell of the cg model goes to its own column.
    Residence times: every drawn wait (count, sum, sum of squares, plus the first `reservoir` values per site).
    """
    n_sites = len(SITES)
    Rtot, cum, to, disp, tau = net["Rtot"], net["cum"], net["to"], net["disp"], net["tau_extra"]
    N = int(n_walkers)
    t_s = t_end * np.arange(1, n_samples + 1) / n_samples
    site = rng.choice(n_sites, size=N, p=np.asarray(p0) / np.sum(p0))
    pos = np.zeros((N, 2))
    t = np.zeros(N)
    nxt = np.zeros(N, dtype=np.int64)
    samples = np.zeros((n_samples, N, 2))
    occ = np.zeros((N, n_sites + 1))             # last column: T dwell (cg only)
    n_ev = np.zeros(N, dtype=np.int64)
    res_n = np.zeros(n_sites, dtype=np.int64)
    res_s1 = np.zeros(n_sites)
    res_s2 = np.zeros(n_sites)
    res_keep = [[] for _ in range(n_sites)]
    res_kept = np.zeros(n_sites, dtype=np.int64)
    active = np.arange(N)
    while active.size:
        s = site[active]
        R = Rtot[s]
        e1 = rng.standard_exponential(active.size)
        w = np.divide(e1, R, out=np.full(active.size, np.inf), where=R > 0)
        t0 = t[active]
        t_w = t0 + w                               # end of the wait on the site
        t_n = t_w + tau[s]                         # end of the event incl. any T dwell (cg)
        # positions at sample times inside [t0, t_n)
        kk = nxt[active]
        while True:
            m = kk < n_samples
            m[m] = t_s[kk[m]] < t_n[m]
            if not m.any():
                break
            idx = active[m]
            samples[kk[m], idx] = pos[idx]
            kk[m] += 1
        nxt[active] = kk
        # occupation (clipped at t_end)
        cw = np.minimum(t_w, t_end)
        occ[active, s] += cw - t0
        if net["extra_name"]:
            occ[active, n_sites] += np.minimum(t_n, t_end) - cw
        # residence statistics: every drawn wait is a full Exp(R_s) sample. Keeping only waits that end
        # before t_end would censor long waits and bias the mean low, so all finite draws are recorded.
        done = np.isfinite(w)
        if done.any():
            sd, wd = s[done], w[done]
            res_n += np.bincount(sd, minlength=n_sites)
            res_s1 += np.bincount(sd, weights=wd, minlength=n_sites)
            res_s2 += np.bincount(sd, weights=wd * wd, minlength=n_sites)
            for i in range(n_sites):
                if res_kept[i] < reservoir:
                    v = wd[sd == i][: reservoir - res_kept[i]]
                    if v.size:
                        res_keep[i].append(v)
                        res_kept[i] += v.size
        # event selection and jump
        u = rng.random(active.size)
        ev = (cum[s] < u[:, None]).sum(axis=1)
        pos[active] += disp[s, ev]
        site[active] = to[s, ev]
        n_ev[active] += (t_w <= t_end) & np.isfinite(w)
        t[active] = t_n
        active = active[(t_n < t_end) | (nxt[active] < n_samples)]
    return {"t_samples": t_s, "samples": samples, "occ": occ, "n_events": n_ev,
            "res_n": res_n, "res_s1": res_s1, "res_s2": res_s2,
            "res_values": [np.concatenate(v) if v else np.zeros(0) for v in res_keep]}


# ----------------------------------------------------------------------------- analysis
def ks_exponential(x, mean):
    """KS distance between sample x and Exp(mean); asymptotic p-value (Kolmogorov series)."""
    n = x.size
    if n == 0:
        return float("nan"), float("nan")
    xs = np.sort(x)
    F = 1.0 - np.exp(-xs / mean)
    i = np.arange(1, n + 1)
    d = float(max((i / n - F).max(), (F - (i - 1) / n).max()))
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
    j = np.arange(1, 101)
    p = float(np.clip(2.0 * np.sum((-1.0) ** (j - 1) * np.exp(-2.0 * j * j * lam * lam)), 0.0, 1.0))
    return d, p


def analyse(sim, net, a_A, n_batches=32, fit_from=0.2):
    """D_x, D_y (cm^2/s) from batch MSD slopes; occupations; residence statistics."""
    a2 = (a_A * 1e-8) ** 2
    t_s, X = sim["t_samples"], sim["samples"]
    N = X.shape[1]
    nb = max(2, min(n_batches, N))
    b_of = np.arange(N) % nb
    sel = t_s >= fit_from * t_s[-1]

    def slope(y):
        return np.polyfit(t_s[sel], y[sel], 1)[0]

    out = {}
    msd = {}
    for j, ax in enumerate("xy"):
        sq = X[:, :, j] ** 2
        msd[ax] = sq.mean(axis=1)
        per_b = np.array([slope(sq[:, b_of == b].mean(axis=1)) for b in range(nb)]) / 2.0 * a2
        out[f"D_{ax}"] = float(slope(msd[ax]) / 2.0 * a2)
        out[f"se_{ax}"] = float(per_b.std(ddof=1) / np.sqrt(nb))
        out[f"_batch_{ax}"] = per_b
    diff = out["_batch_x"] - out["_batch_y"]
    out["se_diff"] = float(diff.std(ddof=1) / np.sqrt(nb))
    out["msd_A2"] = {ax: (msd[ax] * a_A ** 2).tolist() for ax in "xy"}
    # occupations
    occ = sim["occ"]
    tot = occ.sum(axis=1, keepdims=True)
    frac = occ / tot
    names = SITES + ["T_dwell"]
    occ_b = np.array([frac[b_of == b].mean(axis=0) for b in range(nb)])
    cols = {s: i for i, s in enumerate(names)}
    occ_mean = frac.mean(axis=0)
    occ_se = occ_b.std(axis=0, ddof=1) / np.sqrt(nb)
    occupation, occupation_se = {}, {}
    for s in SITES:
        i = cols[s]
        m, se = occ_mean[i], occ_se[i]
        if s == "T" and net["extra_name"] == "T":   # cg: T time is the dwell column
            m, se = occ_mean[cols["T_dwell"]], occ_se[cols["T_dwell"]]
        occupation[s], occupation_se[s] = float(m), float(se)
    out["occupation"], out["occupation_se"] = occupation, occupation_se
    # residence
    res = {}
    for i, s in enumerate(SITES):
        n = int(sim["res_n"][i])
        if n == 0 or net["Rtot"][i] == 0:
            continue
        mu_th = 1.0 / net["Rtot"][i]
        m1 = sim["res_s1"][i] / n
        m2 = sim["res_s2"][i] / n
        d_ks, p_ks = ks_exponential(sim["res_values"][i], mu_th)
        res[s] = {"n": n, "mean_s": float(m1), "mean_theory_s": float(mu_th),
                  "z_mean": float((m1 - mu_th) / (mu_th / np.sqrt(n))),
                  "m2_over_theory": float(m2 / (2 * mu_th ** 2)),
                  "z_m2": float((m2 - 2 * mu_th ** 2) / (np.sqrt(20.0 / n) * mu_th ** 2)),
                  "cv": float(np.sqrt(max(m2 - m1 * m1, 0.0)) / m1),
                  "ks_D": d_ks, "ks_p": p_ks, "ks_n": int(sim["res_values"][i].size)}
    out["residence"] = res
    ne = sim["n_events"]
    out["events_per_walker"] = float(ne.mean())
    out["events_total"] = int(ne.sum())
    return out


def t_end_for(k, p_full, n_events):
    """Simulated time giving ~n_events full-model events per walker (stationary event rate sum_s p_s R_s)."""
    R = {"T": 2 * (k["T->B1"] + k["T->B2"]), "B1": 2 * k["B1->T"], "B2": 2 * k["B2->T"]}
    rate = sum(p_full[s] * R[s] for s in SITES)
    return n_events / rate


def run_case(k, a_A, p0, mode, n_walkers, t_end, seed, events=None, n_samples=50, n_batches=32):
    """Build the network for `mode`, simulate and analyse. k: {hop: rate}. Returns a summary dict."""
    if mode == "full":
        if events is None:
            raise ValueError("full mode needs the lattice event table")
        net = build_full(events, k)
        p_init = [p0[s] for s in SITES]
    elif mode == "cg":
        net = build_cg(k)
        pb = p0["B1"] + p0["B2"]
        p_init = [0.0, p0["B1"] / pb, p0["B2"] / pb]
    else:
        raise ValueError(mode)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    sim = simulate(net, p_init, n_walkers, t_end, n_samples, rng)
    wall = time.perf_counter() - t0
    res = analyse(sim, net, a_A, n_batches=n_batches)
    res.update({"mode": mode, "seed": int(seed), "n_walkers": int(n_walkers), "t_end_s": float(t_end),
                "n_samples": int(n_samples), "n_batches": int(n_batches), "wall_s": float(wall),
                "t_samples_s": sim["t_samples"].tolist()})
    return res


def compare(res, ref, prefix):
    """Add z-scores of res D_x, D_y against a reference pair (cm^2/s)."""
    for j, ax in enumerate("xy"):
        se = res[f"se_{ax}"]
        res[f"{prefix}_{ax}"] = ref[j]
        res[f"z_{prefix}_{ax}"] = float((res[f"D_{ax}"] - ref[j]) / se) if se > 0 else (
            0.0 if res[f"D_{ax}"] == ref[j] else float("inf"))
    return res


# ----------------------------------------------------------------------------- plot
def msd_plot(res, Dx_ex, Dy_ex, png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.asarray(res["t_samples_s"]) * 1e9                    # ns
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=150)
    for a_, D, c, lab in (("x", Dx_ex, "C0", "x"), ("y", Dy_ex, "C3", "y")):
        ax.plot(t, res["msd_A2"][a_], "o", ms=3, color=c, label=f"kMC ⟨Δ{lab}²⟩")
        ax.plot(t, 2 * D * 1e16 * t * 1e-9, "-", color=c, lw=1,
                label=f"exact 2D$_{lab}$t, D$_{lab}$ = {D:.3g} cm²/s")
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("MSD (Å²)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.text(0.99, 0.02, "MLIP-based hTST rates, 1 ML barriers; quick look",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="0.4")
    fig.tight_layout()
    fig.savefig(png)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rates", help="rates.json (default: newest 01_rates/<date>_rates/rates.json)")
    ap.add_argument("--exact", help="exact_D.json (default: newest 01_rates/<date>_exact_D/exact_D.json)")
    ap.add_argument("--models", nargs="+", default=["M1", "M0"])
    ap.add_argument("--variants", nargs="+", default=["classical"])
    ap.add_argument("--temps", nargs="+", type=float, default=[200, 300, 400, 500, 600])
    ap.add_argument("--modes", nargs="+", default=["full", "cg"], choices=["full", "cg"])
    ap.add_argument("--n-walkers", type=int, default=20000)
    ap.add_argument("--n-events", type=float, default=2000, help="full-model events per walker (sets t_end)")
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--n-batches", type=int, default=32)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", help="output folder (default: 02_kmc/<date>_kmc)")
    args = ap.parse_args(argv)

    root, root_src = portfolio_root()
    r_def, e_def = default_inputs()
    rates_path = Path(args.rates) if args.rates else r_def
    exact_path = Path(args.exact) if args.exact else e_def
    book = RateBook(rates_path, exact_path)
    date = datetime.date.today().isoformat()
    out = Path(args.out) if args.out else HERE / f"{date}_kmc"
    out.mkdir(parents=True, exist_ok=True)
    print(f"rates : {portable(rates_path, root)}\nexact : {portable(exact_path, root)}\nout   : {portable(out, root)}")

    cases, rows, rrows = [], [], []
    seed_seq = np.random.SeedSequence(args.seed)
    n_cases = len(args.models) * len(args.variants) * len(args.temps)
    child = iter(seed_seq.spawn(n_cases * len(args.modes)))
    png_done = False
    for model in args.models:
        for variant in args.variants:
            for T in args.temps:
                k = book.rates(model, variant, T)
                p_full = book.populations(model, variant, T)
                t_end = t_end_for(k, p_full, args.n_events)
                cf = closed_form_D(k, book.a_A)
                ex = book.exact_D(model, variant, T)
                per_mode = {}
                for mode in args.modes:
                    ss = next(child)
                    seed = int(ss.generate_state(1)[0])
                    res = run_case(k, book.a_A, p_full, mode, args.n_walkers, t_end, seed, book.events,
                                   args.n_samples, args.n_batches)
                    compare(res, cf, "closed")
                    if ex is not None:
                        compare(res, ex, "exact")
                    per_mode[mode] = res
                    zx = res.get("z_exact_x", res["z_closed_x"])
                    zy = res.get("z_exact_y", res["z_closed_y"])
                    print(f"{model} {variant:16s} {T:5.0f} K {mode:4s}  D_x {res['D_x']:.3e}±{res['se_x']:.1e} "
                          f"(z {zx:+.2f})  D_y {res['D_y']:.3e}±{res['se_y']:.1e} (z {zy:+.2f})  "
                          f"{res['events_per_walker']:.0f} ev/walker  {res['wall_s']:.1f} s")
                    occ = res["occupation"]
                    rows.append({"model": model, "variant": variant, "T_K": T, "mode": mode,
                                 "D_x_cm2_s": res["D_x"], "se_x": res["se_x"],
                                 "D_y_cm2_s": res["D_y"], "se_y": res["se_y"],
                                 "exact_D_x": ex[0] if ex else "", "exact_D_y": ex[1] if ex else "",
                                 "z_exact_x": res.get("z_exact_x", ""), "z_exact_y": res.get("z_exact_y", ""),
                                 "closed_D_x": cf[0], "closed_D_y": cf[1],
                                 "z_closed_x": res["z_closed_x"], "z_closed_y": res["z_closed_y"],
                                 "occ_T": occ["T"], "occ_B1": occ["B1"], "occ_B2": occ["B2"],
                                 "pop_T": p_full["T"], "pop_B1": p_full["B1"], "pop_B2": p_full["B2"],
                                 "events_per_walker": res["events_per_walker"], "wall_s": res["wall_s"],
                                 "n_walkers": res["n_walkers"], "t_end_s": res["t_end_s"], "seed": res["seed"]})
                    for s, r in res["residence"].items():
                        rrows.append({"model": model, "variant": variant, "T_K": T, "mode": mode, "site": s, **r})
                    if (not png_done and mode == "full" and model == "M1" and variant == "classical"
                            and abs(T - 300) < 1e-6):
                        Dex = ex if ex else cf
                        msd_plot(res, Dex[0], Dex[1], out / "msd_300K_M1_classical.png",
                                 f"M1 classical, 300 K: {res['n_walkers']} walkers, BKL kMC vs exact lattice D")
                        png_done = True
                speed = None
                if "full" in per_mode and "cg" in per_mode:
                    f_, c_ = per_mode["full"], per_mode["cg"]
                    speed = {"wall_ratio": f_["wall_s"] / c_["wall_s"],
                             "event_ratio": f_["events_per_walker"] / c_["events_per_walker"]}
                    for ax in "xy":
                        speed[f"z_cg_vs_full_{ax}"] = float((c_[f"D_{ax}"] - f_[f"D_{ax}"]) /
                                                            np.hypot(c_[f"se_{ax}"], f_[f"se_{ax}"]))
                    print(f"      cg vs full: {speed['wall_ratio']:.2f}x wall, {speed['event_ratio']:.2f}x events, "
                          f"z_x {speed['z_cg_vs_full_x']:+.2f}, z_y {speed['z_cg_vs_full_y']:+.2f}")
                slim = {m: {kk: v for kk, v in r.items() if not kk.startswith("_") and kk not in
                            ("msd_A2", "t_samples_s")} for m, r in per_mode.items()}
                cases.append({"model": model, "variant": variant, "T_K": T, "rates_s-1": k,
                              "populations_boltzmann": p_full, "closed_form_D": list(cf),
                              "exact_D": list(ex) if ex else None, "runs": slim, "cg_vs_full": speed})

    with open(out / "kmc_D.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out / "kmc_residence.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rrows[0].keys()))
        w.writeheader()
        w.writerows(rrows)
    zs = [abs(r[c]) for r in rows for c in ("z_exact_x", "z_exact_y") if r[c] != ""]
    summary = {
        "provenance": {
            "date": date, "script": "src/kmc/02_kmc/kmc.py", "root": root_src,
            "input": {"rates": {"file": portable(rates_path, root), "sha256": sha256(rates_path)},
                      "exact_D": {"file": portable(exact_path, root), "sha256": sha256(exact_path)}},
            "upstream": book.R["provenance"], "versions": versions(),
            "seed": args.seed, "seed_note": "per-case seeds spawned from numpy SeedSequence(seed) in loop order",
            "n_walkers": args.n_walkers, "n_events_target": args.n_events, "n_samples": args.n_samples,
            "n_batches": args.n_batches, "fit_window": "MSD slope over t >= 0.2 t_end",
            "note": ("Engine validated against the exact lattice solution; physical numbers are MLIP-based "
                     "harmonic-TST estimates for a lattice model parameterised with 1 ML collective barriers, "
                     "not DFT-validated values."),
        },
        "max_abs_z_vs_exact": max(zs) if zs else None,
        "cases": cases,
    }
    (out / "kmc_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"max |z| vs exact_D.json: {summary['max_abs_z_vs_exact']:.2f}   -> {portable(out, root)}")
    return summary


if __name__ == "__main__":
    main()
