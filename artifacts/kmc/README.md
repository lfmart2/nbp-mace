# hTST + kMC diffusion artifacts

Small derived files from the H surface diffusion study described in
[`docs/KMC_DIFFUSION.md`](../../docs/KMC_DIFFUSION.md). Every number is an
MLIP-based, harmonic-TST, lattice-kMC estimate for the Nb-terminated NbP slab
with 1 ML collective barriers (M1 = five-epoch fine-tune, SHA-256
fa762686…cf8d; M0 = MACE-MP-0-small), not a measured or DFT-validated value.
The ±0.05 eV band is a judgment-based range, not a statistical error bar.
Generated 2026-11-09 with Python 3.12.13, NumPy 2.5.2, SciPy 1.18.1, and
Matplotlib 3.11.1; no coordinates, checkpoints, or local paths are included.

## `01_rates/`

- `rates/rates.csv`, `rates.json`: harmonic-TST rates for the four hops
  (B1 → T, T → B1, T → B2, B2 → T), five variants (classical Vineyard,
  quantum-harmonic, ± Wigner, fixed 10¹³ s⁻¹), M1 and M0, 200–600 K in 10 K
  steps; the lattice event table; Boltzmann site populations; checks.
- `exact_D/exact_D.csv`, `exact_D.json`: exact (Bloch master-equation) D_x and
  D_y for every variant and temperature, Arrhenius fits, and cross-checks
  (finite differences, closed form).

## `02_kmc/`

- `kmc/kmc_D.csv`: production BKL kMC (20 000 walkers, ~2000 events each),
  M1 and M0, classical, 200–600 K, full and coarse-grained models: D ± SE,
  exact D, z-scores, occupations, timing, seeds.
- `kmc/kmc_residence.csv`: residence-time statistics per site.
- `kmc/kmc_summary.json`: provenance, cases, and coarse-grained speed-up.
- `tests/test_report.csv`, `test_report.json`: the five validation checks
  (occupations, residence times, exact D, coarse-grained = full, limits).

## `03_results/`

- `sensitivity/D_variants.csv`, `fits.csv`, `band.csv`, `results.json`: exact
  D for 22 model × variant sets, Arrhenius fits, the barrier and
  barrier × prefactor bands, band drivers, and the direct B1 ↔ B2 route.
- `kmc_spot/spot_D.csv`, `spot_summary.json`: kMC spot checks of every
  variant at 200, 300, and 600 K (66 cases, all |z| < 4).
- `kmc_spot/msd_300K_M1_classical.json`: MSD(t) data for `fig_kmc_msd`.

## `figures/`

`fig_kmc_arrhenius`, `fig_kmc_msd`, `fig_kmc_occupation`, and
`fig_kmc_network` as PNG (300 dpi) and PDF, with `CAPTIONS.md` and
`figures_provenance.json`.

## Provenance paths and hashes

These files were produced in the author's private workspace before being
copied here byte for byte. Paths recorded inside them are relative to that
workspace and map to this repository as follows:

| Recorded path | In this repository |
|---|---|
| `25_track3_kmc/01_rates/2026-11-09_rates/` | `artifacts/kmc/01_rates/rates/` |
| `25_track3_kmc/01_rates/2026-11-09_exact_D/` | `artifacts/kmc/01_rates/exact_D/` |
| `25_track3_kmc/02_kmc/2026-11-09_kmc/` | `artifacts/kmc/02_kmc/kmc/` |
| `25_track3_kmc/03_results/2026-11-09_sensitivity/` | `artifacts/kmc/03_results/sensitivity/` |
| `25_track3_kmc/03_results/2026-11-09_kmc_spot/` | `artifacts/kmc/03_results/kmc_spot/` |
| `25_track3_kmc/<step>/<script>.py` | `src/kmc/<step>/<script>.py` |
| `20_track1_neb/03_analysis/2026-10-10_path_analysis/key_points.json` | `artifacts/neb/03_analysis/key_points.json` |
| `20_track1_neb/01_endpoints/2026-10-09_pes_xy/xy_map_M*.csv` | `artifacts/neb/01_endpoints/pes_xy/xy_map_M*.csv` |

The copies are byte-identical, and `.gitattributes` marks them `-text` so Git
stores them without line-ending conversion. Every SHA-256 recorded for a
Track 3 file can therefore be checked against the file here on any platform.
The Track 1 inputs are stored with normalised line endings: the two lateral
maps match their recorded hashes in a Windows checkout (CRLF) and match in
content elsewhere. The committed `key_points.json` holds the same JSON content
as the file the rates were built from (SHA-256 ad542994…2a99), but its bytes
differ because the Track 1 staging step re-serialised it; a rerun from this
repository records the committed file's own hash.
