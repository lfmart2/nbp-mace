# MLIP-driven CI-NEB artifacts

Small derived files from the H-on-NbP pathway study described in
[`docs/NEB_PATHWAY.md`](../../docs/NEB_PATHWAY.md). Every number is an MLIP
estimate from M0 (MACE-MP-0-small) or M1 (five-epoch fine-tune, SHA-256
fa762686…cf8d), not DFT. Coordinate-bearing files (`*.extxyz`, `*.traj`) and
model checkpoints stay local. Local filesystem paths in run summaries were
replaced with portable `local://<name>` identifiers before commit.

## `01_endpoints/` (generated 2026-10-09)

- `relax_r0/`: M0/M1 relaxation of the DFT r0 structure (H height, energies).
- `substrate_check/`: M1 substrate response and top-layer forces versus DFT.
- `h_modes/`: H-only finite-difference modes at r0, M0/M1 versus collinear DFT.
- `pes_z/`: dense vertical H scan above B1 with both models.
- `pes_xy/`: 12×12 frozen-substrate lateral map (H height relaxed per point).
- `endpoints/`: free-substrate relaxations of B1, B2, and T.
  `endpoints_summary.json` still carries the old key
  `H_lateral_drift_from_site_A`, which was measured in absolute coordinates and
  reflects substrate re-centring, not H leaving its site. Relative to the
  surface Nb, H returns to within 0.01 Å of each site.

## `02_neb_runs/` (generated 2026-10-10)

- `runs/<model>_<images>_<band>/summary.json`: converged split bands
  (B1T = B1 → T, TB2 = T → B2) for M1 with 9 images and M0 with 7 images, plus
  the M1 7-image sensitivity runs.
- `compare/`: M0/M1 energy profile, comparison table, and image-count
  sensitivity (`sensitivity.csv`).

## `03_analysis/` and `figures/` (generated 2026-10-10)

- `path_geometry.csv`: per-image energy, H height, and H--Nb distances along
  the stitched B1 → T → B2 path.
- `key_points.csv/.json`: B1, S1, T, S2, and B2 energies, H-only modes, and
  H-only ZPE, with input and checkpoint hashes.
- `figures/fig_neb_path.*`: three-panel path figure (energy, H height,
  H--Nb distances). `figures/fig_neb_energy.*`: the energy panel alone.
