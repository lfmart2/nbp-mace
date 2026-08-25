# Measured artifacts

These small files are generated from local VASP outputs by `src/audit_scf.py`
and are safe to version. Raw VASP files are not distributed.

- `scf_audit.json`: machine-readable completion, method, force, energy, and
  provenance-hash audit for the clean slab, r0-r8, and H2.
- `zscan_manifest.csv`: compact table used by the DFT and MACE comparisons.
- `dft_zscan.svg`: DFT-only constrained H-height curve generated without
  plotting dependencies.
- `mace_zeroshot.json`: full zero-shot metrics, environment provenance,
  checkpoint hash, timing, and point-level predictions.
- `mace_zeroshot.csv`: compact point-level DFT/MACE comparison.
- `mace_zeroshot.png`: portfolio figure comparing relative energies and H
  vertical forces.

Public source identifiers are portable (`raw://...`). SHA-256 hashes connect
each record to the private raw file without exposing a local filesystem path.

