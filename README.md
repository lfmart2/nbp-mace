# Zero-shot MACE-MP-0 benchmark for H on NbP

This repository measures whether the pretrained MACE-MP-0 interatomic
potential transfers to hydrogen interacting with an NbP surface. The primary
test is a nine-point, fixed-composition DFT scan in which the substrate is
frozen and one H atom is displaced along `z`.

The project is deliberately baseline-first. It does not claim that MACE is
accurate, that fine-tuning helps, or that the scan is a desorption barrier.
Those conclusions are limited to the measured results reported below.

![DFT constrained H-height scan](artifacts/dft_zscan.svg)

## Why this is a useful MLIP case study

The atomic geometries originated from SOC/noncollinear relaxations. The
reference energies and forces used here were then recalculated with non-SOC,
collinear, self-consistent DFT on those frozen geometries. Separating geometry
provenance from label provenance is central to the study.

NbP-H is a useful transfer test because it combines a surface, an adsorbate,
and an electronically unusual Weyl-semimetal substrate. More importantly, the
r0-r8 comparison is reference-invariant: every point contains the same atoms,
so `E(z) - E_min` is unaffected by arbitrary elemental energy offsets between
DFT and MACE.

This repository demonstrates:

- read-only auditing of legacy VASP data;
- explicit geometry and label provenance;
- separation of atomistic labels from DOS, band, Wannier, and transport output;
- force-aware, fixed-composition foundation-model evaluation;
- generated, machine-readable evidence rather than hand-copied results;
- an evidence-based decision on whether new labels and fine-tuning are justified.

## Data actually used

| Calculation | Geometry origin | Reference labels | Role |
|---|---|---|---|
| Clean 56-atom NbP slab | SOC relaxation | Collinear SCF | Clean reference |
| r0 NbP-H, 57 atoms | SOC-relaxed H-covered slab | Collinear SCF | Adsorbed reference and scan minimum |
| r1-r8 NbP-H | Frozen r0 substrate; H shifted along `z` | Collinear SCF | Primary held-out scan |
| H2, two atoms | Fixed 0.740 Å geometry | Collinear SCF | Candidate gas reference; currently flagged |
| Local H finite displacements | Frozen structure | Collinear finite differences | Optional secondary curvature check |

DOS, non-SCF, band-structure, Wannier, and transport calculations are completed
parts of the original electronic-structure work, but they do not add unique,
method-consistent atomistic labels for this MLIP benchmark.

## Measured input audit

The committed audit was generated from the private raw outputs on 2026-08-25:

- clean plus r0-r8 are all present with no duplicate or missing labels;
- all ten slab SCF jobs completed without a detected electronic error;
- all r0-r8 points contain complete 57-by-3 force arrays;
- all r0-r8 calculations share one collinear DFT method signature;
- all r0-r8 calculations have an identical substrate geometry;
- the successful r7/r8 reruns replace earlier failed archives;
- r0 is the lowest-energy sampled point;
- H heights span 0.573117 to 1.773117 Å above the topmost substrate atom.

The H2 SCF completed at `E(sigma->0) = -6.76275276 eV`, but the atoms retain
forces of `0.372344 eV/Å` and the lateral cell dimensions are inherited from
the slab (`3.358 Å × 3.358 Å`). It is therefore recorded as a candidate, not
yet accepted as a converged isolated-molecule reference. No final adsorption
energy is claimed from it.

See [`artifacts/scf_audit.json`](artifacts/scf_audit.json) and
[`artifacts/zscan_manifest.csv`](artifacts/zscan_manifest.csv) for the measured
records and SHA-256 provenance hashes.

## Measured zero-shot result

MACE-MP-0-small was evaluated without fitting on all nine r0-r8 structures.
Forces require no elemental reference correction, and the fixed-composition
energy curve requires no fitted elemental shift.

![DFT versus zero-shot MACE energy and H-force curves](artifacts/mace_zeroshot.svg)

| Metric | Measured value |
|---|---:|
| Relative-energy MAE, all nine points | 0.1943 eV |
| Relative-energy RMSE | 0.2647 eV |
| Maximum relative-energy error | 0.5677 eV |
| Relative-energy Pearson correlation | 0.9445 |
| Near-minimum MAE (`DFT relative energy <= 0.10 eV`, 5 points) | 0.0903 eV |
| Short-range MAE (`z < 0.90 A`, 2 points) | 0.1703 eV |
| Large-height MAE (`z > 1.40 A`, 2 points) | 0.4783 eV |
| All force-component MAE | 0.0445 eV/A |
| H vertical-force MAE | 0.8057 eV/A |
| H vertical-force RMSE | 0.8926 eV/A |
| DFT sampled minimum | r0, 1.1731 A |
| MACE sampled minimum | r8, 1.0731 A |

The model captures the overall energy ordering well enough to produce a high
correlation, but it makes the interaction well too stiff: errors grow on the
large-height side, H-force magnitudes are systematically too large away from
the minimum, and the sampled minimum shifts by 0.10 A. The low aggregate
force-component MAE is not sufficient evidence by itself because the 56 slab
atoms dilute the error on the chemically active H coordinate.

## Reproduce the DFT audit

The raw VASP files remain local and are not committed. With the same directory
layout, run:

```powershell
python src\audit_scf.py `
  --scf-root $env:NBP_SCF_ROOT `
  --h2-dir $env:H2_SCF_DIR `
  --out artifacts `
  --dataset-out data\zscan.xyz
```

The script uses only the Python standard library. It writes portable
`raw://...` identifiers rather than publicizing local filesystem paths.
The coordinate-bearing `data/zscan.xyz` file is kept local and ignored by Git.

## Reproduce the zero-shot MACE-MP-0 result

```powershell
python src\eval_zscan.py --data data\zscan.xyz --model small --device cpu
```

The measured run used `mace-torch 0.3.16`, CPU PyTorch 2.13.0, ASE 3.29.0,
NumPy 2.5.2, and float64. Nine 57-atom predictions took 11.25 s after model
loading (1.25 s/structure). The exact working numerical stack is in
`requirements-lock.txt`; the checkpoint filename and SHA-256 hash are recorded
in `artifacts/mace_zeroshot.json`.

Fine-tuning is not part of the minimum successful project. The present data
contain static collinear labels rather than a collinear relaxation trajectory.
If zero-shot results motivate an extension, diverse geometries will be selected
from the existing SOC trajectories and recalculated as frozen collinear SCF
points before any train/validation/test split is attempted. The entire r0-r8
family will remain outside training and hyperparameter selection.

## Adsorption-energy status

For one H,

```text
E_ads = E(slab+H) - E(clean slab) - 0.5 E(H2)
```

The clean and H-covered slab terms are available as collinear single-point
energies on SOC-relaxed geometries. A final value is intentionally withheld
until a method-consistent, sufficiently isolated, near-equilibrium H2 reference
is verified or recalculated.

## Limitations

- The r0-r8 curve is a constrained one-dimensional cut, not a relaxed pathway,
  kinetic barrier, or minimum-energy path.
- Geometry optimization used SOC, while benchmark labels use collinear SCF.
- The current dataset is sufficient for zero-shot evaluation but not for a
  defensible fine-tune.
- No claim is made about electronic bands, Weyl points, Fermi arcs, DOS,
  Wannier Hamiltonians, or transport; an MLIP does not predict those outputs.
- No molecular dynamics, reactive sampling, high-temperature stability, or
  transfer beyond the measured NbP-H configurations has been validated.
- The available ZPE information is secondary and incomplete across states.

## Repository map

```text
src/audit_scf.py       reproducible DFT input audit and manifest generation
src/eval_zscan.py      primary zero-shot energy and H-force evaluation
artifacts/             small generated evidence committed to Git
tests/                 automated scientific and parsing checks
RUNBOOK.md             minimal reproduction commands and data policy
```

## Data availability

Raw VASP outputs are not distributed through this repository because they are
large and contain the broader source research archive. The public artifacts
contain derived scalar values, portable source identifiers, and cryptographic
hashes sufficient to trace results back to the private originals. A publishable
sample-data policy will be reviewed before release.

## Status

**DFT audit and zero-shot MACE-MP-0-small evaluation complete.** Every numeric
claim above is backed by a committed generated artifact. Fine-tuning remains a
conditional extension requiring additional collinear labels.
