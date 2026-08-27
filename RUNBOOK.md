# Reproduction runbook

## 1. Create the local DFT manifest

```powershell
$env:NBP_SCF_ROOT = "<path-to-extracted-03_SCF>"
$env:H2_SCF_DIR = "<path-to-extracted-H2-SCF>"

python src\audit_scf.py `
  --scf-root $env:NBP_SCF_ROOT `
  --h2-dir $env:H2_SCF_DIR `
  --out artifacts `
  --dataset-out data\zscan.xyz
```

The coordinate-bearing dataset stays under the Git-ignored `data/` directory.

## 2. Install the measured numerical stack

```powershell
python -m pip install -r requirements-lock.txt
```

## 3. Run the zero-shot benchmark

```powershell
python src\eval_zscan.py `
  --data data\zscan.xyz `
  --model small `
  --device cpu `
  --dtype float64
```

The first run downloads the official MACE-MP-0-small checkpoint. The evaluator
records its filename and SHA-256 hash with package versions, hardware, timing,
point-level predictions, and aggregate metrics.

## 4. Run tests

```powershell
python -m unittest discover -s tests -v
```

## 5. Build the expanded relaxation benchmark

```powershell
$env:NBP_MLIP_RAW_ROOT = "<path-to-curated-raw_sources>"

python src\build_benchmark_dataset.py `
  --raw-root $env:NBP_MLIP_RAW_ROOT `
  --dataset data\m0_benchmark.xyz `
  --summary artifacts\dataset_summary.json
```

The builder parses QE and VASP trajectories, restores fixed-atom masks,
deduplicates geometries within each source calculation, and keeps method labels
separate. The coordinate-bearing Extended XYZ remains ignored by Git.

## 6. Run the expanded M0 benchmark

```powershell
$env:MACE_MP0_SMALL_CHECKPOINT = "<path-to-20231210mace128L0_energy_epoch249model>"

python src\eval_relaxations.py `
  --data data\m0_benchmark.xyz `
  --checkpoint $env:MACE_MP0_SMALL_CHECKPOINT `
  --out artifacts\m0_relaxations `
  --device cpu `
  --dtype float64
```

No parameters or elemental reference energies are fitted in M0. The evaluator
records the exact checkpoint hash and reports metrics by source calculation.

## 7. Prepare and run M1

```powershell
python src\prepare_m1.py `
  --data data\m0_benchmark.xyz `
  --checkpoint $env:MACE_MP0_SMALL_CHECKPOINT `
  --out data\m1 `
  --manifest artifacts\m1_split_manifest.json

mace_run_train --config configs\m1_5epoch.yaml
```

`prepare_m1.py` fits elemental reference shifts using training structures only.
Nb/P shifts are rank-deficient individually because every slab has equal Nb and
P counts; their combined reference and the H contribution are identifiable.

The trained model remains under Git-ignored `results/`. Reproduce its public
metrics with:

```powershell
python src\eval_m1.py `
  --data-dir data\m1 `
  --m0-model $env:MACE_MP0_SMALL_CHECKPOINT `
  --m1-model results\m1_5epoch\models\M1_5epoch.model `
  --m1-label "M1 five-epoch fine-tune" `
  --out artifacts\m1_5epoch

python src\eval_m1_zscan.py `
  --data data\zscan.xyz `
  --m0-model $env:MACE_MP0_SMALL_CHECKPOINT `
  --m1-model results\m1_5epoch\models\M1_5epoch.model `
  --out artifacts\m1_5epoch\zscan
```

## Data policy

Raw VASP files, coordinate-bearing extxyz data, model checkpoints, and temporary
results are not committed. The repository contains derived tables, figures,
metrics, and SHA-256 provenance hashes.
