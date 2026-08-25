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
python -m unittest tests.test_audit_scf -v
```

## Data policy

Raw VASP files, coordinate-bearing extxyz data, model checkpoints, and temporary
results are not committed. The repository contains derived tables, figures,
metrics, and SHA-256 provenance hashes.

