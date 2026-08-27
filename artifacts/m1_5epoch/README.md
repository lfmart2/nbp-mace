# M1 five-epoch fine-tuning artifacts

M1 initializes from the exact MACE-MP-0-small checkpoint used by M0 and trains
for five CPU epochs on 32 QE collinear structures. Ten contiguous structures are
used for validation. The raw coordinate files and trained model remain ignored.

- `training_summary.json` and `training_history.csv`: measured training history,
  timing, configuration hash, split hash, and trained-model hash.
- `comparison.json`: M0-versus-M1 metrics on identical blocked structures.
- `predictions.csv`: point-level reference-invariant energy comparison.
- `m0_vs_m1.*`: internal and overlap-diagnostic error comparison.
- `zscan/comparison.json`: external r0-r8 result.
- `zscan/m0_vs_m1_zscan.*`: external energy and H-force curves.

The overlap run is not statistically independent of the training trajectory.
The r0-r8 scan is the external test and was absent from training, validation,
energy alignment, checkpoint selection, and hyperparameter decisions.
