# M1 protocol: small-data collinear fine-tuning

M1 tests whether a small, leakage-controlled QE dataset improves the active
hydrogen force without degrading the reference-invariant energy behavior seen
in M0. It is initialized from the exact MACE-MP-0-small checkpoint used by M0.

## Split

- Train: QE clean frames 0--5 and QE+H run-A frames 0--25 (32 structures).
- Validation: QE clean frames 6--8 and QE+H run-A frames 26--32 (10 structures).
- Internal test: QE+H run-A frames 33 onward after deduplication (6 structures).
- Overlap diagnostic: all 36 frames from QE+H run B.
- External test: the entire r0--r8 scan; never used for fitting or selection.

The internal validation and test blocks are highly correlated late-relaxation
structures. They measure local trajectory interpolation, not broad chemical
transfer. Run B follows almost the same path as run A and is not independent.

## Energy alignment

Absolute DFT and foundation-model energies use different elemental reference
conventions. Elemental shifts are fitted only on the 32 training structures and
subtracted from DFT total energies before optimization. Forces and all
fixed-composition energy differences are unchanged. Nb and P always occur in
equal counts, so their individual shifts are rank-deficient; only their combined
NbP reference and the additional H contribution are identifiable.

## Decision gate

1. Time one full CPU epoch using `configs/m1_smoke.yaml`.
2. Continue only if projected runtime is practical and validation is finite.
3. Compare M1 against the saved M0 predictions on identical held-out structures.
4. Claim improvement only if H-force error falls without a material degradation
   of relative-energy error.

The first `1e-4`, 1:100 smoke test reduced validation force RMSE but caused a
large energy regression. `configs/m1_lr1e5_smoke.yaml` therefore tests a tenfold
smaller learning rate and 10:100 energy/force weighting. Final CLI dataset
evaluation is skipped in that timing configuration because the Windows CPU
post-evaluator stalled after the model had already been saved; held-out metrics
will be computed by the repository's explicit evaluator instead.
