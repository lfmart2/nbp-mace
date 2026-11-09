# T3-03 figure captions (H surface diffusion on Nb-terminated NbP)

**fig_kmc_arrhenius.** Tracer diffusion coefficients (a) D_x and (b) D_y of one H on the Nb-terminated NbP slab
versus 1000/T, 200–600 K. Lines: exact solution of the lattice model (Bloch master equation) with classical Vineyard
harmonic-TST rates for M1 (blue) and M0 (orange). Shaded: all saddles shifted by ±0.05 eV with the minima fixed, a
judgment-based range for the MLIP barrier error, not a statistical error bar (factor ≈7 each way at 300 K).
Dashed: prefactor fixed at 10¹³ s⁻¹ instead of Vineyard. Filled circles: kMC (BKL, 20 000 walkers) with classical rates
from T3-02; open triangles: kMC spot checks at the band edges. kMC error bars are standard errors from 32 independent
walker batches and are mostly smaller than the markers. M1, 300 K: D_x = 1.97e-06 cm²/s
(band 2.85e-07–1.37e-05), D_y = 8.66e-07 cm²/s (band 1.25e-07–5.99e-06);
E_a,eff = 0.225 / 0.246 eV (x / y, 200–600 K fit). All values are MLIP-based (MACE M1 fine-tune and M0 = MACE-MP-0-small) harmonic-TST estimates for a lattice model parameterised with 1 ML collective barriers (1×1 cell, all periodic H move together); they are not measured or DFT-validated values, and the model is valid only above ~150–200 K (tunnelling crossover 85–101 K).

**fig_kmc_msd.** Mean-square displacement of H along x and y from BKL kMC (M1, classical rates, 300 K,
20000 independent walkers, unwrapped coordinates) with the exact lines 2D_x t and 2D_y t. D from the slope over
t ≥ 0.2 t_end (shaded region excluded). The anisotropy D_x/D_y ≈ 2.3 follows from the lower saddle S2 on the
x route. This validates the kMC engine against the exact lattice solution only, not the rates themselves. All values are MLIP-based (MACE M1 fine-tune and M0 = MACE-MP-0-small) harmonic-TST estimates for a lattice model parameterised with 1 ML collective barriers (1×1 cell, all periodic H move together); they are not measured or DFT-validated values, and the model is valid only above ~150–200 K (tunnelling crossover 85–101 K).

**fig_kmc_occupation.** Time-weighted kMC site occupations (filled circles: full B1/T/B2 network; open squares:
coarse-grained model with T eliminated, shifted by +6 K for visibility) versus Boltzmann populations from the same
harmonic partition functions (lines), (a) M1 and (b) M0, classical rates. Agreement within statistical error confirms
detailed balance of the event table; T stays a minority site (≤ ~11 % at 600 K for M1). All values are MLIP-based (MACE M1 fine-tune and M0 = MACE-MP-0-small) harmonic-TST estimates for a lattice model parameterised with 1 ML collective barriers (1×1 cell, all periodic H move together); they are not measured or DFT-validated values, and the model is valid only above ~150–200 K (tunnelling crossover 85–101 K).

**fig_kmc_network.** Site network used in the lattice model: per surface cell (a = 3.358 Å) one atop site T
and two Nb–Nb bridges B1 (the DFT-anchored r0 site) and B2. B1 ↔ T hops cross saddle S1 along y, T ↔ B2 hops cross S2 along
x (both a/2 long). The dotted direct B1 ↔ B2 route (sensitivity analysis only) uses the straight-line maximum of the
frozen-substrate T1-01 map, an upper estimate of its saddle (0.379 / 0.423 eV for M1 / M0). Energies are
CI-NEB key points relative to B1 (M1 / M0), classical, without ZPE. All values are MLIP-based (MACE M1 fine-tune and M0 = MACE-MP-0-small) harmonic-TST estimates for a lattice model parameterised with 1 ML collective barriers (1×1 cell, all periodic H move together); they are not measured or DFT-validated values, and the model is valid only above ~150–200 K (tunnelling crossover 85–101 K).
