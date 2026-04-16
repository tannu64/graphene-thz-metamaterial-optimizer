# Phase 3: Top 5 Candidates for COMSOL Validation

**For:** Dr. Riccardo Degl'Innocenti
**From:** Linah Salem Alrejaee (Student ID: 231001766)
**Date:** 16 April 2026
**Subject:** ML-suggested parameter sets for surrogate model validation

---

## Summary

The Phase 3 NSGA-II multi-objective optimiser (using the trained ML
surrogates from the 27-simulation dataset) produced a Pareto front of
28 non-dominated parameter combinations. The 5 candidates below were
selected as the most informative test points to validate the surrogate.

If you could run these in COMSOL when you have capacity, the results
will form an **independent held-out test set**, enabling us to compute
true out-of-sample MAE and R² for the ML pipeline — a milestone
deliverable for the Phase 3 chapter of the dissertation.

---

## Top 5 Candidate Parameter Sets

All candidates use **w_graph = 1 μm** (fixed in current dataset).

| # | d_x (μm) | g_w (μm) | c_w (μm) | h_graph (μm) | w_au (μm) | Predicted S12 dip (dB) | Predicted \|freq shift\| (GHz) |
|---|----------|----------|----------|--------------|-----------|------------------------|-------------------------------|
| 1 | 34 | 3 | 27 | -3 | 4 | -14.19 | 21.8 |
| 2 | 34 | 3 | 27 | -2 | 4 | -14.17 | 27.4 |
| 3 | 33 | 3 | 27 | -3 | 4 | -14.14 | 29.0 |
| 4 | 34 | 3 | 27 |  0 | 4 | -14.11 | 30.5 |
| 5 | 32 | 3 | 27 |  0 | 4 | -14.08 | 32.0 |

### COMSOL Filename Convention
Following the existing naming convention, both ON (σ=0.3 mS) and OFF
(σ=1.2 mS) states are needed for each candidate:

- `Sdx34_gw3cw27wgraph1hgraphm3wau4_sigma0.3.txt`
- `Sdx34_gw3cw27wgraph1hgraphm3wau4_sigma1.2.txt`
- `Sdx34_gw3cw27wgraph1hgraphm2wau4_sigma0.3.txt`
- `Sdx34_gw3cw27wgraph1hgraphm2wau4_sigma1.2.txt`
- `Sdx33_gw3cw27wgraph1hgraphm3wau4_sigma0.3.txt`
- `Sdx33_gw3cw27wgraph1hgraphm3wau4_sigma1.2.txt`
- `Sdx34_gw3cw27wgraph1hgraph0wau4_sigma0.3.txt`
- `Sdx34_gw3cw27wgraph1hgraph0wau4_sigma1.2.txt`
- `Sdx32_gw3cw27wgraph1hgraph0wau4_sigma0.3.txt`
- `Sdx32_gw3cw27wgraph1hgraph0wau4_sigma1.2.txt`

(10 total simulations → ~120 hours of COMSOL time on a local workstation;
a subset would still be highly valuable for validation.)

---

## Validation Acceptance Criteria

When the COMSOL results are returned, the surrogate's reliability will
be assessed against:

| Metric | Threshold | Meaning |
|---|---|---|
| **MAE (S12 dip)** | ≤ 1.5 dB | ~10% of current best dip depth |
| **MAE (frequency)** | ≤ 20 GHz | ~5% of typical 400 GHz resonance |
| **R² on 5 points** | ≥ 0.6 | Independent out-of-sample correlation |

If these thresholds are met, the ML pipeline will be declared
**reliable for surrogate-driven design exploration** in the dissertation.

---

## Notes on Candidate Selection Rationale

- All 5 top-ranked candidates converge on **c_w = 27 μm, g_w = 3 μm, w_au = 4 μm**
  — close to but not identical to the existing best configuration
  (d_x=35, c_w=28, h_graph=6), providing a useful test of the surrogate's
  generalisation.
- The candidates span **h_graph ∈ {-3, -2, 0}**, a region with sparse
  existing data — this is where validation information is most valuable.
- The Gaussian Process surrogate has moderate R² (≈0.08) on S12 dip depth
  after the Phase 3 data expansion; these 5 simulations will provide the
  honest evidence for whether the ML pipeline is actually reliable enough
  for design use.

---

## File Location

This document is automatically generated and kept in sync with the NSGA-II
output at:
- `output/phase3_candidates_for_comsol.md` (this file)
- `output/phase3_pareto_front.csv` (full Pareto front)
- `output/phase3_bo_top10.csv` (Bayesian Optimisation top-10)

---

Thank you for considering this validation request.

Kind regards,
Linah Salem Alrejaee
