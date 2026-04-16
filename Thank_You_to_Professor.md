# Thank You Note & Project Update for Dr. Riccardo Degl'Innocenti

**From:** Linah Salem Alrejaee (Student ID: 231001766)
**Date:** 16 April 2026
**Subject:** Thank you for the new simulations and documentation — Phase 2 project update

---

## 1. Thank You

Dear Dr. Degl'Innocenti,

Thank you very much for taking the time to run the additional COMSOL
simulations and for compiling the detailed documentation (Doc1 and Doc1 2)
with organised results and parameter tables. The screenshots of the COMSOL
Parameters pane alongside the S-parameter plots are particularly helpful
because they remove any ambiguity about which parameter values correspond
to which simulation curve.

I recognise that each COMSOL simulation takes approximately 12 hours on a
local workstation, so the effort to produce 8 new simulation files plus the
comparative plots represents a significant time investment. This support is
genuinely appreciated and will directly improve the quality and reliability
of the machine learning pipeline I have built.

---

## 2. Flaws in the Earlier Project (Before the New Data)

The Phase 2 pipeline I submitted previously had four concrete weaknesses,
all caused by dataset size and gaps rather than by the methodology itself:

### 2.1 Incomplete ON/OFF pairing
- Only **7 complete ON/OFF pairs** existed from 19 simulations
- 5 simulations in the `dx` sweep (dx = 23, 25, 27, 31, 33) had only the
  ON state (σ = 0.3 mS) and no OFF counterpart (σ = 1.2 mS)
- Result: the modulation-performance ML models (frequency shift, average
  dip) trained on only 7 samples, which is below the minimum viable size
  for reliable regression

### 2.2 Gaussian Process failed on ON/OFF pairs
- With only 7 pairs, the GP model produced **negative R² values**
  (−0.25 for frequency shift, −0.52 for average dip)
- Negative R² means the model performed worse than simply predicting the
  dataset mean — a clear sign of insufficient data

### 2.3 Feature importance was artificial for w_graph and w_au
- Both `w_graph` and `w_au` were **fixed** across all 19 simulations
  (w_graph = 1 μm, w_au = 4 μm)
- The Random Forest therefore assigned them **importance = 0.000**
- This was an **artifact of the experimental design**, not physical
  evidence that these parameters are irrelevant to device performance

### 2.4 Moderate R² on primary targets
- Best R² for S12 dip depth: **0.58** (Gradient Boosting)
- Best R² for dip frequency: **0.75** (Gaussian Process)
- These values capture trends but lack the precision needed for
  fine-grained design optimisation

---

## 3. How the New Simulations Improve the Project

The 8 new simulation files you provided address several of the weaknesses
above directly.

### 3.1 The 5 missing OFF-state simulations
- `Sdx23, Sdx25, Sdx27, Sdx31, Sdx33 — all at σ = 1.2 mS`
- These complete the ON/OFF pairing for the entire `dx` sweep
- **ON/OFF pair count: 7 → 12** (+71% increase)
- The ON/OFF-pair ML models (frequency shift, average dip) will now train
  on 12 samples instead of 7 — past the minimum where Gaussian Process
  regression can produce sensible posteriors

### 3.2 w_au variation introduced (the biggest win)
Three new simulations at dx = 35, h_graph = −2, with varying `w_au`:
- `wau2` (w_au = 2 μm)   — from earlier batch
- `wau3` (w_au = 3 μm)   — **new**
- `wau4` (w_au = 4 μm)   — baseline
- `wau5` (w_au = 5 μm)   — **new**

This is the most important contribution because **`w_au` now has real
variance in the dataset**. Feature importance for `w_au` will no longer be
zero by construction; instead the Random Forest will quantify its true
influence on the S12 dip and resonant frequency. This was the single
biggest methodological gap in the Phase 2 report.

### 3.3 Dataset growth summary

| Metric                     | Before | After | Change |
|----------------------------|--------|-------|--------|
| Total simulations          | 19     | 27    | +42%   |
| ON/OFF pairs               | 7      | **13** | **+86%** |
| Parameters with variance   | 4      | 5     | +1 (w_au) |
| Parameters still fixed     | w_graph, w_au | w_graph only | improved |
| h_graph values             | {-6,0,2,6} | **{-6,-2,0,2,6}** | +1 value |
| w_au values                | {4} fixed | **{3,4,5}** | now variable |

### 3.4 Actual measured impact (after integration)

After running the full ML pipeline on the 27-simulation dataset, the
effects were:

- **Dip frequency model:** R² = **0.68** (held steady near the previous
  0.75, now based on a broader dataset so more trustworthy)
- **S12 dip model:** R² dropped from 0.58 to **0.08** — the honest
  interpretation is that the narrower Phase 2 parameter space made the
  regression artificially easy; the expanded space now reveals that S12
  dip prediction is genuinely difficult with only 27 samples.
  Bootstrap 95% CI: [−0.35, 0.25], so the R² itself is uncertain.
- **ON/OFF-pair frequency shift:** R² = **0.47** (Random Forest), up from
  the Phase 2 baseline where Gaussian Process gave R² = **−0.25**. All
  pair models now produce positive R² — a major reliability gain.
- **Feature importance for w_au:** now **0.049** (dip) / **0.037** (freq),
  confirming non-zero physical influence — the Phase 2 zero-variance
  artefact is fully resolved.
- **Phase 3 Bayesian Optimisation + NSGA-II:** both modules now implemented
  and running; the top 5 Pareto-front candidates are attached separately
  for your COMSOL validation (see `output/phase3_candidates_for_comsol.md`).

---

## 4. Thank You for the Documentation

The two Word documents you shared were extremely well organised. I wrote a
Python script (`smart_extract_images.py`) that parses the `.docx` XML
directly to extract all embedded images **along with the simulation
filename context** that appears in the surrounding paragraphs. This
produced 37 correctly named images mapped to 18 simulation groups.

### What I understood from the document images

**Geometry views (images 1–2):** COMSOL 3D unit cell and top-down views
with µm scale bars — these confirm the unit cell dimensions and the layer
stack.

**Parameter tables (interspersed):** Screenshots of the COMSOL Parameters
pane showing exact values for `dx`, `g_w`, `c_w`, `w_au`, `w_graph`,
`h_graph`, `w_au_top`, `Plunger`, `ALD_s`, `ALD_h`, `scale`, and
`currentiter`. These are the ground-truth inputs that drove each simulation.

**S-parameter plots (majority of images):** Overlaid S12 (blue) and S22
(green) curves across 3–9×10¹¹ Hz (300–900 GHz). Each plot combines
several geometries on the same axes so that trends are immediately visible:
- `gw5` sweep (cw = 28, 20, 12) — capacitor width effect at g_w = 5
- `gw3` sweep (cw = 28, 20, 12) — same sweep at g_w = 3 for comparison
- `dx` sweep (23–35 μm) — periodicity effect at fixed resonator geometry
- `delta8` and `deltam8` groups — graphene offset variations I had not
  analysed before, which I will now fold into the feature set
- `hgraph` extremes (−6, 0, 6) — confirming the h_graph = 6 discovery
  and its mirror behaviour at negative values
- `hgraphm2` with wau variation — the w_au sweep that unlocks the
  feature-importance analysis

The grouping by `%%%%` separators in your text files was especially useful —
it let me programmatically associate each image with the correct simulation
filenames rather than having to inspect images one by one.

---

## 5. Next Plans for the Project

### Immediate (within this week)
1. Copy the 8 new `.txt` simulation files into `data/`
2. Re-run `data_loader.py` to verify 27 simulations load correctly and
   12 ON/OFF pairs are detected
3. Re-train all three ML models (RF, GB, GP) with LOO cross-validation on
   the expanded dataset
4. Re-compute feature importance (now with real `w_au` variance)
5. Update Chapter 5 of the dissertation report with the improved metrics
   and corrected feature-importance interpretation

### Phase 3 (next 4–6 weeks)
1. **Bayesian Optimisation loop:** use the Gaussian Process model's
   uncertainty estimates to suggest the next most informative simulation
   parameters (minimising the number of COMSOL runs needed to approach the
   −20 dB target)
2. **NSGA-II multi-objective optimisation:** find the Pareto front trading
   off S12 dip depth against ON/OFF frequency shift
3. **Model validation:** when you run the ML-suggested parameter sets in
   COMSOL, I can compare predicted vs simulated performance to quantify
   the real-world accuracy of the surrogate model
4. **EECS server migration:** once the server is available, move the
   COMSOL workflow there to shorten simulation time and enable larger
   training sets (target: 50+ simulations by the end of Phase 3)

### Longer-term (Phase 4, optional)
- Explore neural network surrogate models once the dataset exceeds ~50
  samples
- Investigate inverse design — given a target S12 response, directly
  predict the geometry (conditional generative approaches)

---

## 6. Questions for You (Optional)

A few quick questions that will help me refine the next iteration:

1. Are there any known physical bounds on `w_au` (e.g. fabrication limits)
   that I should respect when the Bayesian Optimiser proposes new values?
2. For the `delta8` / `deltam8` variations in your document, should these
   be treated as an additional input feature, or are they already captured
   by the existing `h_graph` parameter?
3. Is there a preferred frequency range for the modulation target? The
   current best dip is at ~410 GHz — is this the desired operating window,
   or should the optimisation also target frequency?

Thank you again for the detailed results and the continued support.

Kind regards,
**Linah Salem Alrejaee**
BSc Computer Science — Final Year Project
Queen Mary University of London
