# freqgen — Master Plan & Status

One page that organizes everything: what exists, what the thesis is, the exact
dataset, the step-by-step plan, and the compute budget. Updated 2026-06-22.

---

## 0. The thesis (what the paper argues)

> Real images have an invariant spectral distribution; AI fakes deviate from it
> (a measurable **high-frequency deficit**, ~21× in the high band). We show this
> fingerprint can be **erased by a cheap FFT post-process** that preserves image
> content — defeating hand-crafted spectral detectors 100%. We then ask whether
> the same attack defeats the **learned** SOTA detector (SPAI, CVPR 2025), and
> identify which cues (2-D peaks, phase) survive. Conclusion: spectral-magnitude
> detection is necessary but not sufficient.

Reference: SPAI — arXiv [2411.19417](https://arxiv.org/abs/2411.19417),
code [github.com/mever-team/spai](https://github.com/mever-team/spai).

---

## 1. Where we are — recap of ALL work

### 1a. Pre-existing (original work, before this session)
- `notebooks/FFT On Real vs Fake Images.ipynb` — FFT pipeline, CIFAR reals +
  SD fakes, `compute_spectral_metrics` (radial profile, bands, slope).
- `notebooks/Visualizing The Band Breakdown.ipynb` — the band-energy figure.
- **Finding established:** fakes are deficient in high frequencies; ~21× gap;
  real slope ≈ −2 (1/f²). → `results/Spectral Analysis (Real vs ai Generated).png`.
- `paper/Spectral Learning .pdf` — the SPAI reference paper.

### 1b. This session — built + verified (all CPU)
| Artifact | What it does | Status |
|---|---|---|
| `src/spectral.py` | radial profile, real target, **`spectral_match`** (the attack), report | ✅ verified |
| `src/detector.py` | radial + residual features, `LogReg`, `run_evasion` | ✅ verified |
| `notebooks/Spectral Matching - Closing the Gap.ipynb` | the attack (paper §4) | ✅ runs |
| `notebooks/Spectral Detectors and Evasion.ipynb` | detectors + evasion (paper §5) | ✅ runs headless |
| `MATH.md` | plain-English + full derivation + SPAI breakdown + GPU budget | ✅ |
| `DATASETS.md` | exact SPAI dataset list + downloads | ✅ |

**Verified numbers (synthetic, CPU):**
- Attack: high-band gap **15× → 1.35×**, slope −2.10 → −1.05, phase agreement **0.999**.
- Evasion: radial detector **100% fooled**; residual detector **0% fooled** (survives).

### 1c. Key realization this session
- The reference is SPAI (CVPR 2025) — a *learned* detector, not hand-crafted.
- **Plan pivot:** stop generating fakes with a GPU. Use SPAI's **exact public
  dataset** instead (Synthbuster + RAISE-1k + author Drive bundle). Generation
  GPU step is eliminated. See `DATASETS.md`.

---

## 2. The dataset (locked)

Exactly what SPAI used — 13 generators + 5 real sources (see `DATASETS.md`).
**Start with:** RAISE-1k (reals) + Synthbuster SD1.4/SDXL (fakes), scene-paired,
direct download, no GPU, no login.

---

## 3. The plan — step by step

### Phase A — real data ← **PARTIALLY DONE (2026-06-23)**
- [x] A1. Download COCO val2017 (real, 778 MB) + Synthbuster SD1.4 (1000 fakes).
- [x] A2. Run spectral matching attack on real Synthbuster fakes → gap closes to ~1x.
- [x] A3. Measure high-band gap and slope on real data.
- [ ] A1b. Download RAISE-1k (scene-paired with Synthbuster) — manual form download.
      Needed to reproduce the expected 21x deficit direction (COCO reverses the gap).
- [ ] A5. Run radial+residual detector evasion on real data (currently only synthetic).

**Key finding 2026-06-23:** Gap is dataset-dependent.
  COCO real vs SD1.4 fake: gap = 0.8x (reversed — fakes have MORE HF than COCO)
  Matched fakes vs COCO:   gap = 1.03x (attack closes it either direction)
  RAISE-1k real vs SD1.4:  expected ~21x deficit (need to verify with real data)

### Phase B — test against SPAI  (small <8GB GPU, or slow CPU)
- [ ] B1. Install SPAI (inference only), download weights checkpoint.
- [ ] B2. Run `python -m spai infer` on (a) raw fakes, (b) matched fakes.
- [ ] B3. Compare SPAI scores raw vs matched. **Paper headline #2 — the answer
      to "does the attack beat the SOTA?"**

### Phase C — generalization + writing
- [ ] C1. Repeat A4/A5/B on `flux`, `sd3` (Drive bundle) → newer generators.
- [ ] C2. Sweep `gain_clip`; ablate `smooth`, DC-preservation.
- [ ] C3. Add a phase-based detector to round out "what survives".
- [ ] C4. Write §4 (attack) and §5 (evasion) from `MATH.md`; pull figures from
      `results/`; position against SPAI.

---

## 4. Compute budget

| Phase | GPU | Notes |
|---|---|---|
| A (matching, detectors, evasion) | **none** | pure numpy, laptop-fine |
| B (SPAI inference) | <8GB or CPU | ViT-B/16; CPU ~seconds/image |
| C3 retrain SPAI (optional) | 48GB L40S | **not needed** for first draft |

→ The whole paper is reachable with **little or no GPU**. See `MATH.md §5`.

---

## 5. Open question (the crux)

**Does radial spectral matching evade SPAI?** SPAI's SRS is computed in ViT
latent space, so it *may* capture features beyond the azimuthal magnitude our
attack equalizes. Phase B answers this:
- evaded → we found a gap in CVPR 2025 SOTA (strong result),
- not evaded → learned detectors are robust; our attack is a clean evaluation
  tool and a motivation for learned methods.

Either way: publishable.
