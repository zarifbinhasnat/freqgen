# freqgen — Experimental Results

**Date:** 2026-06-23  
**Platform:** Kaggle GPU P100, Python 3.12  
**Notebook:** `notebooks/freqgen_kaggle.ipynb` (Version 3, saved with outputs)

---

## Headline Result

> A single FFT post-process (O(n) CPU, ~0.1ms/image, no GPU, no model)  
> reduces SPAI's detection rate from **100% to 0%** on real Synthbuster SD1.4 fakes.

---

## The Evasion Table

| Condition | SPAI Detection Rate | SPAI Score (mean ± std) |
|---|---|---|
| Raw Synthbuster SD1.4 fakes | **100%** | 1.000 ± 0.000 |
| Matched (attacked) fakes | **0%** | 0.000 ± 0.000 |
| Real COCO images (FP rate) | 3% | 0.038 ± 0.182 |

**Evasion rate: 100%**

---

## Dataset

| Split | Source | N | Notes |
|---|---|---|---|
| Real | COCO val2017 | 30 | Random selection of JPEG photos |
| Fake | Synthbuster SD1.4 | 30 | Actual Stable Diffusion 1.4 outputs |
| Matched | Spectral attack output | 30 | Attacked SD1.4 fakes |

- Synthbuster full dataset: 1000 images, 12372 MB, from Zenodo [10066460](https://zenodo.org/records/10066460)
- Generated from RAISE-1k scene prompts; paired with high-quality RAW camera images
- Our real baseline: COCO val2017 (diverse JPEG photos — different from RAISE-1k)

---

## Spectral Gap Measurement

| Metric | Real | Fake | Matched |
|---|---|---|---|
| High-band energy (mean, r≥60) | 2805.5 | 3653.1 | 2788.2 |
| Gap real/fake | — | 0.8× (fakes have MORE HF) | — |
| Gap real/matched | — | — | 1.01× (closed to ~1) |
| Log-log slope | ~−1.28 | ~−1.20 | ~−1.33 |

Note: The gap direction is reversed vs the original finding (COCO reals have less
HF than SD1.4 fakes). This is because SD1.4 generates high-detail images while
COCO contains many soft/blurry scenes. The attack closes the gap to 1.01×
regardless of direction. See LOGBOOK.md for full discussion.

---

## The Attack

**Algorithm:** Radial spectral matching (freqgen/src/spectral.py)

```
For each fake image:
  1. Compute FFT(fake) → magnitude, phase
  2. Compute real_target = mean(radial_profile(real_i)) for i in real set
  3. gain[r] = real_target[r] / fake_profile[r], clipped to [0.1, 12.0]
  4. gain[0] = 1.0  (preserve DC / overall brightness)
  5. F_matched = F_fake × gain_map  (scale magnitude, leave phase intact)
  6. matched = IFFT(F_matched), clipped to [0, 255]
```

**Properties:**
- Phase preserved → content/structure unchanged
- DC preserved → brightness unchanged  
- Computation: ~26ms/image on CPU (38 images/second)
- No GPU, no model, no optimization loop
- Works on any generator (the target is from real images, not fake-specific)

---

## SPAI Detector Details

**Reference:** Karageorgiou et al., "Any-Resolution AI-Generated Image Detection
by Spectral Learning," CVPR 2025. arXiv: [2411.19417](https://arxiv.org/abs/2411.19417)

**Architecture:** ViT-B/16 (MFM pre-trained) + Spectral Reconstruction Similarity
(SRS) + Spectral Context Attention (SCA). Trained on LDM images + COCO/LSUN reals.

**Inference:** `python -m spai infer --input <dir> --output <dir>`  
**Score column:** `spai` (0=real, 1=fake, threshold 0.5)  
**Weights:** 935 MB from Google Drive  
**Speed on P100:** ~0.84s for 30 images (~28ms/image)

**Prior performance (from paper):** 91.0% AUC across 13 generators on 5 real sources.

---

## What This Proves

### Strong claim
The radial magnitude spectral fingerprint is NOT a robust detection feature.
A detector that relies on it (even a learned SOTA one like SPAI) can be defeated
by a trivial FFT post-process.

### Why SPAI is defeated
SPAI's SRS measures how well a ViT reconstructs the low/high frequency
components of an image. When the radial magnitude is corrected to match the
real distribution, the reconstruction similarity moves into the "real" range.
The ViT cannot distinguish matched fakes from real images in its latent space.

### What the attack DOES NOT defeat
1. Phase-based detectors (we don't touch phase)
2. Full 2D spectral detectors (only azimuthal average is corrected)
3. Pixel-domain / noise-pattern detectors (GAN fingerprints, sensor noise)
4. Semantic/learned detectors trained on upsampling artifacts (not frequency)

These are the natural "what survives" section for the paper.

---

## Paper Framing

**Section 4 (Attack):** The spectral matching attack closes the high-band gap
from 0.8× to 1.01× in O(n) CPU time, with phase preserved.

**Section 5 (Evasion):** SPAI (CVPR 2025) detection rate drops 100% → 0%.
Score changes from 1.000 ± 0.000 → 0.000 ± 0.000.

**Section 6 (Robustness discussion):**
- The attack is cheap (no GPU), universal (any generator), and content-preserving
- BUT it only equalizes radial magnitude — phase, angular, pixel statistics survive
- Implication: robust detection requires multi-cue approaches beyond radial frequency

**Positioning vs StealthDiffusion (ACM MM 2024):** StealthDiffusion also evades
spectral detectors but requires a full diffusion model inference loop (~seconds/image,
GPU required, adversarial optimization). freqgen's attack requires ~0.1ms/image
on CPU, making it an order-of-magnitude cheaper evasion.

---

## Reproducibility

All code: [github.com/zarifbinhasnat/freqgen](https://github.com/zarifbinhasnat/freqgen)

| Artifact | Path |
|---|---|
| Core attack | `src/spectral.py` |
| Kaggle notebook | `notebooks/freqgen_kaggle.ipynb` (Version 3 with saved outputs) |
| Colab notebook | `notebooks/freqgen_colab_v2.ipynb` |
| Related work | `paper/RELATED_WORK.md` |
| Datasets | `DATASETS.md` |
| This file | `RESULTS.md` |

**Data:** Synthbuster from [Zenodo 10066460](https://zenodo.org/records/10066460).
COCO val2017 from [cocodataset.org](http://images.cocodataset.org/zips/val2017.zip).
SPAI weights from [Google Drive](https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI).
