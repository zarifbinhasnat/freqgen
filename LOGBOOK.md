# Logbook

## 2026-06-22

- Added `src/spectral.py`: radial magnitude profile, real-image target, and
  `spectral_match` (rewrites a fake's Fourier magnitude per radius to match the
  real target; phase preserved -> structure unchanged). Pure numpy/Pillow.
- Added notebook `Spectral Matching - Closing the Gap.ipynb` = paper Section 4
  (attack/robustness). Builds the real target, matches an SD fake, reports
  before/after band energies + slope, and renders the headline figure (+ color).
- Verified locally (synthetic 1/f vs HF-deficit): high-band gap 15.0x -> 1.35x,
  slope -2.10 -> -1.05, phase agreement 0.999 (DC preserved -> brightness intact).
- Finding for the paper: the radial spectrum is NOT a robust detector - one FFT
  post-process removes the fingerprint. Caveat to state: phase stats, full 2-D
  spectrum, and learned/pixel-domain detectors are untouched by this attack.
- Added `src/detector.py`: `radial_features` (log radial profile) + the attack's
  target; `residual_features` (peakiness of the spectral residual R = |FFT| /
  ring-mean, which exposes 2-D grid peaks the radial attack can't remove);
  `LogReg` (numpy, no sklearn); `run_evasion` harness.
- Added notebook `Spectral Detectors and Evasion.ipynb` (paper Section 5);
  executed headless end-to-end on CPU synthetic data.
- Result (60/60/60, synthetic 1/f-real vs HF-deficit+grid fake):
  radial detector  -> clean acc 1.00, evasion 1.00 (fully fooled);
  residual detector -> clean acc 1.00, evasion 0.00 (still catches matched).
  => frequency-magnitude detectors fragile; 2-D/phase cues survive.
- Added `MATH.md`: plain-English + full derivation (why gap closes, why phase is
  preserved, why residual/phase survive) + verified numbers + GPU budget.
- GPU handoff: only image *generation* (SD/SDXL) needs GPU. Set
  `USE_SYNTHETIC=False` and feed real CIFAR + SD fakes to get the headline table
  on real data. SD1.5 fits free Colab T4; SDXL wants L4/A100. See MATH.md table.
- Identified reference paper: SPAI (arXiv 2411.19417, CVPR 2025).
  Code: https://github.com/mever-team/spai
  Key insight: SPAI uses a frozen ViT-B/16 (MFM pre-trained) + Spectral
  Reconstruction Similarity (SRS) + Spectral Context Attention (SCA).
  It models the real spectral distribution in *latent* space, not hand-crafted
  radial profiles - making it a learned version of freqgen's spectral_report.
- Added SPAI section to MATH.md: architecture breakdown, comparison table,
  and the key open question: does the radial matching attack evade SPAI too?
  If yes -> CVPR 2025 SOTA has a gap; if no -> learned detectors are robust.
  Either is publishable. SPAI inference needs <8GB GPU (free T4 enough).
- Located SPAI's EXACT evaluation dataset (18 CSVs in spai/data): 13 generators
  + 5 real sources. Fakes: 9 from Synthbuster (Zenodo 10066460) + 4 (sd3, mjv61,
  gigagan, flux) from SPAI authors' Google Drive. Reals: RAISE-1k, COCO2017,
  ImageNet, OpenImages, FODB. CSV format: image,class,split (1=fake,0=real).
  Synthbuster fakes are scene-paired with RAISE-1k reals -> ideal for matching.
- Decision: use SPAI's exact public dataset, NO image generation. Eliminates the
  generation GPU step entirely.
- Added DATASETS.md (exact dataset list + downloads + recommended starting
  subset) and PLAN.md (master plan: recap of all work, locked dataset, phased
  roadmap A/B/C, compute budget, open question).
- Next (Phase A, CPU): download RAISE-1k + Synthbuster SD1.4/SDXL, write loader
  for SPAI CSVs, reproduce 21x gap on real data, run attack + evasion on real
  images. Then Phase B: run SPAI inference on raw vs matched fakes.

## 2026-06-07

- Initialized repo structure: notebooks/, src/, results/, paper/
- Existing work: FFT comparison notebook, spectral analysis (real vs AI-generated), band breakdown visualization
- Next: migrate existing notebooks into notebooks/ with sequential numbering
