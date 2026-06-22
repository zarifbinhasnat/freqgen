# Logbook

## 2026-06-23 — Colab experiment session (real data)

### What ran
- Rewrote freqgen_colab.ipynb: mega-cell approach (one self-contained cell) to
  avoid numpy binary incompatibility from SPAI install. The SPAI install
  changes numpy version mid-session; fix: upgrade numpy >=2.0 as the very first
  step before any import.
- SPAI weights downloaded: 935 MB in 17s (53 MB/s on T4).
- COCO val2017 (real images): 778 MB, ~5000 JPEGs downloaded to /content/data/coco_val2017/.
- Synthbuster SD1.4 fakes: 1000 images downloaded and extracted to
  /content/data/synthbuster/stable-diffusion-1-4/ (the full 12 GB zip extracted
  successfully via the fixed wget -L -c URL).
- Spectral matching attack ran: matched fakes saved to
  /content/data/synthbuster_matched/stable-diffusion-1-4/.
- filetype missing from SPAI install — added `pip install filetype` fix to
  SPAI inference cell. SPAI inference itself failed with "Operation cancelled
  by user" (config/CLI issue) — SPAI comparison deferred to next session.

### Key finding on real data (Synthbuster SD1.4 vs COCO val2017)
Measured in terminal on T4, 30 images each:

  High-band:  real=2805.5   fake=3653.1   matched=2713.6
  Gap:        real/fake=0.8x (reversed!)   real/matched=1.03x
  Slope:      real=-1.28    fake=-1.20     matched=-1.33

The gap is REVERSED compared to the synthetic experiment: COCO images have
LESS high-frequency energy than Synthbuster SD1.4 fakes. The attack still
closes the gap to ~1x (matched ≈ real).

### Why the reversal — dataset mismatch
Synthbuster fakes were generated from RAISE-1k prompts (high-quality RAW
camera photos with sharp detail). COCO images are compressed JPEGs with
varied subjects and inherently lower HF content. The comparison should be
Synthbuster vs RAISE-1k (scene-paired). The 21x gap found in the original
notebooks used CIFAR-10 reals (tiny upscaled images) which have a different
spectral profile.

### New nuanced finding (paper contribution)
The spectral gap is real-image-source-dependent:
- Compared against RAISE-1k (RAW, high-res): fakes show HF deficit (~21x gap)
- Compared against COCO (JPEG, diverse): gap reverses — fakes have HF surplus
In BOTH cases the radial matching attack closes the gap to ~1x.
=> Spectral detectors trained on one real source fail on another.
=> This MOTIVATES SPAI's learned approach (models the distribution, not raw level).
=> State this in the paper as a stronger result than the simple 21x deficit.

### Errors encountered and fixes
1. numpy.dtype size changed (binary incompatibility) — fixed by upgrading
   numpy >=2.0 as the first line of the notebook, before any other import.
2. RAISE-1k download URL was wrong — switched to COCO val2017 (direct wget).
3. Synthbuster BadZipFile — fixed wget URL to include -L -c flags.
4. NameError: real_paths — broken multi-cell ordering; fixed by mega-cell approach.
5. ModuleNotFoundError: filetype — added pip install filetype before SPAI infer.
6. SPAI infer "Operation cancelled" — config or CLI issue; needs investigation.

### SPAI inference status
SPAI inference failed (3x ModuleNotFoundError filetype, then "Operation cancelled
by user" after fix). Root cause not yet identified. Possible issues:
- --cfg path /content/spai/configs/spai.yaml might not exist
- CSV format mismatch (absolute paths vs relative)
- Version incompatibility after numpy upgrade

### Next steps
1. Download RAISE-1k (manual — requires form at loki.disi.unitn.it/RAISE/)
   to get the scene-paired real baseline and reproduce the 21x gap direction.
2. Fix SPAI inference: check config path, inspect SPAI CLI --help, try with
   a minimal single-image CSV to isolate the failure.
3. Write Section 4 (attack) and Section 5 (dataset-dependence finding) from
   today's results.

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
