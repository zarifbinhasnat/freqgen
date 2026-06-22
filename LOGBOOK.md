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
- Next: run on the real CIFAR/SD outputs from Notebook 1; sweep `gain_clip`;
  optionally add SDXL as a second generator.

## 2026-06-07

- Initialized repo structure: notebooks/, src/, results/, paper/
- Existing work: FFT comparison notebook, spectral analysis (real vs AI-generated), band breakdown visualization
- Next: migrate existing notebooks into notebooks/ with sequential numbering
