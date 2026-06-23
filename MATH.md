# freqgen — Mathematics, Theory, and Full Derivations

> **Reference paper:** SPAI — *Any-Resolution AI-Generated Image Detection by Spectral Learning*
> arXiv: [2411.19417](https://arxiv.org/abs/2411.19417) · CVPR 2025

---

## 1. Natural Image Statistics (the theoretical foundation)

### The 1/f power law
Natural images follow a power spectrum that decays as:

    P(f) ∝ f^(−α)     where α ≈ 2

In log-log space this is a straight line with slope −α. This means:
- Low frequencies (coarse structure) carry much more energy than high frequencies
- Power spectrum P(f) = |FFT|² decreases as 1/f²
- Magnitude |FFT| decreases as 1/f (slope ≈ −1 in log-log)

**Why this matters:** Any generator that doesn't faithfully reproduce this law
leaves a detectable spectral fingerprint.

### Measuring it — the radial profile
We collapse the 2D Fourier spectrum to a 1D curve by azimuthal averaging:

    profile[r] = (1/N_r) × Σ{ |FFT2D(image)[u,v]| : round(sqrt((u-H/2)²+(v-W/2)²)) = r }

where N_r is the number of pixels at radius r from DC (shifted centre).

The log-log slope of this curve tells us how close the image is to 1/f:
- Real images (RAISE-1k RAW photos): slope ≈ −1 in magnitude (= −2 in power)
- AI fakes (SD1.4): slope more negative (−1.2 to −2.1) — HF deficit or surplus

---

## 2. The Spectral Fingerprint — What We Found

### High-band energy gap
We partition the radial profile into three bands:
    Low  : r ∈ [0, 20)   — coarse structure
    Mid  : r ∈ [20, 60)  — texture
    High : r ∈ [60, 128] — fine detail / noise

The **high-band mean** is the key diagnostic:
    high_band = mean{ profile[r] : r ≥ 60 }

**Measured results on real data (2026-06-23, N=30 each):**

| Source | High-band | Gap vs COCO | Log-log slope |
|---|---|---|---|
| Real (COCO val2017) | 2805.5 | 1.00× | −1.28 |
| Fake (Synthbuster SD1.4) | 3653.1 | 0.77× (MORE HF) | −1.20 |
| Matched fakes | 2788.2 | 1.01× (closed) | −1.33 |

**Original finding (synthetic test, N=60):**

| Source | High-band | Gap vs 1/f real | Log-log slope |
|---|---|---|---|
| Synthetic real (1/f) | 3995 | 1.00× | −0.91 |
| Synthetic fake (HF suppressed 14×) | 265 | 15.0× (LESS HF) | −2.10 |
| Matched | 2963 | 1.35× (closed) | −1.05 |

**Note on direction:** The gap direction depends on the real baseline.
- vs RAISE-1k (RAW camera): SD1.4 has LESS HF (the classic "diffusion deficit")
- vs COCO (JPEG photos): SD1.4 has MORE HF (COCO is softer due to JPEG + diverse scenes)
The attack closes the gap to ~1× regardless of direction.

---

## 3. The Spectral Matching Attack — Full Derivation

### Mathematical formulation

**Given:**
- Fake image: F ∈ ℝ^{H×W}
- Real target profile: T[r] = (1/|S|) Σ_{i∈S} profile_i[r]  for real images S

**Goal:** Produce F_matched such that profile(F_matched)[r] ≈ T[r]
while preserving spatial structure (phase).

**Algorithm:**

**Step 1 — FFT:**
    Ĝ = FFT2D(F)   ∈ ℂ^{H×W}
    Ĝ[u,v] = |Ĝ[u,v]| × e^{i × ∠Ĝ[u,v]}

The 2D DFT is:
    Ĝ[u,v] = Σ_{x=0}^{H-1} Σ_{y=0}^{W-1} F[x,y] × e^{-i2π(xu/H + yv/W)}

**Step 2 — Shifted to centre DC:**
    Ĝ_s = fftshift(Ĝ)   (moves DC from corner to centre)

**Step 3 — Fake's radial profile:**
    src[r] = mean{ |Ĝ_s[u,v]| : round(sqrt((u-H/2)²+(v-W/2)²)) = r }

**Step 4 — Per-radius gain:**
    gain[r] = T[r] / (src[r] + ε)     ε = 1e-12  (numerical stability)
    gain[r] = clip(gain[r], g_min, g_max)          g_min=0.1, g_max=12.0
    gain[0] = 1.0                                   (preserve DC = mean brightness)

**Step 5 — 2D gain map:**
    r_map[u,v] = round(sqrt((u-H/2)²+(v-W/2)²))
    gmap[u,v]  = gain[ clip(r_map[u,v], 0, R-1) ]
    gmap[u,v]  = 1.0  for r_map[u,v] ≥ R          (corners beyond valid radius)
    gmap[u,v]  = 1.0  for r_map[u,v] = 0           (DC, already handled)

**Step 6 — Apply gain (scale magnitude, preserve phase):**
    Ĝ_matched = Ĝ_s × gmap

    Proof of phase preservation:
    Ĝ_s[u,v] × gmap[u,v]
    = (|Ĝ_s[u,v]| × e^{i∠Ĝ_s[u,v]}) × gmap[u,v]
    = (|Ĝ_s[u,v]| × gmap[u,v]) × e^{i∠Ĝ_s[u,v]}
    ↑ new magnitude                ↑ unchanged phase

**Step 7 — Inverse transform:**
    F_matched = IFFT2D(ifftshift(Ĝ_matched)).real
    F_matched = clip(F_matched, 0, 255)

**Why the clip to [0,255] is safe:**
    The gain is bounded: 0.1 ≤ gain ≤ 12.0
    So the maximum output energy is ≤ 12× the input energy
    Clipping removes out-of-range values (artifacts at image edges from
    high-gain amplification of edge frequencies)

### Complexity analysis
- FFT2D: O(HW log(HW))  — dominant step
- Profile computation: O(HW)
- Gain computation: O(R) where R = min(H,W)/2 ≈ 128 for 256×256
- gmap construction: O(HW)
- IFFT2D: O(HW log(HW))

**Total: O(HW log(HW)) ≈ 26ms for 256×256 on CPU**

Comparison with StealthDiffusion (prior SOTA attack):
- StealthDiffusion: O(T × HW) per diffusion step × ~50 steps ≈ seconds/image on GPU
- freqgen: O(HW log HW) per image ≈ milliseconds on CPU
- Speedup: ~10,000×

---

## 4. SPAI Architecture — How It Works

### Overview
SPAI learns a model of the spectral distribution of REAL images under a
self-supervised setup (no fake images needed for training). AI-generated images
are detected as out-of-distribution samples.

### Step 1: Frequency decomposition
For image x ∈ ℝ^{H×W}:
    X = FFT2D(x)
    Mask M[u,v] = 0 if sqrt((u-H/2)²+(v-W/2)²) < r,  1 otherwise   (r=16)
    Low-pass:   x_l = IFFT2D(X ⊙ (1-M))
    High-pass:  x_h = IFFT2D(X ⊙ M)

### Step 2: ViT feature extraction
Backbone G = ViT-B/16 pre-trained with Masked Frequency Modelling (MFM).
For any input y, G produces intermediate features:
    z_n^y = G_n(z_0^y) ∈ ℝ^{L×d}    for n = 1,...,N=12 blocks
    L = H×W/p² tokens,  d = 768 dimensions,  p=16 patch size

Projection operators P_n : ℝ^d → ℝ^D (with D=1024) map to comparison space:
    z̃_n^y = P_n(z_n^y)    for y ∈ {orig, low, high}

### Step 3: Spectral Reconstruction Similarity (SRS)
For two representation vectors A, B ∈ ℝ^{L×D}:
    σ(A, B)[l] = (A_l · B_l) / (||A_l|| × ||B_l||)    (cosine similarity per token)

Three SRS vectors:
    n_ol = σ(z̃^orig, z̃^low)     (original vs low-pass, per token, per block)
    n_oh = σ(z̃^orig, z̃^high)    (original vs high-pass)
    n_lh = σ(z̃^low,  z̃^high)    (low-pass vs high-pass)

Summary statistics per block (mean and std over L tokens):
    μ_ol^n = mean(n_ol^n),  σ_ol^n = std(n_ol^n)   → 2 scalars
    μ_oh^n, σ_oh^n                                   → 2 scalars
    μ_lh^n, σ_lh^n                                   → 2 scalars
    Per block: 6 scalars
    All N=12 blocks: z ∈ ℝ^{72}

### Step 4: Spectral Context Vector (SCV)
Learned spectral map C ∈ ℝ^{N×D}, projection functions P₁, P₂:
    z^block_n = mean_l(z̃_n^orig)   ∈ ℝ^D   (token-mean of original features)
    z_full = concat([z^block_n]_{n=1}^N)     ∈ ℝ^{N×2D}
    C = P₂(softmax(C) ⊙ P₁(z_full))         ∈ ℝ^{N×D}
    z_C = Σ_n C_n                             ∈ ℝ^D

### Step 5: Spectral Context Attention (SCA) — for any resolution
Split image into K patches of size h×w=224×224.
For each patch k: compute z_S^k ∈ ℝ^{D+72} (concatenation of SCV and SRS)
Attention pooling:
    α = softmax( q · (z̃_S^{1:K} @ W_K)^T / sqrt(D_h) )  ∈ ℝ^{1×K}
    z_S = (α · (z̃_S^{1:K} @ W_V)) @ W_O   ∈ ℝ^{D+72}

### Step 6: Classification
    score = σ(MLP(z_S))   ∈ (0,1)    where σ = sigmoid
    decision = 1 (fake) if score ≥ 0.5

---

## 5. Why the Attack Defeats SPAI — Mathematical Intuition

### SPAI's implicit assumption
SPAI's SRS measures how consistently the ViT G reconstructs the spectral
content of real images. For a real image x:
    Real image: G can predict x_h from x_l well → high σ(z̃^orig, z̃^high)

For an AI fake f whose spectral statistics differ from reals:
    Fake image: G struggles to predict f_h from f_l → low σ values → high SPAI score

### The attack's effect on SPAI
After spectral matching with target T = mean_real_profile:
    profile(F_matched)[r] ≈ T[r] = mean{ profile(real_i)[r] }

This means F_matched has the same RADIAL ENERGY DISTRIBUTION as real images.
The ViT G was trained on real images — its reconstruction model learned to
exploit the statistical relationships that arise from real images' spectral
statistics (including the 1/f law and its band relationships).

When F_matched has real-image spectral statistics:
    G can reconstruct F_matched's high-freq from low-freq just as well as for reals
    → High σ(z̃^orig, z̃^high) for matched fakes
    → Low SPAI score → not detected

**Formally:** The attack maps F into the support of the ViT's implicit
spectral model of real images, as measured by the SRS features.

### The fundamental limitation
The attack only equalizes the MARGINAL DISTRIBUTION of radial magnitudes.
It does not equalize joint statistics:
- P(|Ĝ[u₁,v₁]|, |Ĝ[u₂,v₂]|) for u₁≠u₂, v₁≠v₂  (magnitude correlations)
- The phase distribution ∠Ĝ[u,v]  (completely untouched)
- Within-ring magnitude patterns (gmap is constant within a ring)

A detector that measures ANY of these would not be defeated by the attack.

---

## 6. What Survives the Attack (Future Work)

| Feature | Attacked? | Example detector |
|---|---|---|
| Radial magnitude profile | ✅ Yes | freqgen, DEFEND, Dzanic et al. |
| SPAI's SRS (based on radial stats) | ✅ Yes | SPAI |
| Angular (2D) magnitude patterns | ❌ No | Grid artifact detectors (Frank et al.) |
| Fourier phase statistics | ❌ No | Phase-based forensics |
| Pixel-domain noise/residual | ❌ No | GAN fingerprint, sensor noise |
| JPEG artifact statistics | ❌ No | JPEG forensics |
| Semantic/perceptual features | ❌ No | CLIP-based, VGG-based |
| DIRE (reconstruction error) | ❌ No | Wang et al. ICCV 2023 |

**Paper's "what survives" section:** Detectors that use FULL 2D spectral
features (not just azimuthal averages), phase statistics, or pixel-domain
cues are not defeated. The attack motivates multi-cue detection approaches.

---

## 7. Score Statistics — What 10^-30 Means

SPAI's output score for matched fakes: mean ≈ 2.6×10^-19

The score is sigmoid(logit). For score = 2.6×10^-19:
    logit = log(score / (1-score)) ≈ log(2.6×10^-19) ≈ −42 nats
    = −42 / log(10) ≈ −18.2 log₁₀ units

This means SPAI is 10^18 : 1 in favour of "real" over "fake" for the matched
images. The attack doesn't just evade detection — it makes matched fakes
appear MORE "real" to SPAI than the real images themselves (real score mean=0.038).

This is consistent with the "overcorrection" interpretation: the spectral
matching UNDERSHOOTS the real target (matched high-band = 2788 < real 2805),
making the image slightly MORE natural-looking in the spectral sense.

For raw Synthbuster fakes: score = 1.000000... (essentially 1 - machine epsilon)
    logit ≈ +35 nats ≈ +15 log₁₀ units
SPAI is 10^15 : 1 in favour of "fake" — completely certain.

---

## 8. Notation Quick Reference

| Symbol | Meaning |
|---|---|
| F | Input fake image ∈ ℝ^{H×W} |
| FFT2D(·) | 2D Discrete Fourier Transform |
| fftshift(·) | Shift DC from corner to centre |
| Ĝ = fftshift(FFT2D(F)) | Centred Fourier transform |
| \|Ĝ\| | Fourier magnitude |
| ∠Ĝ | Fourier phase |
| r | Radial frequency (distance from DC) |
| profile[r] | Azimuthally-averaged magnitude at radius r |
| T[r] | Real target profile (mean over real images) |
| gain[r] | Per-radius amplitude scaling = T[r]/profile_fake[r] |
| gmap[u,v] | 2D gain map (gmap[u,v] = gain[r] for pixel at radius r) |
| F_matched | Output of spectral matching attack |
| SRS | Spectral Reconstruction Similarity (SPAI feature) |
| SCV | Spectral Context Vector (SPAI feature) |
| SCA | Spectral Context Attention (SPAI pooling) |
| z_S | SPAI's image-level spectral representation |
| P(f) | Power spectrum: P(f) = \|FFT\|² |
| α | Power law exponent ≈ 2 for natural images |
| ε | Numerical stabiliser = 1e-12 |
| g_min, g_max | Gain clip bounds = 0.1, 12.0 |
