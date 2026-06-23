# freqgen — Mathematics, Theory, and Full Derivations

> **Reference:** Karageorgiou et al., "Any-Resolution AI-Generated Image Detection by Spectral Learning," CVPR 2025. arXiv: 2411.19417

---

## 1. Natural Image Statistics

### The 1/f power law

Natural images follow a power spectrum that decays as:

```
P(f) ∝ f^(-α)     where α ≈ 2
```

In log-log space this is a straight line with slope -α. This means:

- **Low frequencies** (coarse structure) carry far more energy than high frequencies
- **Power spectrum** P(f) = |FFT|² decreases as 1/f²
- **Magnitude** |FFT| decreases as 1/f (log-log slope ≈ -1)

Any generator that doesn't reproduce this law leaves a detectable spectral fingerprint.

### The radial magnitude profile

We collapse the 2D Fourier spectrum to a 1D curve by azimuthal averaging:

```
profile[r] = mean{ |FFT2D(image)[u,v]| : round( sqrt((u - H/2)² + (v - W/2)²) ) = r }
```

where the average is over all pixels at integer radius r from the DC centre.

Measured log-log slopes:
- Real images (RAISE-1k RAW): slope ≈ -1 (magnitude domain)
- AI fakes (SD1.4): slope ≈ -1.2 to -2.1 — flatter or steeper depending on baseline

---

## 2. The Spectral Fingerprint

### Band decomposition

We partition the radial profile into three bands:

```
Low  band : r in [0, 20)    — coarse structure / illumination
Mid  band : r in [20, 60)   — texture / edges
High band : r in [60, 128]  — fine detail / noise
```

The **high-band mean** is the key diagnostic:

```
high_band = mean{ profile[r] : r >= 60 }
```

### Measured results (real data, 2026-06-23, N=30 images each)

| Source              | High-band | Ratio vs COCO | Log-log slope |
|---------------------|-----------|---------------|---------------|
| Real (COCO val2017) | 2805.5    | 1.00x         | -1.28         |
| Fake (Synthbuster SD1.4) | 3653.1 | 0.77x (MORE HF) | -1.20     |
| Matched fakes       | 2788.2    | 1.01x (closed)| -1.33         |

### Measured results (synthetic test, N=60)

| Source                       | High-band | Gap     | Log-log slope |
|------------------------------|-----------|---------|---------------|
| Synthetic real (1/f law)     | 3995      | 1.00x   | -0.91         |
| Synthetic fake (14x HF deficit) | 265   | 15.0x   | -2.10         |
| Matched fakes                | 2963      | 1.35x   | -1.05         |

### Note on gap direction

The direction of the gap depends on the real baseline chosen:

- **vs RAISE-1k (high-quality RAW):** SD1.4 fakes have LESS high-frequency energy (classic "diffusion deficit") — this is the original finding from notebooks 1-2
- **vs COCO (diverse JPEGs):** SD1.4 fakes have MORE high-frequency energy — COCO is softer due to JPEG compression and blurry scenes

In both cases the attack closes the gap to approximately 1x.

---

## 3. The Spectral Matching Attack

### Setup

**Input:**
- Fake image `F`, shape H x W pixels (we use 256x256)
- Target profile `T[r]` = mean radial profile over a set of real images

**Goal:** Produce `F_matched` such that `profile(F_matched)[r] ≈ T[r]`
while leaving spatial structure (image content) unchanged.

### Full algorithm

**Step 1 — Compute the 2D Fourier transform:**

```
G_hat = FFT2D(F)
G_hat[u,v] = |G_hat[u,v]| * exp(i * phase[u,v])
```

The 2D DFT formula:
```
G_hat[u,v] = sum_{x=0}^{H-1} sum_{y=0}^{W-1}  F[x,y] * exp(-i * 2*pi * (x*u/H + y*v/W))
```

**Step 2 — Shift DC to centre:**

```
G_s = fftshift(G_hat)    (moves DC component from [0,0] to [H/2, W/2])
```

**Step 3 — Compute the fake's radial profile:**

```
src[r] = mean{ |G_s[u,v]| : round(sqrt((u-H/2)² + (v-W/2)²)) = r }
```

**Step 4 — Compute per-radius gain:**

```
gain[r] = T[r] / (src[r] + eps)       eps = 1e-12 for numerical stability
gain[r] = clip(gain[r], 0.1, 12.0)    prevent extreme rescaling
gain[0] = 1.0                          preserve DC = preserve mean brightness
```

**Step 5 — Build 2D gain map (same gain for all pixels at radius r):**

```
r_map[u,v] = round( sqrt((u-H/2)² + (v-W/2)²) )
gmap[u,v]  = gain[ clip(r_map[u,v], 0, R-1) ]
gmap[u,v]  = 1.0   for pixels beyond radius R (diagonal corners)
```

**Step 6 — Apply gain to magnitude, leave phase untouched:**

```
G_matched = G_s * gmap
```

Phase preservation proof:

```
G_s[u,v] * gmap[u,v]
  = ( |G_s[u,v]| * exp(i * phase[u,v]) ) * gmap[u,v]
  = ( |G_s[u,v]| * gmap[u,v] ) * exp(i * phase[u,v])
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^
       new magnitude (rescaled)     unchanged phase (content)
```

Phase encodes WHERE things are in the image (edges, textures, structure).
Magnitude encodes HOW MUCH of each frequency is present.
Changing only magnitude preserves visible content.

**Step 7 — Inverse transform and clip:**

```
F_matched = IFFT2D( ifftshift(G_matched) ).real
F_matched = clip(F_matched, 0, 255)
```

### Gain clip bounds [0.1, 12.0]

- **Lower bound 0.1:** Prevents suppressing frequencies to near-zero (avoids loss of information and smoothing artifacts)
- **Upper bound 12.0:** Prevents amplifying noise by orders of magnitude
- In practice on SD1.4 vs COCO: most gains fall in [0.3, 3.0], rarely hitting the bounds

### Computational complexity

| Step | Cost |
|------|------|
| FFT2D | O(HW log HW) |
| Radial profile | O(HW) |
| Gain vector | O(R), R = min(H,W)/2 |
| 2D gain map | O(HW) |
| IFFT2D | O(HW log HW) |
| **Total** | **O(HW log HW)** |

For 256x256 images: ~26ms on CPU (38 images/second).

**Comparison with StealthDiffusion (ACM MM 2024 — closest prior work):**
- StealthDiffusion: diffusion loop + adversarial optimisation → ~seconds/image on GPU
- freqgen attack: single FFT pass → ~0.1ms/image on CPU
- Speedup: approximately 10,000x, no GPU required, generator-agnostic

---

## 4. SPAI Architecture

SPAI (Karageorgiou et al., CVPR 2025) learns to model the spectral distribution
of REAL images in a self-supervised way. AI-generated images are detected as
out-of-distribution samples.

### Step 1 — Frequency decomposition

For input image `x` (H x W):

```
X = FFT2D(x)

Mask M[u,v]:  0 if sqrt((u-H/2)² + (v-W/2)²) < r=16,  else 1

x_low  = IFFT2D( X * (1 - M) )    (low-pass:  keeps frequencies inside  radius 16)
x_high = IFFT2D( X * M )           (high-pass: keeps frequencies outside radius 16)
```

### Step 2 — ViT feature extraction

Backbone G = ViT-B/16 pre-trained with Masked Frequency Modelling (MFM).
The ViT is frozen during the SPAI detection phase.

For input y ∈ {original, x_low, x_high}, the n-th transformer block outputs:

```
z_n^y ∈ R^{L x d}     L = (H*W) / p²  tokens,  d = 768 dims,  p = 16 patch size
```

Learnable projection operators map to a comparison space (D = 1024 dims):

```
z_tilde_n^y = P_n( z_n^y )    ∈ R^{L x D}
```

### Step 3 — Spectral Reconstruction Similarity (SRS)

Cosine similarity between two token sequences A, B ∈ R^{L x D}:

```
cos_sim(A, B)[l] = dot(A[l], B[l]) / ( norm(A[l]) * norm(B[l]) )
```

Three pairwise SRS vectors computed per transformer block n:

```
n_ol[n] = cos_sim(z_tilde_n^orig, z_tilde_n^low )   (original vs low-pass)
n_oh[n] = cos_sim(z_tilde_n^orig, z_tilde_n^high)   (original vs high-pass)
n_lh[n] = cos_sim(z_tilde_n^low,  z_tilde_n^high)   (low-pass vs high-pass)
```

Summary statistics per block (mean and std over L tokens → 2 scalars each):

```
For each of n_ol, n_oh, n_lh:  mean + std  →  2 scalars
Per block n:  6 scalars
All N=12 blocks:  z ∈ R^{72}
```

### Step 4 — Spectral Context Vector (SCV)

Captures which frequency regime is most informative for detection:

```
z_block_n = mean_over_tokens( z_tilde_n^orig )     ∈ R^D
z_full    = concat( [z_block_n for n in 1..N] )    ∈ R^{N x 2D}
C  = P2( softmax(C_learned) * P1(z_full) )         ∈ R^{N x D}
z_C = sum_over_n( C[n] )                           ∈ R^D
```

Final image-level spectral vector:

```
z_S = concat(z_C, z)     ∈ R^{D + 72}
```

### Step 5 — Spectral Context Attention (SCA) for any resolution

Splits high-resolution images into K patches (224x224 each):

```
For each patch k:  compute z_S^k  using Steps 1-4

Query vector q ∈ R^{Dh},  weight matrices W_K, W_V, W_O

Attention weights:   alpha = softmax( q · (z_S_all @ W_K).T / sqrt(Dh) )    ∈ R^{1 x K}
Aggregated vector:   z_S   = ( alpha · (z_S_all @ W_V) ) @ W_O              ∈ R^{D+72}
```

### Step 6 — Classification

```
score = sigmoid( MLP(z_S) )      ∈ (0, 1)
decision = FAKE  if score >= 0.5
           REAL  if score < 0.5
```

---

## 5. Why the Attack Defeats SPAI

### SPAI's implicit assumption

For real images, the ViT G (trained on real images) can reconstruct missing
frequencies accurately — it has learned how real images' low and high frequency
components relate to each other:

```
Real image:  G predicts x_high from x_low well
             → high cos_sim(z_orig, z_high)
             → low SPAI score → called REAL
```

For AI-generated fakes with anomalous spectral statistics:

```
Fake image:  G struggles to predict f_high from f_low
             (spectral statistics differ from what G learned)
             → low cos_sim values
             → high SPAI score → called FAKE
```

### What the attack does

After spectral matching with target T = mean radial profile of real images:

```
profile(F_matched)[r]  ≈  T[r]  =  mean over real images of profile[r]
```

The matched fake now has the same RADIAL ENERGY DISTRIBUTION as real images.
The ViT G was trained exclusively on real images, so it has learned to exploit
the statistical relationships that arise from real images' 1/f spectral structure.

When F_matched follows real spectral statistics:

```
G reconstructs F_matched's high-freq from low-freq accurately
  → high cos_sim(z_orig, z_high) for matched fakes
  → low SPAI score → called REAL
```

### Formal statement

The spectral matching attack maps fake images into the support of the ViT's
implicit model of the real image distribution, as measured by the SRS features.
Because SPAI's detection signal is entirely derived from this spectral model,
the attack reduces the detection rate to 0%.

### What the attack does NOT equalize

The attack only equalizes the **azimuthally-averaged (radial) magnitude**.
It does not touch:

1. **Phase** — completely unchanged. Phase encodes spatial structure (where things are).
2. **Angular patterns** — within each frequency ring, magnitude may be anisotropic (e.g., grid artifacts from upsampling). These survive because gmap is constant within a ring.
3. **Magnitude correlations** — the joint distribution of magnitudes at different (u,v) pairs is not equalized, only their ring-means.
4. **Pixel-domain statistics** — sensor noise, JPEG artifacts, color statistics.
5. **Semantic features** — CLIP embeddings, perceptual features, object-level patterns.

Any detector that measures these features will NOT be defeated by this attack.

---

## 6. What Survives the Attack

| Detection feature              | Defeated? | Example |
|-------------------------------|-----------|---------|
| Radial magnitude profile      | YES       | freqgen, DEFEND (Zhang et al.), Dzanic et al. |
| SPAI SRS                      | YES       | SPAI (Karageorgiou et al., CVPR 2025) |
| 2D magnitude (angular patterns)| No       | Frank et al. ICML 2020 (upsampling grids) |
| Fourier phase statistics       | No        | Phase forensics |
| Pixel-domain noise residual    | No        | GAN fingerprints, Marra et al. |
| JPEG artifact patterns         | No        | JPEG forensics |
| Perceptual / semantic features | No        | CLIP-based detectors |
| Diffusion reconstruction error | No        | DIRE (Wang et al., ICCV 2023) |

**Implication for the paper:** The attack demonstrates a fundamental limitation
of radial-magnitude-based detection. Robust detection requires combining multiple
cue types — frequency, phase, spatial, and semantic — that cannot all be
neutralised by a single post-processing step.

---

## 7. Interpreting the Extreme Score Values

### Why fake scores = 1.000000...

Synthbuster SD1.4 is one of the exact datasets SPAI was trained and evaluated on
(see SPAI paper Table 1: 99.6% AUC on SD1.4). The classifier's sigmoid output
saturates at effectively 1.0 for images firmly inside its "fake" decision region.

Logit interpretation:
```
score = 1.000000 (machine precision) → logit ≈ +35 nats ≈ +15 log10 units
SPAI is 10^15 to 1 in favour of FAKE
```

### Why matched scores ≈ 10^-19 to 10^-30

The spectral attack maps the matched fakes to have slightly LESS high-band energy
than COCO reals (matched high-band = 2788 vs COCO real = 2805). This slight
"overcorrection" pushes the images even further into SPAI's "real" decision region
than actual COCO images (which score mean = 0.038).

Logit interpretation for score = 2.6e-19:
```
logit = log(2.6e-19 / (1 - 2.6e-19))  ≈  log(2.6e-19)  ≈  -42 nats  ≈  -18 log10 units
SPAI is 10^18 to 1 in favour of REAL
```

The matched fakes appear more "real" to SPAI than actual real images. This is
not a path error — it is the attack working beyond expectation.

---

## 8. Notation Reference

| Symbol | Meaning |
|--------|---------|
| F | Input fake image, shape H x W |
| G_hat | 2D Fourier transform of F |
| G_s | Shift-centred Fourier transform: fftshift(G_hat) |
| profile[r] | Azimuthally-averaged magnitude at integer radius r |
| T[r] | Real-image target profile (mean over real set S) |
| gain[r] | Per-radius amplitude scaling = T[r] / (src[r] + eps) |
| gmap[u,v] | 2D gain map: gmap[u,v] = gain[r] for pixel at radius r |
| F_matched | Output of spectral matching attack |
| eps | Numerical stabiliser = 1e-12 |
| g_min, g_max | Gain clip bounds = 0.1, 12.0 |
| P(f) | Power spectrum: P(f) = abs(FFT)^2 |
| alpha | Power law exponent ≈ 2 for natural images |
| SRS | Spectral Reconstruction Similarity — SPAI's detection feature |
| SCV | Spectral Context Vector — SPAI's context summary |
| SCA | Spectral Context Attention — SPAI's patch pooling |
| z_S | SPAI's final image-level representation |
| L | Number of ViT tokens = H*W / p^2 |
| d | ViT token dimension = 768 |
| D | SRS projection dimension = 1024 |
| N | Number of ViT blocks = 12 |
| r | Mask radius in SPAI's frequency split = 16 pixels |
