# freqgen — the idea, the math, and the work so far

This file has three layers: a **plain-English** version (no math), the **actual
math**, and a **work log** of what is built and verified. Read whichever layer
you need. GPU requirements are at the bottom.

---

## 1. Plain English (the dumbed-down version)

**The fingerprint.** Take any photo and ask "how much fine detail (sharp edges,
tiny texture) vs. how much big smooth shape does it have?" Real camera photos
always have a predictable amount of fine detail — it fades out smoothly as the
detail gets finer (the "1/f" rule that natural images obey). AI image
generators (Stable Diffusion, etc.) get the big shapes right but **under-produce
the finest detail**. So if you measure detail-by-scale, a fake has a visible
*dip* at the fine end. In our numbers the fake had ~21× less fine-detail energy
than a real image. That dip is a fingerprint — you can catch fakes with it.

**The attack ("make a perfect fake").** Here's the catch: that fingerprint is
easy to forge. We measure the real "detail-by-scale" curve, then go into the
fake's frequency representation and **turn the fine-detail dial back up** until
its curve matches a real one. Crucially we only change *how much* energy is at
each scale — we never move *where* things are — so the picture looks identical,
but the fingerprint is gone. After this, a detector that relied on the
fingerprint is fooled 100% of the time.

**The honest part (what still catches it).** Our "dial" turns each scale up or
down *uniformly*. But AI generators also leave a different mark: faint repeating
grid patterns (from how they upscale images), which show up as sharp *spikes* in
the frequency picture at specific spots. Turning a whole scale up/down doesn't
remove a spike sitting inside that scale — it just scales the spike too. So a
second detector that looks for those spikes still catches the "perfect" fake.

**The paper's point in one line:** frequency-energy detectors are fragile (a
cheap trick beats them), so real deepfake detection must also use phase /
2-D-structure / learned cues. We show both halves with code and numbers.

---

## 2. The math

### 2.1 Setup and the fingerprint

A grayscale image is $x[m,n]$, $m,n\in\{0,\dots,N-1\}$ (we use $N=256$). Its 2-D
DFT and centered spectrum:

$$X[u,v]=\sum_{m,n} x[m,n]\,e^{-j2\pi(um+vn)/N},\qquad X_s=\mathrm{fftshift}(X).$$

Write it in magnitude/phase form $X_s = A\,e^{j\phi}$, i.e. $A[u,v]=|X_s[u,v]|$
and $\phi[u,v]=\angle X_s[u,v]$. The DC term sits at the center $(c,c)$, $c=N/2$.

Define the integer radius from DC:

$$r[u,v]=\Big\lfloor \sqrt{(u-c)^2+(v-c)^2}\,\Big\rceil .$$

The **radial (azimuthally-averaged) magnitude profile** averages $A$ over each
ring $S_\rho=\{(u,v):r[u,v]=\rho\}$:

$$P(\rho)=\frac{1}{|S_\rho|}\sum_{(u,v)\in S_\rho} A[u,v],\qquad \rho=0,1,\dots,\tfrac{N}{2}-1 .$$

This 1-D curve is the fingerprint. Natural images obey a power law
$P(\rho)\propto \rho^{-\alpha}$, so $\log P$ vs $\log\rho$ is roughly a straight
line. The detector features used in Notebook 1:

$$\text{low}=\!\!\underset{0\le\rho<20}{\mathrm{mean}}\!\! P(\rho),\quad
  \text{mid}=\!\!\underset{20\le\rho<60}{\mathrm{mean}}\!\! P(\rho),\quad
  \text{high}=\!\!\underset{60\le\rho<N/2}{\mathrm{mean}}\!\! P(\rho),$$

and the log–log slope $s$ from a least-squares fit $\log P(\rho)\approx s\log\rho+b$.
The empirical gap we exploit:

$$\text{gap}=\frac{\text{high}_{\text{real}}}{\text{high}_{\text{fake}}}\approx 21 .$$

### 2.2 The attack (radial spectral matching)

Build a target from $K$ real images: $T(\rho)=\frac1K\sum_{k} P_{\text{real}_k}(\rho)$.
For a fake with profile $P_f$, define a **per-radius magnitude gain**

$$g(\rho)=\mathrm{clip}\!\left(\frac{T(\rho)}{P_f(\rho)+\varepsilon},\,g_{\min},g_{\max}\right),
\qquad g(0)=1\ \text{(preserve DC = brightness)} ,$$

optionally smoothed by a length-$w$ box filter to avoid ringing. Lift it to 2-D,
$G[u,v]=g(r[u,v])$ (and $G=1$ for $r\ge N/2$), and apply it to the **complex**
spectrum:

$$\hat X_s[u,v]=G[u,v]\,X_s[u,v]=g(r)\,A\,e^{j\phi}.$$

Then invert: $\hat x=\mathrm{clip}\big(\mathrm{Re}\,\mathrm{ifft2}(\mathrm{ifftshift}(\hat X_s)),\,0,255\big).$

**Why the fingerprint flips to "real".** On a ring, $r\equiv\rho$ is constant, so

$$\hat P(\rho)=\frac{1}{|S_\rho|}\sum_{S_\rho} g(r)A
            =g(\rho)\,\frac{1}{|S_\rho|}\sum_{S_\rho} A
            =g(\rho)\,P_f(\rho)\;\approx\;T(\rho).$$

So (without clipping) the matched fake's profile **equals the real target
exactly** — high-band mean and slope become real-like, gap $\to 1$. Clipping and
DC-preservation explain the small residual (we measure $\approx1.0$–$1.35\times$).

**Why the picture is unchanged.** $g(r)>0$ is real, so it multiplies magnitude
only; the phase $\phi$ is untouched. Phase carries the location of edges and
structure (the classic Oppenheim–Lim result that phase dominates perception), so
content is preserved. Measured phase agreement $\frac1{N^2}\sum\cos(\phi-\hat\phi)\approx0.999$.

### 2.3 Why the attack is limited (what survives)

Matching forces only the ring **average** to the target. Decompose the spectrum
on ring $\rho$ into mean plus azimuthal variation, $A=P_f(\rho)+\delta$. The
attack sends $A\mapsto g(\rho)A=g(\rho)P_f(\rho)+g(\rho)\delta$: the *relative*
variation $\delta/P_f$ is preserved. Therefore any cue carried by **within-ring
structure** survives. Define the **spectral residual**

$$R[u,v]=\frac{A[u,v]}{P(r[u,v])}.$$

Under the attack $R$ is invariant ($g$ cancels top and bottom). Upsampling /
checkerboard artifacts are sharp peaks in $R$ at fixed $(u,v)$, so a detector on
$R$'s peakiness (percentiles, kurtosis of $R$ over $r\ge20$) is **untouched** by
the attack. Phase statistics are likewise untouched. This is the "necessary but
not sufficient" message.

### 2.4 The detectors

Both are standardized logistic regressions, label $1=$ fake, $0=$ real:

$$\Pr(\text{fake}\mid z)=\sigma(w^\top \tilde z+b),\quad
  \tilde z=\frac{z-\mu}{\sigma_z},\quad \sigma(t)=\tfrac1{1+e^{-t}} .$$

- **Radial detector:** features $z=\log P(\rho)\in\mathbb{R}^{N/2}$. This is the
  cue the attack overwrites.
- **Residual detector:** features $z=\big[p_{90},p_{95},p_{99},\max,\mathrm{kurt},\mathrm{mean}\big]$
  of $R$ over $r\ge20$. Survives the attack.

**Evasion rate** $=$ fraction of matched fakes the detector now calls real
$=\frac1{|\text{matched}|}\sum \mathbf 1[\hat y=0]$.

---

## 3. Work log (what is built + verified, CPU only)

Reusable code:
- `src/spectral.py` — `radial_profile`, `radial_target`, `spectral_match`
  (DC-preserving, gain-clipped, phase-preserving), `spectral_report`.
- `src/detector.py` — `radial_features`, `residual_features`, `LogReg`,
  `run_evasion`.

Notebooks (paper sections 4–5):
- `notebooks/Spectral Matching - Closing the Gap.ipynb`
- `notebooks/Spectral Detectors and Evasion.ipynb`

Verified locally on synthetic 1/f-real vs (HF-deficit + grid) fake, no GPU:

| quantity | real | fake | matched |
|---|---|---|---|
| high-band mean | 3995 | 266 | 2962 |
| log–log slope | −0.91 | −2.10 | −1.05 |
| high-band gap (real/·) | — | **15×** | **1.35×** |
| phase agreement vs fake | — | — | **0.999** |

Evasion (60 real / 60 fake / 60 matched):

| detector | clean acc | fake recall | evasion (matched→real) |
|---|---|---|---|
| radial   | 1.00 | 1.00 | **1.00** (fully fooled) |
| residual | 1.00 | 1.00 | **0.00** (still catches) |

---

## 4. Where a GPU is needed, and roughly how much

**Everything above is CPU.** The matching, the detectors, the evasion harness,
and the figures all run on a laptop. A GPU is needed for exactly one thing:

> **Generating real fake images** (Stable Diffusion / SDXL) to replace the
> synthetic stand-ins, so the paper's numbers are on real data.

CIFAR-10 download (the reals) is network/CPU only — no GPU.

### Approximate GPU budget (generation only)

| model | precision | VRAM (min) | ~time / image | 200 images | good fit |
|---|---|---|---|---|---|
| SD 1.5 @ 512² | fp16, 50 steps | ~4–6 GB | ~2–4 s (T4) | ~10–15 min | Colab **free T4** |
| SD 1.5 @ 512² | fp16 | ~5 GB | ~0.6–1 s (A100) | ~3–5 min | A100/L4 |
| SDXL base @ 1024² | fp16 | ~10–12 GB | ~10–20 s (T4) | ~40–60 min | T4 (tight) / L4 / A100 |
| SDXL base+refiner | fp16 | ~14–16 GB | ~20–30 s (T4) | ~70–100 min | **A100** preferred |

Rules of thumb:
- A **free Colab T4 (16 GB)** is enough for SD 1.5 comfortably and SDXL base if
  you keep batch size at 1 and enable attention slicing / `enable_vae_tiling()`.
- For SDXL at scale or many generators, use **L4 (24 GB)** or **A100 (40 GB)**.
- Cost ballpark: a couple hundred images is **well under an hour** of GPU on any
  of these — single-digit dollars on paid Colab/cloud, or free on Colab T4.

### What to run next (the part that needs the GPU)
1. In `notebooks/Spectral Detectors and Evasion.ipynb`, set `USE_SYNTHETIC=False`
   and feed it the real CIFAR images + SD/SDXL fakes from Notebook 1.
2. Re-run the evasion table on real data → that table is the paper's headline.
3. (Optional) add SDXL as a second generator to show the result generalizes.
