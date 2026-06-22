# Datasets — the exact SPAI evaluation set

This is precisely what SPAI (CVPR 2025) evaluated on: **13 generative models +
5 real sources = 18 sources**, one CSV each in
[spai/data](https://github.com/mever-team/spai/tree/main/data). We use the same
set so our results are directly comparable to the paper.

CSV format (all 18): `image,class,split` where **class 1 = fake, 0 = real**.

## Real images (5 sources)

| CSV | Source dataset | Download | Format |
|---|---|---|---|
| `real_raise.csv` | **RAISE-1k** | [loki.disi.unitn.it/RAISE](https://loki.disi.unitn.it/RAISE/download.html) | RAW `.tif` |
| `real_coco.csv` | COCO 2017 test | [cocodataset.org/#download](https://cocodataset.org/#download) | JPEG |
| `real_imagenet.csv` | ImageNet test | [Kaggle ILSVRC](https://www.kaggle.com/c/imagenet-object-localization-challenge/data) (login) | JPEG |
| `real_openimages.csv` | Open Images test | [Open Images v7](https://storage.googleapis.com/openimages/web/download_v7.html) | JPEG |
| `real_fodb.csv` | FODB | [FAU mmsec/fodb](https://faui1-files.cs.fau.de/public/mmsec/datasets/fodb/) | JPEG |

## Fake images (13 generators)

**9 come from Synthbuster** (one Zenodo download) and **4 from the SPAI authors'
Google Drive bundle.**

| CSV | Generator | Source |
|---|---|---|
| `fake_glide.csv` | GLIDE | **Synthbuster** — [Zenodo 10066460](https://zenodo.org/records/10066460) |
| `fake_sd13.csv` | Stable Diffusion 1.3 | Synthbuster |
| `fake_sd14.csv` | Stable Diffusion 1.4 | Synthbuster |
| `fake_sd2.csv` | Stable Diffusion 2 | Synthbuster |
| `fake_sdxl.csv` | Stable Diffusion XL | Synthbuster |
| `fake_dalle2.csv` | DALL·E 2 | Synthbuster |
| `fake_dalle3.csv` | DALL·E 3 | Synthbuster |
| `fake_firefly.csv` | Adobe Firefly | Synthbuster |
| `fake_mjv5.csv` | Midjourney v5 | Synthbuster |
| `fake_sd3.csv` | Stable Diffusion 3 | **SPAI Drive** — [bundle](https://drive.google.com/file/d/1no5T89h97TZvAKNCHKt2PQfKZ1UDtbI4/view?usp=sharing) |
| `fake_mjv61.csv` | Midjourney v6.1 | SPAI Drive |
| `fake_gigagan.csv` | GigaGAN | SPAI Drive |
| `fake_flux.csv` | Flux | SPAI Drive |

## Training set (only if re-training SPAI)

Not needed for our attack/evaluation. SPAI trained on Corvi et al.'s latent-
diffusion set (180k real + 180k LDM @ 256²) plus COCO + LSUN:
[grip-unina/DMimageDetection](https://github.com/grip-unina/DMimageDetection).

---

## ⭐ Recommended starting subset (no GPU, minimal download)

For the first spectral-matching experiment you do **not** need all 18 sources.
Start with the **scene-paired** pair — Synthbuster fakes were generated from
RAISE-1k scene prompts, so a fake has a corresponding real scene:

1. **RAISE-1k** (reals) → build the real spectral target `T(ρ)`.
2. **Synthbuster → SD1.4 + SDXL** (fakes) → the images to attack.

This is one Zenodo download + RAISE-1k, both direct (no Kaggle login), and
covers the generators most relevant to the high-frequency-deficit finding.
Add `flux` / `sd3` later from the Drive bundle to show the result generalizes
to newer generators.

Everything downstream (spectral matching, radial/residual detectors, evasion
table) runs on **CPU**. Only SPAI inference benefits from a small (<8GB) GPU,
and even that runs on CPU slowly since the backbone is just ViT-B/16.
