"""
spectral.py - radial power-spectrum analysis and spectral matching.

freqgen finding (Notebook 1/2): real images have an azimuthally-averaged
Fourier magnitude profile that falls off smoothly (~1/f, slope ~= -2 in
log-log of the *power* / ~ -1 of the magnitude), while diffusion-generated
images preserve low frequencies but carry a high-frequency energy *deficit*
(the "21x gap" in the high band).

`spectral_match` rewrites a generated image's Fourier magnitude along each
radius so that its azimuthally-averaged magnitude profile matches a target
profile measured from real images, while leaving the *phase* untouched.
Phase carries image structure, so the content is preserved; only the
frequency-energy envelope (texture / noise statistics) is corrected.

This is an anti-forensics / detector-robustness experiment: it demonstrates
that a frequency-only detector built on the radial spectrum can be defeated by
a cheap post-process, which motivates multi-cue detection. It is *not* a way to
make undetectable fakes - phase-based, learned, and pixel-statistic detectors
are unaffected.

Pure numpy/Pillow; no torch / diffusers dependency, so it runs anywhere.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# Band edges (radius indices) used throughout freqgen, matching Notebook 1.
LOW_MID_EDGE = 20
MID_HIGH_EDGE = 60


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_gray(path: str, size: int = 256) -> np.ndarray:
    """Load an image as a float64 grayscale array of shape (size, size)."""
    img = Image.open(path).convert("L").resize((size, size))
    return np.asarray(img, dtype=np.float64)


def load_rgb(path: str, size: int = 256) -> np.ndarray:
    """Load an image as a float64 RGB array of shape (size, size, 3)."""
    img = Image.open(path).convert("RGB").resize((size, size))
    return np.asarray(img, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Radial spectrum
# --------------------------------------------------------------------------- #
def _radius_map(h: int, w: int) -> np.ndarray:
    """Integer radius of every pixel from the (fft-shifted) DC centre."""
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    return np.round(np.sqrt((y - cy) ** 2 + (x - cx) ** 2)).astype(int)


def radial_profile(channel: np.ndarray) -> np.ndarray:
    """
    Azimuthally-averaged Fourier *magnitude* profile of a 2-D array.

    Returns an array of length ``min(H, W) // 2`` where index == radius.
    Mirrors ``compute_spectral_metrics`` in Notebook 1 (mean of |FFT|, not |FFT|^2).
    """
    F = np.fft.fftshift(np.fft.fft2(channel))
    mag = np.abs(F)
    r = _radius_map(*channel.shape)
    max_r = min(channel.shape) // 2
    total = np.bincount(r.ravel(), weights=mag.ravel())
    count = np.bincount(r.ravel())
    return total[:max_r] / np.maximum(count[:max_r], 1)


def radial_target(paths, size: int = 256, color: bool = False) -> np.ndarray:
    """
    Average radial magnitude profile over a set of (real) images.

    color=False -> returns shape (R,)      (luminance target)
    color=True  -> returns shape (3, R)    (per-channel R,G,B targets)
    """
    profiles = []
    for p in paths:
        if color:
            arr = load_rgb(p, size)
            profiles.append(np.stack([radial_profile(arr[..., c]) for c in range(3)]))
        else:
            profiles.append(radial_profile(load_gray(p, size)))
    return np.mean(profiles, axis=0)


# --------------------------------------------------------------------------- #
# Spectral matching (the "make it pass" step)
# --------------------------------------------------------------------------- #
def _match_channel(
    channel: np.ndarray,
    target: np.ndarray,
    gain_clip=(0.1, 12.0),
    smooth: int = 3,
    preserve_dc: bool = True,
) -> np.ndarray:
    """Match one 2-D channel's radial magnitude profile to ``target``."""
    F = np.fft.fftshift(np.fft.fft2(channel))
    r = _radius_map(*channel.shape)
    max_r = len(target)

    src = radial_profile(channel)
    gain = target / (src + 1e-12)                      # per-radius magnitude gain

    if smooth and smooth > 1:                           # damp ringing from spiky gains
        k = np.ones(smooth) / smooth
        gain = np.convolve(gain, k, mode="same")
    gain = np.clip(gain, *gain_clip)
    if preserve_dc:
        gain[0] = 1.0                                  # keep DC -> overall brightness intact

    rr = np.clip(r, 0, max_r - 1)
    gmap = gain[rr]
    gmap[r >= max_r] = 1.0                              # leave diagonal corners untouched
    if preserve_dc:
        gmap[r == 0] = 1.0

    F_matched = F * gmap                                # scales magnitude, preserves phase
    out = np.fft.ifft2(np.fft.ifftshift(F_matched)).real
    return out


def spectral_match(
    img: np.ndarray,
    target: np.ndarray,
    gain_clip=(0.1, 12.0),
    smooth: int = 3,
    clip_out: bool = True,
    preserve_dc: bool = True,
) -> np.ndarray:
    """
    Spectrally correct a fake image so its radial magnitude profile matches
    ``target``. Phase (structure) is preserved.

    img    : (H, W) grayscale or (H, W, 3) RGB float array
    target : (R,) for grayscale, or (3, R) for RGB  [from ``radial_target``]
    """
    if img.ndim == 2:
        out = _match_channel(img, target, gain_clip, smooth, preserve_dc)
    else:
        out = np.stack(
            [_match_channel(img[..., c], target[c], gain_clip, smooth, preserve_dc)
             for c in range(img.shape[-1])],
            axis=-1,
        )
    return np.clip(out, 0, 255) if clip_out else out


# --------------------------------------------------------------------------- #
# Reporting (before/after, in Notebook 1's terms)
# --------------------------------------------------------------------------- #
def spectral_report(channel: np.ndarray) -> dict:
    """Band energies + log-log slope, comparable to Notebook 1's metrics."""
    prof = radial_profile(channel)
    freqs = np.arange(1, len(prof))
    slope, _ = np.polyfit(np.log(freqs + 1e-8), np.log(prof[1:] + 1e-8), 1)
    return {
        "low": float(prof[:LOW_MID_EDGE].mean()),
        "mid": float(prof[LOW_MID_EDGE:MID_HIGH_EDGE].mean()),
        "high": float(prof[MID_HIGH_EDGE:].mean()),
        "slope": float(slope),
        "profile": prof,
    }


def high_band_gap(real_channel: np.ndarray, fake_channel: np.ndarray) -> float:
    """real_high / fake_high - the '21x gap' number, >1 means fake is deficient."""
    return spectral_report(real_channel)["high"] / spectral_report(fake_channel)["high"]
