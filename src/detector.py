"""
detector.py - spectral detectors + evasion harness for freqgen.

Two detectors, both CPU-only (numpy):

  * radial_features   -> log radial magnitude profile (length N/2).
    The detector the `spectral_match` attack is designed to defeat.

  * residual_features -> peakiness stats of the spectral residual
    R[u,v] = |X_s[u,v]| / mean_over_ring(|X_s|).
    Captures 2-D peaks (upsampling / checkerboard artifacts) that live
    *within* a frequency ring. Radial matching scales each ring uniformly,
    so it cannot remove these -> this detector survives the attack.

Classifier is a small standardized logistic regression (no sklearn dependency).
Convention: label 1 = fake, 0 = real. "Detected as fake" = prediction 1.
"""
from __future__ import annotations

import numpy as np

import spectral as sp


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def radial_features(channel: np.ndarray) -> np.ndarray:
    """Log radial magnitude profile -> the fingerprint the attack targets."""
    return np.log(sp.radial_profile(channel) + 1e-8)


def _kurtosis(v: np.ndarray) -> float:
    mu, sd = v.mean(), v.std() + 1e-12
    return float(((v - mu) ** 4).mean() / sd ** 4)


def residual_features(channel: np.ndarray) -> np.ndarray:
    """Peakiness of the spectral residual over mid+high rings (r >= 20)."""
    prof = sp.radial_profile(channel)
    F = np.fft.fftshift(np.fft.fft2(channel))
    A = np.abs(F)
    r = sp._radius_map(*channel.shape)
    ringmean = prof[np.clip(r, 0, len(prof) - 1)]
    R = A / (ringmean + 1e-8)
    v = R[r >= 20]
    return np.array([
        np.percentile(v, 90), np.percentile(v, 95), np.percentile(v, 99),
        v.max(), _kurtosis(v), v.mean(),
    ])


def build_features(imgs, kind: str) -> np.ndarray:
    """Stack features for a list of 2-D images. kind in {'radial','residual'}."""
    fn = radial_features if kind == "radial" else residual_features
    return np.stack([fn(im) for im in imgs])


# --------------------------------------------------------------------------- #
# Tiny logistic-regression classifier (standardized, L2-regularized)
# --------------------------------------------------------------------------- #
class LogReg:
    def __init__(self, lr=0.5, epochs=2000, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        y = np.asarray(y, dtype=np.float64)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-8
        Xs = (X - self.mu) / self.sd
        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            p = 1.0 / (1.0 + np.exp(-(Xs @ self.w + self.b)))
            self.w -= self.lr * (Xs.T @ (p - y) / n + self.l2 * self.w)
            self.b -= self.lr * (p - y).mean()
        return self

    def proba(self, X):
        Xs = (X - self.mu) / self.sd
        return 1.0 / (1.0 + np.exp(-(Xs @ self.w + self.b)))

    def predict(self, X):
        return (self.proba(X) >= 0.5).astype(int)


# --------------------------------------------------------------------------- #
# Evasion harness
# --------------------------------------------------------------------------- #
def accuracy(det, X, y) -> float:
    return float((det.predict(X) == np.asarray(y)).mean())


def evasion_rate(det, X_matched_fakes) -> float:
    """Fraction of matched fakes the detector now calls REAL (label 0)."""
    return float((det.predict(X_matched_fakes) == 0).mean())


def run_evasion(real_imgs, fake_imgs, matched_imgs, kind="radial", seed=0):
    """
    Train `kind` detector on a real/fake split, then measure how many matched
    fakes slip past. Returns a dict of headline numbers.
    """
    rng = np.random.default_rng(seed)
    Xr, Xf = build_features(real_imgs, kind), build_features(fake_imgs, kind)
    Xm = build_features(matched_imgs, kind)

    nr, nf = len(Xr), len(Xf)
    ri, fi = rng.permutation(nr), rng.permutation(nf)
    rtr, fte = ri[: nr // 2], fi[nf // 2:]
    ftr = fi[: nf // 2]
    rte = ri[nr // 2:]

    Xtr = np.vstack([Xr[rtr], Xf[ftr]])
    ytr = np.r_[np.zeros(len(rtr)), np.ones(len(ftr))]
    Xte = np.vstack([Xr[rte], Xf[fte]])
    yte = np.r_[np.zeros(len(rte)), np.ones(len(fte))]

    det = LogReg().fit(Xtr, ytr)
    return {
        "kind": kind,
        "clean_acc": accuracy(det, Xte, yte),          # real-vs-fake before attack
        "fake_recall": accuracy(det, Xf[fte], np.ones(len(fte))),  # fakes caught
        "evasion_rate": evasion_rate(det, Xm[fte]),    # matched fakes now called real
        "detector": det,
    }
