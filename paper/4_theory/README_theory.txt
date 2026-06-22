Theory papers — natural image statistics (1/f power law)

These are classic papers from the 1980s-1990s, not freely available as PDFs.
Cite from any university library or Google Scholar.

1. Field, D.J. (1987). Relations between the statistics of natural images
   and the response properties of cortical cells. Journal of the Optical
   Society of America A, 4(12), 2379-2394.
   -> The original 1/f^2 power law paper. Alpha ~ 2 for natural images.

2. Ruderman, D.L., & Bialek, W. (1994). Statistics of natural images:
   Scaling in the woods. Physical Review Letters, 73(6), 814.
   -> Confirms scale invariance of natural image statistics.

3. van der Schaaf, A., & van Hateren, J.H. (1996). Modelling the power
   spectra of natural images: Statistics and information.
   Vision Research, 36(17), 2759-2770.
   DOI: https://doi.org/10.1016/0042-6989(96)00002-8
   -> Extended analysis; available on ScienceDirect.

Why this matters for freqgen:
- Natural images have P(f) ~ f^-2 (power) = f^-1 (magnitude)
- AI fakes have a STEEPER slope (more negative) in the high band
- This is why the spectral_report() slope metric works as a fingerprint
- Our matched fakes move the slope back toward the real range (-1.05 vs -2.10)
