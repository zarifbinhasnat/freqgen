# freqgen

Frequency-domain analysis of real vs. AI-generated images using spectral methods (FFT, etc.).

## Structure

```
freqgen/
  notebooks/    # exploratory analysis, numbered sequentially
  src/          # stable, reusable modules extracted from notebooks
  results/      # saved plots, tables, .npy metric arrays
  paper/        # LaTeX manuscript + figures pulled from results/
  LOGBOOK.md    # dated running notes
```

## Getting Started

```bash
pip install -r src/requirements.txt
jupyter lab notebooks/
```

## Status

Active research — see [LOGBOOK.md](LOGBOOK.md) for running notes.
