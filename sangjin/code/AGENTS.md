# Vectorization Comparison Guide

- Keep this experiment separate from `1차년도_final`; do not overwrite prior runs.
- Compare BF16 VaultGemma methods because direct `vmap+grad` is incompatible with bitsandbytes 4-bit autograd.
- Preserve the shared data split, response-only sequence loss, Poisson sampling, clipping, noise, and PRV accounting.
- Label direct `vmap+grad`, ExpandedWeights, Ghost, and FastDP separately.
- Run smoke tests before full runs and retain failed run logs.
- Report resource metrics from standalone runs only; parallel runs are compatibility or utility runs.
- Record target/actual epsilon, sigma, q, steps, delta, loss, time, throughput, and peak VRAM.
