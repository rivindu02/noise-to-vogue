# Visual inspection

Both 32-sample grids were opened after the 8,000-update runs. They show recognizable tops, trousers, dresses, shoes, boots and bags. Some samples are blurry, distorted or ambiguous; this is an educational 28 × 28 baseline, with no FID, class-coverage or convergence claim.

The genuine denoising animations and viewer frames were checked in the browser. Generated and forward-corrupted images are independent, rather than a reconstruction pair. The microscope deliberately corrupts held-out images so injected noise is known.

At target signal variance ᾱ ≈ 0.5, held-out noise MSE is 0.041126 ± 0.000643 SE for linear and 0.040569 ± 0.000633 SE for cosine. Integer timesteps achieve slightly different signal levels; metrics.json records them. One paired training seed and per-example standard errors do not establish a schedule winner or account for training uncertainty.

Both models trained sequentially on a Quadro RTX 5000 with 16 GB of GPU memory. All controls match the planned protocol. Optimizer counters and final data/noise generator states were verified separately in training-verification.json. No extra training was needed after grid inspection.
