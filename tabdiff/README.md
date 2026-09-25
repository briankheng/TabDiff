# TabDiff model code

Unmodified files from the authors' repository,
[MinkaiXu/TabDiff](https://github.com/MinkaiXu/TabDiff) at commit
`5ecdb3356261aea72716cc9a779f31d7ad083bf4`, under the MIT licence in `LICENSE`.

- `models/`: the mixed-type continuous-time diffusion and its learnable noise schedules
- `modules/`: the denoising network
- `configs/tabdiff_configs.toml`: the authors' default hyperparameters

Only the model code is included. The data loading, training loop and sampling
around it live in `../generator.py`.

Shi et al., *TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation*, ICLR 2025.
