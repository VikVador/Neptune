```text
     ┌───────┐
     │ ◉   ◉ │       C   L   A   U   D   E
     │   ▿   │
     └─┬───┬─┘       Resources: my projects, my supervisors and the
     ┌─┴───┴─┐       libraries to know and reuse before coding.
     │ ▓▓▓▓▓ │
     └─┘   └─┘
```

# RESOURCES

**Last checked: 2026-10-27** (every link below was opened and verified on that day).

**Contents**

0. In short
1. Projects
2. People
3. Libraries
4. Documentation
5. Monitoring runs (WandB)

---

## 0. IN SHORT

1. **Two projects**: Heimdall (global ocean, NYU, cluster Torch) and Neptune (Black Sea, ULiège, cluster Lucia).
2. **Three supervisors**: Gilles Louppe (SAIL) and Marilaure Grégoire (MAST) at ULiège, Laure Zanna (M²LInES) at NYU.
3. **Check the libraries before coding**: diffusion → `azula`, flows → `zuko`, autoencoders and losses → `shaggy`,
   metrics → `albus`.
4. **You can read my WandB runs yourself** (§ 5): check the curves before asking me for screenshots.

---

## 1. PROJECTS

| Project  | Where        | Repository                                                           | Cluster   |
| -------- | ------------ | -------------------------------------------------------------------- | --------- |
| Heimdall | NYU, M²LInES | [github.com/VikVador/Heimdall](https://github.com/VikVador/Heimdall) | NYU Torch |
| Neptune  | ULiège       | [github.com/VikVador/Neptune](https://github.com/VikVador/Neptune)   | Lucia     |

### Heimdall

*"Exploring the Limits of Global Ocean State Estimation from Satellite Observations"*

| Item  | Details                                                                                                         |
| ----- | --------------------------------------------------------------------------------------------------------------- |
| Goal  | a latent generative model of the **global ocean state**, to find the limits of ocean predictability from space  |
| Data  | daily GLORYS reanalysis + daily surface observations (temperature, salinity, wind, chlorophyll; Copernicus L3)  |
| Model | one deterministic encoder per dataset, a stochastic transformer learning the joint dynamics in the latent space |
| Use   | long joint trajectories generated with score-based data assimilation                                            |
| Where | data 2000–2025 on the OSN pod of M²LInES, trainings on NYU Torch                                                |

### Neptune

*"Mastering the Black Sea Physics and Biogeochemistry with latent generative models"*

| Item   | Details                                                                                           |
| ------ | ------------------------------------------------------------------------------------------------- |
| Goal   | a generative **emulator of the Black Sea** forecasting the next daily state                       |
| Data   | daily NEMO 4.2.0 simulation (576 × 256 grid, 48 levels, 1998–2023), ERA5 atmospheric forcing      |
| Splits | training 1998–2017, validation 2018–2020, test 2021–2023                                          |
| State  | surface: wind stress, sea surface height                                                          |
|        | physics: currents, temperature, salinity                                                          |
|        | biogeochemistry: chlorophyll, oxygen, light, phosphate, silicate, nitrate                         |
| Model  | **FGN** (`experiments/forecast-ambient-increments`): two `shaggy` ConvAEs (surface 2D, ocean 3D)  |
|        | and an `azula` DiT, trained end to end to output the daily increments                             |
|        | **Latent** (`experiments/forecast-latent-states`): staged, encoders trained with the GME loss,    |
|        | then a noise-conditioned processor of the latent state, then a decoder                            |
| Train  | FGN: ensemble members with a fair CRPS, conditioned on the date (and ERA5 for the surface),       |
|        | then fine-tuned on autoregressive rollouts                                                        |
| Where  | data in `/gpfs/projects/acad/bsmfc/nemo4.2.0`, trainings on Lucia (jobs submitted with `dawgz`)   |

---

## 2. PEOPLE

| Person             | Role                                  | Lab                              | Links                                            |
| ------------------ | ------------------------------------- | -------------------------------- | ------------------------------------------------ |
| Gilles Louppe      | Supervisor, Professor at ULiège       | leads SAIL (Science with AI Lab) | [homepage][louppe], [SAIL][sail]                 |
| Marilaure Grégoire | Supervisor, Full Professor at ULiège  | leads MAST (ocean modelling)     | [MAST][mast], [FOCUS][focus]                     |
| Laure Zanna        | Supervisor at NYU (Courant Institute) | Lead PI of M²LInES               | [group][zanna], [M²LInES][m2lines], [team][team] |

[louppe]: https://glouppe.github.io/
[sail]: https://glouppe.github.io/sail/
[mast]: https://www.mast.uliege.be/cms/c_4938897/en/mast
[focus]: https://www.focus.uliege.be/cms/c_4262877/en/focus
[zanna]: https://zanna-researchteam.github.io/
[m2lines]: https://m2lines.github.io/
[team]: https://m2lines.github.io/team/

---

## 3. LIBRARIES

Check these **before writing new code**: what you need may already exist.

| Library  | For               | Author       |
| -------- | ----------------- | ------------ |
| `azula`  | diffusion models  | probabilists |
| `zuko`   | normalizing flows | probabilists |
| `shaggy` | autoencoders      | me           |
| `albus`  | metrics           | me           |

### azula — diffusion models

[GitHub](https://github.com/probabilists/azula) · [docs](https://azula.readthedocs.io)

One interface for diffusion models: a **noise schedule**, a **denoiser** and a **sampler**.

| Module           | Contents                                                                                            |
| ---------------- | --------------------------------------------------------------------------------------------------- |
| `azula.noise`    | `VPSchedule`, `VESchedule`, `CosineSchedule`, `RectifiedSchedule`, …                                |
| `azula.denoise`  | `GaussianDenoiser`, `KarrasDenoiser`, … (denoisers and training losses)                             |
| `azula.sample`   | `DDPMSampler`, `DDIMSampler`, `EulerSampler`, `HeunSampler`, `PCSampler`, …                         |
| `azula.guidance` | posterior sampling and guidance: `MMPSDenoiser`, `DPSSampler`, `CFGDenoiser`, …                     |
| `azula.nn`       | `DiT`, `ViT`, `UNet`, `MultiheadSelfAttention` with RoPE, `Patchify`, `RMSNorm`, `SwiGLU`, `ConvNd` |
| `azula.linalg`   | structured covariances, `cg`, `gmres`                                                               |
| `azula.plugins`  | pre-trained models (unstable)                                                                       |

### zuko — normalizing flows

[GitHub](https://github.com/probabilists/zuko) · [docs](https://zuko.readthedocs.io)

Normalizing flows as `nn.Module`s on top of `torch.distributions`: `flow(context).log_prob(x)`, `.sample()`.

| Module                        | Contents                                                   |
| ----------------------------- | ---------------------------------------------------------- |
| `zuko.flows`, `zuko.mixtures` | `MAF`, `NSF`, `RealNVP`, `NICE`, `NAF`, `CNF`, … and `GMM` |
| `zuko.transforms`             | the transformations the flows are built from               |
| `zuko.distributions`          | `DiagNormal`, `BoxUniform`, …                              |
| `zuko.nn`                     | `MLP`, `MaskedMLP`, …                                      |

### shaggy — autoencoders (mine)

[GitHub](https://github.com/VikVador/shaggy) · demos on the [`master-demo`](https://github.com/VikVador/shaggy/tree/master-demo) branch

| Module              | Contents                                                                                           |
| ------------------- | -------------------------------------------------------------------------------------------------- |
| `shaggy.models`     | `ConvEncoder`, `ConvDecoder`, `ConvAE`, `create_ConvAE`, `AutoEncoder`, `FusionAE`                 |
|                     | `ConvAE`: residual, N spatial dimensions, stride per level and per axis,                           |
|                     | patching, periodic axes, optional modulation                                                       |
| `shaggy.layers`     | `FRMSNorm`, `ResidualBlock`, `ResidualGroup`, `ResidualTrunk`, `ModulatedSequential`               |
| `shaggy.loss`       | `loss_reconstruction`, `loss_geometry_embedding` (GME), `loss_continuous_ranked_probability_score` |
| `shaggy.optimizers` | `SOAP`, `Muon`, `HybridMA`, `safe_gradient_step`                                                   |
| `shaggy.tools`      | `save`, `load`, `load_config`, `load_weights` (weights and configuration in one directory)         |

### albus — metrics (mine)

[GitHub](https://github.com/VikVador/albus) · [demo notebook](https://github.com/VikVador/albus/blob/main/notebook/demo.ipynb)

| Module               | Contents                                                                 |
| -------------------- | ------------------------------------------------------------------------ |
| `albus.metrics`      | all with masks: `mean_square_error`, `root_mean_square_error`, `skill`,  |
|                      | `standard_deviation`, `spread`, `spread_skill_ratio`,                    |
|                      | `continuous_ranked_probability_score`, `power_spectrum`                  |
| `albus.ocean.oxygen` | `hypoxia` (detection scores, ROC and PR curves)                          |
| `albus.tools`        | `apply_mask`, `axes`, `drop` (named axes such as `"T C Y X"`), `nanmean` |

---

## 4. DOCUMENTATION

| Topic                  | Link                                                                               |
| ---------------------- | ---------------------------------------------------------------------------------- |
| NYU Torch cluster      | [services.rt.nyu.edu](https://services.rt.nyu.edu/docs/hpc/getting_started/intro/) |
| Lucia cluster          | [doc.lucia.cenaero.be](https://doc.lucia.cenaero.be)                               |
| Copernicus Marine data | [data.marine.copernicus.eu](https://data.marine.copernicus.eu)                     |
| OSN (object storage)   | [openstoragenetwork.github.io/docs](https://openstoragenetwork.github.io/docs/)    |
| Weights & Biases       | [wandb.ai/sail-nyu](https://wandb.ai/sail-nyu)                                     |

---

## 5. MONITORING RUNS (WANDB)

You can read my runs on [wandb.ai/sail-nyu](https://wandb.ai/sail-nyu) with the Python API of `wandb`, from the project
environment: the key is already set up on the cluster (`wandb login`), so no setup is needed.

| Project                      | Runs                                                        |
| ---------------------------- | ----------------------------------------------------------- |
| `heimdall-training-encoders` | one run per encoder, e.g. `ENC_sst_lc128_tk5650__D9D7_164C` |
| `neptune-training-encoders`  | one run per encoder, e.g. `ENC_surface_lc128_tk576__F3D5`   |

| Logged metric                      | Meaning                  |
| ---------------------------------- | ------------------------ |
| `Training/Loss`, `Validation/Loss` | the losses               |
| `Informations/Learning Rate`       | learning rate (schedule) |
| `Informations/Gradient Norm`       | gradient norm (spikes)   |
| `Informations/Steps Update`        | optimizer updates done   |
| `Informations/Samples Seen`        | samples seen             |

```python
import wandb

api = wandb.Api()
runs = api.runs("sail-nyu/heimdall-training-encoders")
history = runs[0].history(keys=["Training/Loss", "Validation/Loss"])  # pandas DataFrame
```

**Rules**

- **Read only**: never delete, edit, tag or move a run.
- Never read nor print the key (`~/.netrc`): the library finds it by itself.
- Use it to check a training (divergence, overfitting, a run that stopped) and to compare runs with numbers.
