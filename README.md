<hr style="color:#808080;">
<p align="center"><b>N E P T U N E</b></p>
<hr style="color:#808080;">

To set-up everything, it is necessary to have access to a [Slurm](https://slurm.schedmd.com) cluster, to login to a [Weights & Biases](https://wandb.ai) account and to install the [neptune](neptune) module as a package. First, create a new Python environment, for example with [conda](https://docs.conda.io).

```
conda create -n neptune python=3.11
conda activate neptune
```

Then, install the [neptune](neptune) module as an [editable](https://pip.pypa.io/en/latest/topics/local-project-installs) package with its dependencies.

```
pip install --editable .[all] --extra-index-url https://download.pytorch.org/whl/cu121
```

Optionally, we provide [pre-commit hooks](pre-commit.yml) to automatically detect code issues.

```
pre-commit install --config pre-commit.yml
```
