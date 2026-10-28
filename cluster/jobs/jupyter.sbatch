#!/bin/bash
# --------------
# Job parameters
# --------------
#SBATCH --account=bsmfc
#SBATCH --job-name=JUPYTER
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --partition=debug-gpu
#SBATCH --output=../logs/jupyter_notebook.log
#
# ----
# Main
# ----
# Removing all the modules previously loaded
module purge

# Creating an instance of the jupyter notebook
jupyter lab --ip='*' --notebook-dir=~/neptune/ --no-browser --port 8888
