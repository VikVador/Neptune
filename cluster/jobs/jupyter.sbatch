#!/bin/bash
# --------------
# Job parameters
# --------------
#SBATCH --account=bsmfc
#SBATCH --job-name=JUPYTER
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --partition=debug
#SBATCH --output=../logs/jupiter_notebook.log
#
# ----
# Main
# ----
# Removing all the modules previously loaded
module purge

# Creating an instance of the jupyter notebook
jupyter lab --ip='*' --notebook-dir=~/neptune/ --no-browser --port 8888
