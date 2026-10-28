# Moving to the root of the repository, whatever the current directory
cd "$(dirname "$0")/.."

# Moving to the jobs directory
cd experiments/forecast-ambient-increments/

# Launching the training of the FGN
python -u train.py --config configs/train.yml

# Moving back to the root directory
cd ../../

# Observing the queue
watch squeue --me
