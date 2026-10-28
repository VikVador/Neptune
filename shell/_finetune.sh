# Moving to the root of the repository, whatever the current directory
cd "$(dirname "$0")/.."

# Moving to the jobs directory
cd experiments/forecast-ambient-increments/

# Launching the fine-tuning on rollouts
python -u finetuning.py --config configs/finetuning.yml

# Moving back to the root directory
cd ../../

# Observing the queue
watch squeue --me
