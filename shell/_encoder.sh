# Moving to the root of the repository, whatever the current directory
cd "$(dirname "$0")/.."

# Moving to the jobs directory
cd experiments/forecast-latent-states/

# Launching the trainings of the ocean and surface encoders
python -u encoders.py --config configs/encoder/encoder-ocean.yml
python -u encoders.py --config configs/encoder/encoder-surface.yml

# Moving back to the root directory
cd ../../

# Observing the queue
watch squeue --me
