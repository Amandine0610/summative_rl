# summative_reinforcement_learning

## Quick Start

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Record a single random demo (headless, produces MP4):

```powershell
python .\demo_random_record.py
```

Record multiple demos and get baseline stats:

```powershell
python .\scripts\record_multiple_demos.py --episodes 5
```

Run an interactive visual demo (opens a pygame window):

```powershell
python .\quick_demo.py
```

Run training experiments (example runner):

```powershell
python .\scripts\run_experiments.py
```

Hyperparameter sweep template (Optuna):

```powershell
python .\scripts\optuna_sweep_template.py --trials 8
```

Utilities
- `utils/experiment_logger.py`: simple wrapper for TensorBoard and optional `wandb` logging.

Files of interest
- `environment/custom_env.py`: the custom Gymnasium environment.
- `demo_random_record.py`: records a random episode to `demos/`.
- `scripts/record_multiple_demos.py`: batch records demos and prints baseline stats.
- `training/*.py`: training scripts for different algorithms (DQN, PPO, A2C, REINFORCE).

If you want me to also integrate wandb fully or add CLI flags to all training scripts, tell me and I will scaffold that next.

Experiment logging
- All training scripts now accept `--log-dir` to write TensorBoard logs and `--use-wandb` to enable optional Weights & Biases logging (if `wandb` installed).

Generate a compiled PDF of results (reads `training_results.json` in `models/`):

```powershell
python .\scripts\compile_results.py
```
