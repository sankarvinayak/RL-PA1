# Reinforcement Learning Training Script

This script trains a reinforcement learning agent using either the SARSA or Q-learning algorithm  
on a Gym environment. The results are visualized using Matplotlib.

## Requirements

Ensure you have all the dependencies installed by running:

```bash
pip install -r requirements.txt
```

## Usage

Run the script with the following command:

```bash
python dynamic_obstacle.py --algorithm <algorithm> [options]
```

### Arguments

| Argument        | Description                                             | Default Value |
|----------------|---------------------------------------------------------|--------------|
| `--algorithm`  |  Choose between `"sarsa"` or `"q-learning"` for training. |**Required**|
| `--env`        | (Optional) Gym environment name.                        | `"MiniGrid-Dynamic-Obstacles-5x5-v0"` |
| `--alpha`      | (Optional) Learning rate for SARSA.                      | `0.1` |
| `--gamma`      | (Optional) Discount factor.                              | `0.99` |
| `--epsilon`    | (Optional) Initial epsilon for SARSA.                    | `1.0` |
| `--epsilon_min`| (Optional) Minimum epsilon for SARSA.                    | `0.05` |
| `--epsilon_decay` | (Optional) Epsilon decay factor for SARSA.            | `0.995` |
| `--tau`        | (Optional) Initial temperature for Q-learning.           | `1.0` |
| `--tau_min`    | (Optional) Minimum temperature for Q-learning.           | `0.05` |
| `--tau_decay`  | (Optional) Temperature decay factor for Q-learning.      | `0.995` |
| `--episodes`   | (Optional) Number of training episodes.                  | `3000` |
| `--num_runs`   | (Optional) Number of independent training runs.          | `5` |

## Example Commands

Train using SARSA:

```bash
python train_agent.py --algorithm sarsa --env MiniGrid-Dynamic-Obstacles-5x5-v0 --alpha 0.1 --gamma 0.99 --epsilon 1.0 --epsilon_decay 0.995 --episodes 3000
```

Train using Q-learning:

```bash
python train_agent.py --algorithm q-learning --env MiniGrid-Dynamic-Obstacles-5x5-v0 --gamma 0.99 --tau 1.0 --tau_decay 0.995 --episodes 3000
```

## Results

The script saves plots of:
- **Rewards per episode**
- **Steps per episode**
- **Success rate**
