I have also added .ipynb files to view results and for reference

# CartPoleV1 Environemnt

cartPoleV1.py can be used to train an agent using either Sarsa or -Learning algorithm on cartPoleV1 environment.
After running the script rewards per episode on each run(multiple runs to tackle stochasticity) are stored in excel files and plots are saved in running directory

## Requirements

Ensure you have all the dependencies installed by running:

```bash
pip install -r requirement.txt
```

## Usage

Run the script with the following command:

```bash
python cartPoleV1.py
```

running it without any arguments runs on best hyperparameters of both sarsa and QLearning

### Arguments

| Argument        | Description                                                           | Default Value |
|----------------|------------------------------------------------------------------------|--------------|
| `--algo`       |  Choose between `"sarsa"` or `"q-learning"` or both for training.      |**Required**|
| `--env_name`        | (Optional) Gym environment name.                                       | `"CartPole-v1"` |
| `--num_episodes` | (Optional) Number of episodes in each training run.                  | `2500` |
| `--alpha`      | (Optional) Learning rate .                                             | `0.1` |
| `--gamma`      | (Optional) Discount factor.                                            | `0.99` |
| `--initial_epsilon`    | (Optional) Initial epsilon for SARSA.                                  | `1.0` |
| `--min_epsilon`| (Optional) Minimum epsilon for SARSA.                                  | `0.001` |
| `--epsilon_decay` | (Optional) Epsilon decay factor for SARSA.                          | `0.995` |
| `--initial_tau`        | (Optional) Initial temperature for Q-learning.                         | `1.0` |
| `--min_tau`    | (Optional) Minimum temperature for Q-learning.                         | `0.001` |
| `--tau_decay`  | (Optional) Temperature decay factor for Q-learning.                    | `0.999` |
| `--num_runs`   | (Optional) Number of independent training runs.                        | `5` |
| `--tiles`      | (Optional) Number of splits in observation space along each dimension. | `5` |
| `--tilings`    | (Optional) Number of different tiles with minor shift in position.     | `5` |

## Example Commands

Train using SARSA:

```bash
python cartPoleV1.py --algo "sarsa" --alpha 0.1 --gamma 0.99 --epsilon_decay 0.995 --num_episodes 2500 --tiles 8 --tilings 10
```

Train using Q-learning:

```bash
python cartPoleV1.py --algo "q-learning" --gamma 0.99 --tau_decay 0.995 --episodes 2500 --tiles 8 --tilings 10
```

## Results

The script saves plots of:
- **Episode vs Episodic Return**
- **Cumulative Regret**
- **Running Avg with window of 100 runs**

Thes script also saves excel files containing rewards of the algorithm across multiple seeds.
