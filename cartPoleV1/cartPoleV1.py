import gymnasium as gym
import numpy as np
from tqdm import tqdm
import time
import wandb
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# This class makes sure to convert continuous space into discretized space.
class Tiling:
  def __init__(self, env, numTilings=8, numTiles=10):
    self.numTilings = numTilings
    self.numTiles = numTiles

    state, _ = env.reset()

    # Define limits
    cart_position_limit = 4.8
    pole_angle_limit = (24 * np.pi) / 180  # 24 degrees in radians
    cart_velocity_limit = 10
    pole_angular_velocity_limit = np.pi  # 3.1416 rad/s

    # Define low, high limits
    self.low = np.array([-cart_position_limit, -cart_velocity_limit, -pole_angle_limit, -pole_angular_velocity_limit])
    self.high = np.array([cart_position_limit, cart_velocity_limit, pole_angle_limit, pole_angular_velocity_limit])

    # Define offsets for tilings
    self.offsets = [
        (high - low) / self.numTiles * np.linspace(0, 1, self.numTilings, endpoint=False)
        for low, high in zip(self.low, self.high)
    ]

    # Create tilings
    self.tilings = self.createTilings()

  def getTilings(self):
    return self.tilings

  def createTilingGrid(self, low, high, bins, offsets):
    """ Create a single tiling grid """
    return [np.linspace(low[dim], high[dim], bins + 1)[1:-1] + offsets[dim] for dim in range(len(low))]

  def createTilings(self):
    """ Create multiple tiling grids with offsets """
    tilings = []
    for i in range(self.numTilings):
        offsets_i = [offset[i] for offset in self.offsets]  # Pick correct offset for each tiling
        tilings.append(self.createTilingGrid(self.low, self.high, self.numTiles, offsets_i))
    return tilings

  def discretize(self, state):
    """ Convert continuous state into discrete indices across all tilings """
    discretized_indices = []
    for tiling in self.tilings:
        tile_indices = []
        for dim, bins in enumerate(tiling):
            tile_indices.append(np.digitize(state[dim], bins))  # Assign bin index for each dimension
        discretized_indices.append(tuple(tile_indices))
    return discretized_indices


# Implementing sarsa using linear approximator and tiling.
class sarsaAgent:
  def __init__(self,env,alpha=0.1,gamma=0.99,epsilon_start=1.0,epsilon_decay=0.995,epsilon_min=0.0001,numTilings=2,numTiles=5,log_wandb=False):
    self.env = env
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon = epsilon_start
    self.epsilon_decay = epsilon_decay
    self.epsilon_min = epsilon_min
    self.numTilings = numTilings
    self.numTiles = numTiles
    self.log_wandb = log_wandb

    self.tiling = Tiling(env,numTilings,numTiles)
    self.numActions = self.env.action_space.n

    # Define feature representation using tile coding
    self.numFeatures = self.numTilings * self.numTiles** 4
    self.weights = np.zeros((self.numActions, self.numFeatures))  # Weights for each action

  def featureVector(self, state, action):
    """ Convert a continuous state into a binary feature vector. """
    discretized_indices = self.tiling.discretize(state)
    featureVector = np.zeros(self.numTilings * self.numTiles**4)

    for i, ds in enumerate(discretized_indices):
        index = sum(ds[d] * (self.numTiles**d) for d in range(len(ds)))  # Multi-dim index
        feature_index = i * self.numTiles**4 + index
        featureVector[feature_index] = 1  # Activate feature

    return featureVector

  def getQvalues(self,state,action):
    """ Approximates Q(s, a) = θ^T * φ(s, a) """
    featureVector = self.featureVector(state,action)
    return np.dot(self.weights[action], featureVector)

  def choose_epsilonGreedy_action(self,state):
    """ Chooses an action using ε-greedy policy. """
    if np.random.rand() < self.epsilon:
        return self.env.action_space.sample()  # Random action
    else:
        q_values = [self.getQvalues(state, a) for a in range(self.numActions)]
        return np.argmax(q_values)  # Best action

  def update(self,state,action,reward,next_state,next_action):
    """ SARSA update rule with linear function approximation """
    phi = self.featureVector(state, action)
    q_sa = self.getQvalues(state, action)
    q_next = self.getQvalues(next_state, next_action)

    td_target = reward + self.gamma * q_next
    td_error = td_target - q_sa
    self.weights[action] += (self.alpha/self.numTilings) * td_error * phi

  def train(self,num_episodes):
    episode_rewards = np.zeros(num_episodes)
    steps_to_completion = np.zeros(num_episodes)

    for i in tqdm(range(num_episodes)):
      totalReward = 0
      steps = 0

      state,_ = self.env.reset() #resetting the environment
      action = self.choose_epsilonGreedy_action(state)
      done = False

      while not done:
        next_state,reward,terminated,truncated,info = self.env.step(action)
        next_action = self.choose_epsilonGreedy_action(next_state)

        # Update weights using SARSA
        self.update(state, action, reward, next_state, next_action)

        state = next_state
        action = next_action

        totalReward += reward
        steps+=1

        done = terminated or truncated


      episode_rewards[i] = totalReward
      steps_to_completion[i] = steps

      #epsilon decay
      self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)

      if i % 100 == 99:
        wandb.log({"episode": i+1, "episodic_return": episode_rewards[i]})
        avg_reward = np.mean(episode_rewards[max(0, i-100):i+1])
        print(f"Episode {i+1}/{num_episodes}, Avg Reward (Last 100): {avg_reward:.2f}, Epsilon: {self.epsilon:.4f}")

    return episode_rewards,steps_to_completion


# Implementing qLearning using linear approximator and tiling.
class QLearningAgent:
  def __init__(self, env, alpha=0.1, gamma=0.99, tau_start=1.0, tau_decay=0.995, tau_min=0.05,numTilings=8, numTiles=10,log_wandb=False):
    self.env = env
    self.alpha = alpha
    self.gamma = gamma
    self.tau = tau_start
    self.tau_decay = tau_decay
    self.tau_min = tau_min
    self.log_wandb = log_wandb

    self.numTilings = numTilings
    self.numTiles = numTiles

    self.tiling = Tiling(env, numTilings, numTiles)
    self.numActions = self.env.action_space.n

    # Feature vector size: numTilings * numTiles * number_of_state_variables
    self.numFeatures = self.numTilings * self.numTiles ** self.env.observation_space.shape[0]
    # Weight vector for each action.
    self.weights = np.zeros((self.numActions, self.numFeatures))

  def featureVector(self, state, action):
    """ Convert a continuous state into a binary feature vector. """
    discretized_indices = self.tiling.discretize(state)
    featureVector = np.zeros(self.numTilings * self.numTiles**4)

    for i, ds in enumerate(discretized_indices):
        index = sum(ds[d] * (self.numTiles**d) for d in range(len(ds)))  # Multi-dim index
        feature_index = i * self.numTiles**4 + index
        featureVector[feature_index] = 1  # Activate feature

    return featureVector

  def getQvalue(self, state, action):
    """ Compute Q(s, a) = θ^T * φ(s, a). """
    phi = self.featureVector(state, action)
    return np.dot(self.weights[action], phi)

  def choose_action_softmax(self, state):
    """ Choose an action using a softmax (Boltzmann) policy. """
    q_values = np.array([self.getQvalue(state, a) for a in range(self.numActions)])
    # For numerical stability, subtract the maximum Q-value.
    exp_values = np.exp((q_values - np.max(q_values)) / self.tau)
    probs = exp_values / np.sum(exp_values)
    return np.random.choice(self.numActions, p=probs)

  def update(self, state, action, reward, next_state):
    """ Q-learning update rule with linear function approximation. """
    phi = self.featureVector(state, action)
    q_sa = self.getQvalue(state, action)
    # For Q-learning, we use the maximum Q-value over next actions.
    q_next_max = max([self.getQvalue(next_state, a) for a in range(self.numActions)])
    td_target = reward + self.gamma * q_next_max
    td_error = td_target - q_sa
    self.weights[action] += (self.alpha/self.numTilings) * td_error * phi

  def train(self, num_episodes):
    episode_rewards = np.zeros(num_episodes)
    steps_to_completion = np.zeros(num_episodes)

    for i in tqdm(range(num_episodes)):
      totalReward = 0
      steps = 0
      state, _ = self.env.reset()
      done = False

      while not done:
          action = self.choose_action_softmax(state)
          next_state, reward, terminated, truncated, _ = self.env.step(action)
          done = terminated or truncated

          self.update(state, action, reward, next_state)

          state = next_state
          totalReward += reward
          steps += 1

      episode_rewards[i] = totalReward
      steps_to_completion[i] = steps

      # Decay the temperature
      self.tau = max(self.tau_min, self.tau * self.tau_decay)

      if i % 100 == 99:
        wandb.log({"episode": i+1, "episodic_return": episode_rewards[i]})
        avg_reward = np.mean(episode_rewards[max(0, i-100):i+1])
        print(f"Episode {i+1}/{num_episodes}, Avg Reward (Last 100): {avg_reward:.2f}, Tau: {self.tau:.4f}")

    return episode_rewards, steps_to_completion


## running sarsa using given env,seed,hyperparams.
## if using wandb logs episode and episodeic return in train part of algorithm.
def runSingleSarsaExperiment(env,seed,hyperparams,num_episodes,isWandb):
  np.random.seed(seed)
  env.reset(seed=seed)

  agent = sarsaAgent(env,
                     alpha=hyperparams.get("alpha",0.1),
                     gamma=hyperparams.get("gamma",0.99),
                     epsilon_start=hyperparams.get("epsilon_start",1.0),
                     epsilon_decay=hyperparams.get("epsilon_decay",0.995),
                     epsilon_min=hyperparams.get("epsilon_min",0.0001),
                     numTilings=hyperparams.get("numTilings",8),
                     numTiles=hyperparams.get("numTiles",10),
                     log_wandb=isWandb
                     )

  if isWandb==False:
      wandb.init(mode="disabled")
  rewards,steps = agent.train(num_episodes)
  return rewards


## running qLearning using given env,seed,hyperparams.
## if using wandb logs episode and episodeic return in train part of algorithm.
def runSingleQLearningExperiment(env,seed,hyperparams,num_episodes,isWandb):
  np.random.seed(seed)
  env.reset(seed=seed)

  agent = QLearningAgent(env,
                        alpha=hyperparams.get("alpha",0.1),
                        gamma=hyperparams.get("gamma",0.99),
                        tau_start=hyperparams.get("tau_start",1.0),
                        tau_decay=hyperparams.get("tau_decay",0.995),
                        tau_min=hyperparams.get("tau_min",0.0001),
                        numTilings=hyperparams.get("numTilings",8),
                        numTiles=hyperparams.get("numTiles",10),
                        log_wandb=isWandb
                        )

  if isWandb==False:
      wandb.init(mode="disabled")
  rewards,steps = agent.train(num_episodes)
  return rewards

## running multiple seeds for a particular config to deal with stochasticity.
## returns reward from all runs. 
def runMultipleSeeds(algorithm,env,hyperparams,num_episodes,isWandb):
  rewardArr = []

  for seed in range(5):
    print(f"Running iteration {seed+1}")
    if algorithm == "sarsa":
      rewards = runSingleSarsaExperiment(env,seed,hyperparams,num_episodes,isWandb)
    else:
      rewards = runSingleQLearningExperiment(env,seed,hyperparams,num_episodes,isWandb)
    rewardArr.append(rewards)

  return rewardArr

def moving_average(data, window_size=100):
  """Compute running average using convolution."""
  return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def compareBestSarsaVsBestQLearning(sarsa_rewards,qlearning_rewards,num_episodes):
  # Compute mean and standard deviation over seeds for each episode.
  mean_rewards_sarsa = np.mean(sarsa_rewards, axis=0)
  std_rewards_sarsa = np.std(sarsa_rewards, axis=0)
  mean_rewards_qlearning = np.mean(qlearning_rewards, axis=0)
  std_rewards_qlearning = np.std(qlearning_rewards, axis=0)

  # Compute running averages (using a window of 100 episodes)
  running_avg_sarsa = moving_average(mean_rewards_sarsa, window_size=100)
  running_avg_qlearning = moving_average(mean_rewards_qlearning, window_size=100)

  # Compute cumulative regret. We assume an optimal reward of 500 per episode.
  optimal_reward = 500
  instant_regret_sarsa = optimal_reward - mean_rewards_sarsa
  cumulative_regret_sarsa = np.cumsum(instant_regret_sarsa)

  instant_regret_qlearning = optimal_reward - mean_rewards_qlearning
  cumulative_regret_qlearning = np.cumsum(instant_regret_qlearning)

  episodes = np.arange(num_episodes)

  # Plotting: We'll create 3 subplots:
  # 1. Episodic Return vs Episode
  # 2. Running Average Reward
  # 3. Cumulative Regret

  fig, axs = plt.subplots(3, 1, figsize=(10, 18))
  timestamp = int(time.time())

  # 1. Episodic Return
  axs[0].plot(episodes, mean_rewards_sarsa, label="SARSA", color="blue")
  axs[0].fill_between(episodes,
                      mean_rewards_sarsa - std_rewards_sarsa,
                      mean_rewards_sarsa + std_rewards_sarsa,
                      color="blue", alpha=0.2)
  axs[0].plot(episodes, mean_rewards_qlearning, label="Q-Learning", color="red")
  axs[0].fill_between(episodes,
                      mean_rewards_qlearning - std_rewards_qlearning,
                      mean_rewards_qlearning + std_rewards_qlearning,
                      color="red", alpha=0.2)
  axs[0].set_xlabel("Episode")
  axs[0].set_ylabel("Episodic Return")
  axs[0].set_title("Episodic Return vs Episode")
  axs[0].legend()

  # 2. Running Average Reward
  # Adjust x-axis for moving average (shorter by window_size - 1).
  episodes_ma = np.arange(len(running_avg_sarsa))
  axs[1].plot(episodes_ma, running_avg_sarsa, label="SARSA Running Avg", color="blue")
  axs[1].plot(episodes_ma, running_avg_qlearning, label="Q-Learning Running Avg", color="red")
  axs[1].set_xlabel("Episode")
  axs[1].set_ylabel("Running Average Reward")
  axs[1].set_title("Running Average Reward (Window=100)")
  axs[1].legend()

  # 3. Cumulative Regret
  axs[2].plot(episodes, cumulative_regret_sarsa, label="SARSA Cumulative Regret", color="blue")
  axs[2].plot(episodes, cumulative_regret_qlearning, label="Q-Learning Cumulative Regret", color="red")
  axs[2].set_xlabel("Episode")
  axs[2].set_ylabel("Cumulative Regret")
  axs[2].set_title("Cumulative Regret (Optimal Reward=500)")
  axs[2].legend()

  plt.tight_layout()
  plt.savefig(f"comparison_plot_{timestamp}.png")
  plt.show()

def plot_episodic_return(rewards):
    plt.figure(figsize=(12, 8))
    mean_rewards = np.mean(rewards, axis=0)
    std_rewards = np.std(rewards, axis=0)
    episodes = np.arange(len(mean_rewards))
    
    plt.plot(episodes, mean_rewards, color='b', label='Mean Reward')
    plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, color='b', alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Episodic Return")
    plt.title("Episodic Return vs Episode")
    plt.legend()
    
    filename = f"episodic_return_{int(time.time())}.png"
    plt.savefig(filename)
    plt.close()

def plot_running_average(rewards):
    window = 100
    plt.figure(figsize=(12, 8))
    
    mean_rewards = np.mean(rewards, axis=0)
    running_avg = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
    episodes = np.arange(window - 1, len(mean_rewards))
    
    plt.plot(episodes, running_avg, color='b', label='Running Average')
    plt.xlabel("Episode")
    plt.ylabel("Running Average Reward")
    plt.title(f"Running Average Reward (window = {window} episodes)")
    plt.legend()
    
    filename = f"running_average_{int(time.time())}.png"
    plt.savefig(filename)
    plt.close()

def plot_cumulative_regret(rewards):
    optimal_reward = 500
    
    mean_rewards_sarsa = np.mean(rewards, axis=0)
    instant_regret_sarsa = optimal_reward - mean_rewards_sarsa
    cumulative_regret_sarsa = np.cumsum(instant_regret_sarsa)
    episodes = np.arange(len(cumulative_regret_sarsa))
    
    plt.plot(episodes, cumulative_regret_sarsa, color="blue", label="Cumulative Regret")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Regret")
    plt.title("Sarsa Cumulative Regret (Optimal Reward=500)")
    plt.legend()
    
    filename = f"cumulative_regret_{int(time.time())}.png"
    plt.savefig(filename)
    plt.close()


def saveSarsaRewards(sarsa_rewards,num_episodes):
  sarsa_rewards_df = pd.DataFrame(sarsa_rewards)
  sarsa_rewards_df.columns = [f"Episode_{i+1}" for i in range(num_episodes)]
  sarsa_rewards_df.insert(0, "Seed", range(1, len(sarsa_rewards) + 1))

  # Save to Excel
  sarsa_rewards_df.to_excel("sarsa_rewards.xlsx", index=False)

  print("SARSA rewards saved to sarsa_rewards.xlsx")

def saveQLearningRewards(qLearning_rewards,num_episodes):
  qlearning_rewards_df = pd.DataFrame(qLearning_rewards)
  qlearning_rewards_df.columns = [f"Episode_{i+1}" for i in range(num_episodes)]
  qlearning_rewards_df.insert(0, "Seed", range(1, len(qLearning_rewards) + 1))

  # Save to Excel
  qlearning_rewards_df.to_excel("qlearning_rewards.xlsx", index=False)

  print("Q-Learning rewards saved to qlearning_rewards.xlsx")
def main():
    parser = argparse.ArgumentParser(description="Cart Pole V1 which can be trained with sarsa and QLearning")
    parser.add_argument("--algo", type=str, choices=["q_learning", "sarsa", "both"], required=True, help="Algorithm to use: q_learning, sarsa, or both to compare between best hyperParams of sarsa and qLearning")
    parser.add_argument("--env_name", type=str, default="CartPole-v1", help="currently CartPole-v1")
    parser.add_argument("--num_episodes", type=int, default=2500, help="number of training episodes")
    parser.add_argument("--tiles", type=int, default=8, help="number of splits in observation space")
    parser.add_argument("--tilings", type=int, default=10, help="number of different tiles with minor shift in position")
    parser.add_argument("--num_runs", type=int, default=5, help="number of runs to average over")
    parser.add_argument("--alpha", type=float, default=0.1, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount parameter (gamma)")
    parser.add_argument("--initial_tau", type=float, default=1.0, help="initial temp softmax")
    parser.add_argument("--tau_decay", type=float, default=0.999, help="temp decay softmax")
    parser.add_argument("--min_tau", type=float, default=0.001, help="min temp softmax")
    parser.add_argument("--initial_epsilon", type=float, default=1.0, help="initial epsilon epsilon greedy")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="epsilon decay epsilon greedy")
    parser.add_argument("--min_epsilon", type=float, default=0.001, help="min epsilon epilon greedy")
    
    np.random.seed(42)  
    args = parser.parse_args()

    numTiles = args.tiles
    numTilings = args.tilings
    alpha = args.alpha
    gamma = args.gamma

    epsilon_start = args.initial_epsilon
    epsilon_min = args.min_epsilon
    epsilon_decay = args.epsilon_decay

    tau_start = args.initial_tau
    tau_min = args.min_tau
    tau_decay = args.tau_decay

    numEpisodes = args.num_episodes
    currEnv = gym.make(args.env_name)

    q_rewards = None
    sarsa_rewards = None

    sarsa_hyperparams = {
    "alpha": alpha,
    "gamma": gamma,
    "epsilon_start": epsilon_start,
    "epsilon_decay": epsilon_decay,
    "epsilon_min": epsilon_min,
    "numTilings": numTilings,   
    "numTiles": numTiles
    }

    qLearning_hyperparams = {
    "alpha": alpha,
    "gamma": gamma,
    "tau_start": tau_start,
    "tau_decay": tau_decay,
    "tau_min": tau_min,
    "numTilings": numTilings,
    "numTiles": numTiles
    }

    best_sarsa_hyperparams = {
    "alpha": 0.1,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.001,
    "numTilings": 10,
    "numTiles": 8
    }

    best_qLearning_hyperparams = {
    "alpha": 0.5,
    "gamma": 0.99,
    "tau_start": 1.0,
    "tau_decay": 0.999,
    "tau_min": 0.001,
    "numTilings": 10,
    "numTiles": 8
    }

    if args.algo in ["sarsa"]:
      print("Running Sarsa ....")
      sarsa_rewards = runMultipleSeeds(algorithm="sarsa",env=currEnv,hyperparams=sarsa_hyperparams,num_episodes=numEpisodes,isWandb=False)
      plot_episodic_return(sarsa_rewards)
      plot_running_average(sarsa_rewards)
      plot_cumulative_regret(sarsa_rewards)
      saveSarsaRewards(sarsa_rewards,numEpisodes)
    elif args.algo in ["q_learning"]:
      print("Running Q Learning ....")
      qLearning_rewards = runMultipleSeeds(algorithm="qLearning",env=currEnv,hyperparams=qLearning_hyperparams,num_episodes=numEpisodes,isWandb=False)
      plot_episodic_return(qLearning_rewards)
      plot_running_average(qLearning_rewards)
      plot_cumulative_regret(qLearning_rewards)
      saveQLearningRewards(qLearning_rewards,numEpisodes)
    else:
      print("Running Sarsa and Q Learning For best hyperparams of both respectively....")
      print("Starting Sarsa ....")
      sarsa_rewards = runMultipleSeeds(algorithm="sarsa",env=currEnv,hyperparams=best_sarsa_hyperparams,num_episodes=numEpisodes,isWandb=False)
      saveSarsaRewards(sarsa_rewards,numEpisodes)
      print("Starting Q-Learning ....")
      qLearning_rewards = runMultipleSeeds(algorithm="qLearning",env=currEnv,hyperparams=best_qLearning_hyperparams,num_episodes=numEpisodes,isWandb=False)
      saveQLearningRewards(qLearning_rewards,numEpisodes)
      compareBestSarsaVsBestQLearning(sarsa_rewards,qLearning_rewards,num_episodes=numEpisodes)

if __name__ == "__main__":
    main()
