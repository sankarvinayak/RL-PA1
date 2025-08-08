
import numpy as np
import argparse
from collections import deque

import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

###############################################################Plotting
def get_unique_filename(base_filename):
    filename, ext = os.path.splitext(base_filename)
    counter = 1
    unique_filename = base_filename
    
    while os.path.exists(unique_filename):
        unique_filename = f"{filename}_{counter}{ext}"
        counter += 1
    
    return unique_filename

def plot_mean_std(avg_rewards, std_rewards, title="Training Performance", ylabel="Reward", filename="training_performance.png"):
    episodes = np.arange(len(avg_rewards))
    
    plt.figure()
    plt.plot(episodes, avg_rewards, label="Mean Reward", color="b", linewidth=2)
    plt.fill_between(episodes, avg_rewards - std_rewards, avg_rewards + std_rewards, color="b", alpha=0.2, label="±1 Std Dev")
    plt.xlabel("Episodes")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.6)
    
    unique_filename = get_unique_filename(filename)
    plt.savefig(unique_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {unique_filename}")

def plot_running_average(rewards, window_size=100, title="Running Average of Rewards", ylabel="Reward", filename="running_average.png"):
    episodes = np.arange(len(rewards))
    running_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    
    plt.figure()
    plt.plot(episodes, rewards, label="Original Rewards", color="gray", alpha=0.4)
    plt.plot(episodes[:len(running_avg)], running_avg, label=f"Running Avg ({window_size} episodes)", color="b", linewidth=2)
    plt.xlabel("Episodes")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    unique_filename = get_unique_filename(filename)
    plt.savefig(unique_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {unique_filename}")

def plot_cumulative_regret(target, avg_rw, filename="cumulative_regret.png"):
    optimal_rewards = np.full_like(avg_rw, target)
    cumulative_regret = np.cumsum(optimal_rewards - avg_rw)

    plt.figure()
    plt.plot(cumulative_regret, label="Cumulative Regret", color='red')
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Regret")
    plt.title("Cumulative Regret over Episodes")
    plt.legend()
    plt.grid()

    unique_filename = get_unique_filename(filename)
    plt.savefig(unique_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {unique_filename}")

def plot_cumulative_regret_both(target, q_learning_rewards, sarsa_rewards, labels=("Q-Learning", "SARSA")):
    optimal_rewards = np.full_like(q_learning_rewards, target)
    cumulative_regret_q = np.cumsum(optimal_rewards - q_learning_rewards)
    cumulative_regret_sarsa = np.cumsum(optimal_rewards - sarsa_rewards)
    plt.plot(cumulative_regret_q, label=f"Cumulative Regret ({labels[0]})", color='blue')
    plt.plot(cumulative_regret_sarsa, label=f"Cumulative Regret ({labels[1]})", color='red')
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Regret")
    plt.title("Cumulative Regret over Episodes (Q-Learning vs SARSA)")
    plt.legend()
    plt.grid()
    plt.show()
def plot_running_average_both(q_rewards, sarsa_rewards, window_size=100, labels=("Q-Learning", "SARSA")):
    episodes_q = np.arange(len(q_rewards))
    episodes_sarsa = np.arange(len(sarsa_rewards))
    running_avg_q = np.convolve(q_rewards, np.ones(window_size)/window_size, mode='valid')
    running_avg_sarsa = np.convolve(sarsa_rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(episodes_q, q_rewards, label=f"Original {labels[0]} Rewards", color="gray", alpha=0.3)
    plt.plot(episodes_q[:len(running_avg_q)], running_avg_q, label=f"Running Avg ({labels[0]})", color="b", linewidth=2)
    plt.plot(episodes_sarsa, sarsa_rewards, label=f"Original {labels[1]} Rewards", color="gray", alpha=0.3)
    plt.plot(episodes_sarsa[:len(running_avg_sarsa)], running_avg_sarsa, label=f"Running Avg ({labels[1]})", color="r", linewidth=2)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Running Average of Rewards (Q-Learning vs SARSA)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()








###############################################################################Q Table
class Split_to_grid:
    def __init__(self, env, n_bins=(20, 20)):
        self.env = env
        self.n_bins = n_bins
        self.space_min = env.observation_space.low
        self.space_max = env.observation_space.high
        self.space_w = (self.space_max - self.space_min) / n_bins
    def find_the_cell(self, val):
        clip_val = np.clip(val, self.space_min, self.space_max)
        bi = np.floor((clip_val - self.space_min) / self.space_w).astype(int)
        bi = np.clip(bi, 0, np.array(self.n_bins) - 1)
        cell_no = bi[0] * self.n_bins[1] + bi[1]
        return cell_no

class Qagent:
    def __init__(self,num_states,num_actions,alpha,gamma,initial_tau,tau_decay,min_tau):
        self.n_states=num_states
        self.n_actions=num_actions
        self.alpha=alpha
        self.gamma= gamma
        self.tau = initial_tau
        self.tau_decay = tau_decay
        self.min_tau = min_tau

        self.q_table = np.zeros((num_states, num_actions))
    def select_action(self, s):
        q_values = self.q_table[s]
        q_values = q_values - np.max(q_values)
        exp_q = np.exp(q_values / max(self.tau, 1e-10))
        if np.any(np.isnan(exp_q)) or np.any(np.isinf(exp_q)):
            return np.argmax(self.q_table[s])
        sum_exp_q = np.sum(exp_q)
        if sum_exp_q == 0:  
            return np.argmax(self.q_table[s])
        probabilities = exp_q / sum_exp_q
        try:
            action = np.random.choice(self.n_actions, p=probabilities)
            return action
        except:
            return np.argmax(self.q_table[s])
    def update(self, s, a, r, ns, done):
        if done:
            R = r
        else:
            R = r + self.gamma * np.max(self.q_table[ns])
        self.q_table[s, a] += self.alpha * (R - self.q_table[s, a])
    def decay_tau(self):
        self.tau = max(self.min_tau, self.tau * self.tau_decay)

def train_q_learning(env_name='MountainCar-v0', num_episodes=1000, n_bins=(20, 20),alpha=0.1, gamma=0.99, initial_tau=1.0,tau_decay=0.99, min_tau=0.01, num_runs=5):
    print("starting Qlearning")
    env = gym.make(env_name)
    mountain_car_grid = Split_to_grid(env, n_bins)
    num_states = np.prod(n_bins)
    num_actions = env.action_space.n
    avg_run_rewards = np.zeros(num_episodes, dtype=np.float32)  
    avg_run_lengths = np.zeros(num_episodes, dtype=np.float32) 
    valid_counts = np.zeros(num_episodes, dtype=np.int32) 
    all_rewards = np.full((num_runs, num_episodes), np.nan, dtype=np.float32)  
    for run in range(num_runs):
        np.random.seed(run)
        agent = Qagent(num_states, num_actions, alpha, gamma,initial_tau, tau_decay, min_tau)
        episode_rewards = np.zeros(num_episodes, dtype=np.float32) 
        episode_lengths = np.zeros(num_episodes, dtype=np.int32) 
        avg_rewards = deque(maxlen=100)  
        for episode in range(num_episodes):
            r_state, _ = env.reset()
            s = mountain_car_grid.find_the_cell(r_state)
            ep_r = 0
            ep_l = 0
            succ = False
            t_end = False
            while not (succ or t_end):
                a = agent.select_action(s)
                nxt, r, succ, t_end, _ = env.step(a)
                ns = mountain_car_grid.find_the_cell(nxt)
                agent.update(s, a, r, ns, succ or t_end)
                s = ns
                ep_r += r
                ep_l += 1
            agent.decay_tau()
            episode_rewards[episode] = ep_r
            episode_lengths[episode] = ep_l
            avg_rewards.append(ep_r)
            avg_reward = np.mean(avg_rewards)
            if len(avg_rewards) == 100 and avg_reward >= -110:
                print(f"run:{run} episode {episode+1}/{num_episodes}, avg Reward: {avg_reward:.2f}, tau: {agent.tau:.4f}")
                print(f" solved in {episode+1} episodes!")
                episode_rewards[episode+1:] = -110  # will fill the remaining timestp with -110 so that the figure will make sense
                break
            if (episode + 1) % 1000 == 0:
                print(f"run:{run} episode {episode+1}/{num_episodes}, avg Reward: {avg_reward:.2f}, tau: {agent.tau:.4f}")
        for ep in range(num_episodes):
            if episode_rewards[ep] != 0:  
                avg_run_rewards[ep] += episode_rewards[ep]
                avg_run_lengths[ep] += episode_lengths[ep]
                valid_counts[ep] += 1
                all_rewards[run, ep] = episode_rewards[ep]  
    env.close()
    avg_rewards_final = np.divide(avg_run_rewards, valid_counts, where=valid_counts > 0)
    avg_lengths_final = np.divide(avg_run_lengths, valid_counts, where=valid_counts > 0)
    std_rewards_final = np.nanstd(all_rewards, axis=0)
    return agent, avg_rewards_final, avg_lengths_final, std_rewards_final




class SARSAagent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.99,initial_epsilon=0.3, epsilon_decay=0.995, min_epsilon=0.01):
        self.n_states = num_states
        self.n_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros((num_states, num_actions))
    def select_action(self, s):
        if np.random.random() < self.epsilon:
            a = np.random.randint(self.n_actions)
        else:
            a = np.argmax(self.q_table[s])
        return a
    def update(self, s, a, r, ns, na, done):
        if done:
            R = r
        else:
            R = r + self.gamma * self.q_table[ns, na]
        self.q_table[s, a] += self.alpha * (R - self.q_table[s, a])
    def decay_eps(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
def train_sarsa(env_name='MountainCar-v0', n_episodes=1000, n_bins=(20, 20),alpha=0.1, gamma=0.99, initial_epsilon=0.3,epsilon_decay=0.995, min_epsilon=0.01, n_runs=5):
    print("training sarsa")
    env = gym.make(env_name)
    mountain_car_grid = Split_to_grid(env, n_bins)
    num_states = np.prod(n_bins)
    action_space = env.action_space.n
    avg_run_rw = np.zeros(n_episodes, dtype=np.float32)  
    avg_run_len = np.zeros(n_episodes, dtype=np.float32) 
    all_rewards = np.full((n_runs, n_episodes), np.nan, dtype=np.float32) 
    for run in range(n_runs):
        np.random.seed(run)
        agent = SARSAagent(num_states, action_space, alpha, gamma,initial_epsilon, epsilon_decay, min_epsilon)
        ep_rs = np.zeros(n_episodes, dtype=np.float32) 
        ep_lens = np.zeros(n_episodes, dtype=np.int32)  
        avg_rewards = deque(maxlen=100)  
        for episode in range(n_episodes):
            r_state, _ = env.reset()
            s = mountain_car_grid.find_the_cell(r_state)
            a = agent.select_action(s)
            ep_r = 0
            ep_len = 0
            succ = False
            t_end = False
            while not (succ or t_end):
                nxt, r, succ, t_end, _ = env.step(a)
                ns = mountain_car_grid.find_the_cell(nxt)
                na = agent.select_action(ns)
                agent.update(s, a, r, ns, na, succ or t_end)
                s = ns
                a = na
                ep_r += r
                ep_len += 1
            agent.decay_eps()
            ep_rs[episode] = ep_r
            ep_lens[episode] = ep_len
            avg_rewards.append(ep_r)
            all_rewards[run, episode] = ep_r  
            if (episode + 1) % 1000 == 0:
                avg_reward = np.mean(avg_rewards)
                print(f"run:{run} episode {episode+1}/{n_episodes}, avg reward: {avg_reward:.2f}")
            if len(avg_rewards) == 100 and np.mean(avg_rewards) >= -110:
                print(f"solved in {episode+1} episodes!")
                ep_rs[episode+1:] = -110  
                break
        avg_run_rw += ep_rs
        avg_run_len += ep_lens
    env.close()
    avg_rewards_final = avg_run_rw / n_runs
    avg_lengths_final = avg_run_len / n_runs
    std_rewards_final = np.nanstd(all_rewards, axis=0)
    return agent, avg_rewards_final, avg_lengths_final, std_rewards_final















###########################################################################################Tiling linear approximator

class Env_tiles:
    def __init__(self, numTilings=8, numTiles=10):
        self.numTilings = numTilings
        self.numTiles = numTiles
        position_min, position_max = -1.2, 0.6
        velocity_min, velocity_max = -0.07, 0.07
        self.low = np.array([position_min, velocity_min])
        self.high = np.array([position_max, velocity_max])
        self.offsets = [
            (high - low) / self.numTiles * np.linspace(0, 1, self.numTilings, endpoint=False)
            for low, high in zip(self.low, self.high)
        ]
        self.tilings = self.create_tilings()
    def boild_tiling_grid(self, low, high, bins, offsets):
        return [np.linspace(low[dim], high[dim], bins + 1)[1:-1] + offsets[dim] for dim in range(len(low))]
    def create_tilings(self):
        return [self.boild_tiling_grid(self.low, self.high, self.numTiles, [offset[i] for offset in self.offsets]) for i in range(self.numTilings)]
    def discretize(self, state):
        return [tuple(np.digitize(state[dim], bins) for dim, bins in enumerate(tiling)) for tiling in self.tilings]

class SARSAAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_decay=0.995, epsilon_min=0.0001, numTilings=8, numTiles=10):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.numTilings = numTilings
        self.numTiles = numTiles
        self.epsilon = epsilon_start
        self.tiling = Env_tiles(numTilings, numTiles)
        self.numActions = self.env.action_space.n
        self.numFeatures = self.numTilings * self.numTiles**2
        self.weights = np.zeros((self.numActions, self.numFeatures))
    def reset_weights(self):
        self.weights = np.zeros((self.numActions, self.numFeatures))
        self.epsilon = self.epsilon_start
    def featureVector(self, state, action):
        discretized_indices = self.tiling.discretize(state)
        featureVector = np.zeros(self.numFeatures)
        for i, ds in enumerate(discretized_indices):
            index = ds[0] * self.numTiles + ds[1]
            feature_index = i * self.numTiles**2 + index
            featureVector[feature_index] = 1
        return featureVector
    def get_Qvalues(self, state, action):
        return np.dot(self.weights[action], self.featureVector(state, action))
    def epsilon_greedy_action_selection(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        q_values = [self.get_Qvalues(state, a) for a in range(self.numActions)]
        return np.argmax(q_values)
    def update_Q(self, state, action, reward, next_state, next_action):
        phi = self.featureVector(state, action)
        q_sa = self.get_Qvalues(state, action)
        q_next = self.get_Qvalues(next_state, next_action)
        td_target = reward + self.gamma * q_next
        td_error = td_target - q_sa
        self.weights[action] += (self.alpha / self.numTilings) * td_error * phi

def train_sarsa_t(agent:SARSAAgent, env, num_episodes, num_runs=5):
    all_rewards = np.zeros((num_runs, num_episodes))
    all_steps = np.zeros((num_runs, num_episodes))
    for run in range(num_runs):
        # print(f"Run:{run+1}")
        agent.reset_weights()
        np.random.seed(run)
        for i in tqdm(range(num_episodes), desc=f"Run {run+1}/{num_runs}"):
            totalReward, steps = 0, 0
            state, _ = env.reset()
            action = agent.epsilon_greedy_action_selection(state)
            done = False
            while not done:
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_action = agent.epsilon_greedy_action_selection(next_state)
                agent.update_Q(state, action, reward, next_state, next_action)
                state, action = next_state, next_action
                totalReward += reward
                steps += 1
                done = terminated or truncated
            all_rewards[run, i] = totalReward
            all_steps[run, i] = steps
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        print(f"Final average reward: {np.mean(all_rewards[run])}")
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    mean_steps = np.mean(all_steps, axis=0)
    std_steps = np.std(all_steps, axis=0)
    return mean_rewards, std_rewards, mean_steps, std_steps

def train_sarsa_tiling(env_name, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_decay=0.995, epsilon_min=0.0001, numTilings=8, numTiles=10,num_episodes=3500,num_runs=5):
    env = gym.make(env_name)
    agent = SARSAAgent(env, alpha, gamma, epsilon_start, epsilon_decay, epsilon_min, numTilings, numTiles)
    sarsa_mean_rewards, sarsa_std_rewards, sarsa_mean_steps, sarsa_std_steps = train_sarsa_t(agent, env, num_episodes, num_runs=num_runs)
    print("Training complete. Mean rewards and standard deviations computed.")
    return sarsa_mean_rewards, sarsa_std_rewards, sarsa_mean_steps, sarsa_std_steps


class Tiling_Q:
    def __init__(self, env, numTilings=8, numTiles=10):
        self.numTilings = numTilings
        self.numTiles = numTiles
        position_min, position_max = -1.2, 0.6
        velocity_min, velocity_max = -0.07, 0.07
        self.low = np.array([position_min, velocity_min])
        self.high = np.array([position_max, velocity_max])
        self.offsets = [
            (high - low) / self.numTiles * np.linspace(0, 1, self.numTilings, endpoint=False)
            for low, high in zip(self.low, self.high)
        ]
        self.tilings = self.create_tilings()
    def create_tiling_grid(self, low, high, bins, offsets):
        return [np.linspace(low[dim], high[dim], bins + 1)[1:-1] + offsets[dim] for dim in range(len(low))]
    def create_tilings(self):
        return [self.create_tiling_grid(self.low, self.high, self.numTiles, [offset[i] for offset in self.offsets]) for i in range(self.numTilings)]
    def discretize(self, state):
        return [tuple(np.digitize(state[dim], bins) for dim, bins in enumerate(tiling)) for tiling in self.tilings]
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, tau_start=1.0, tau_decay=0.995, tau_min=0.05, numTilings=8, numTiles=10):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.tau_start = tau_start
        self.tau_decay = tau_decay
        self.tau_min = tau_min
        self.tau = tau_start 
        self.numTilings = numTilings
        self.numTiles = numTiles
        self.tiling = Tiling_Q(env, numTilings, numTiles)
        self.numActions = self.env.action_space.n
        self.numFeatures = self.numTilings * self.numTiles**2
        self.weights = np.zeros((self.numActions, self.numFeatures))
    def get_feature_vector(self, state, action):
        discretized_indices = self.tiling.discretize(state)
        featureVector = np.zeros(self.numFeatures)
        for i, ds in enumerate(discretized_indices):
            index = ds[0] * self.numTiles + ds[1]
            feature_index = i * self.numTiles**2 + index
            featureVector[feature_index] = 1
        return featureVector
    def reset_weights(self):
        self.weights = np.zeros((self.numActions, self.numFeatures))
        self.tau = self.tau_start  
    def get_Qvalue(self, state, action):
        return np.dot(self.weights[action], self.get_feature_vector(state, action))
    def softmax_action_selection(self, state):
        q_values = np.array([self.get_Qvalue(state, a) for a in range(self.numActions)])
        exp_values = np.exp((q_values - np.max(q_values)) / self.tau)
        probs = exp_values / np.sum(exp_values)
        return np.random.choice(self.numActions, p=probs)
    def update_Q(self, state, action, reward, next_state):
        phi = self.get_feature_vector(state, action)
        q_sa = self.get_Qvalue(state, action)
        q_next_max = max([self.get_Qvalue(next_state, a) for a in range(self.numActions)])
        td_target = reward + self.gamma * q_next_max
        td_error = td_target - q_sa
        self.weights[action] += (self.alpha / self.numTilings) * td_error * phi

def train_Q_t(agent:QLearningAgent, env, num_episodes, num_runs=5):
    episode_rewards = np.zeros((num_runs, num_episodes))
    steps_to_completion = np.zeros((num_runs, num_episodes))
    for run in range(num_runs):
        print(f"Run: {run+1}")
        agent.reset_weights()
        np.random.seed(run)
        for i in tqdm(range(num_episodes), desc=f"Run {run+1}/{num_runs}"):
            totalReward, steps = 0, 0
            state, _ = env.reset()
            done = False
            while not done:
                action = agent.softmax_action_selection(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.update_Q(state, action, reward, next_state)
                state = next_state
                totalReward += reward
                steps += 1
            episode_rewards[run, i] = totalReward
            steps_to_completion[run, i] = steps
            agent.tau = max(agent.tau_min, agent.tau * agent.tau_decay)
        print(f"Final average reward: {np.mean(episode_rewards[run])}")
    mean_rewards = np.mean(episode_rewards, axis=0)
    std_rewards = np.std(episode_rewards, axis=0)
    mean_steps = np.mean(steps_to_completion, axis=0)
    std_steps = np.std(steps_to_completion, axis=0)

    return mean_rewards, std_rewards, mean_steps, std_steps

def tain_q_learning_tiling(env_name, alpha=0.1, gamma=0.99, tau_start=1.0, tau_decay=0.9, tau_min=0.005, numTilings=8, numTiles=10, num_episodes=3500, num_runs=5):
    env = gym.make(env_name)
    agent = QLearningAgent(env, alpha, gamma, tau_start, tau_decay, tau_min, numTilings, numTiles)
    Q_mean_rewards, Q_std_rewards, Q_mean_steps, Q_std_steps = train_Q_t(agent, env, num_episodes, num_runs)

    print("Training complete. Mean rewards and standard deviations computed.")
    return Q_mean_rewards, Q_std_rewards, Q_mean_steps, Q_std_steps








############################################################################Main
def main():
    parser = argparse.ArgumentParser(description="mountain car v0 with discrete action support softmax with q learning and epsilon greedy with sarsa")
    parser.add_argument("--algo", type=str, choices=["q_learning", "sarsa", "both"], required=True, help="Algorithm to use: q_learning, sarsa, or both")
    parser.add_argument("--type", type=str, choices=["q_table","tiling"], required=True, help="Tiling or Q table")
    parser.add_argument("--env_name", type=str, default="MountainCar-v0", help="currently MountainCar-v0 dont know if other will work ")
    parser.add_argument("--n_episodes", type=int, default=10000, help="number of training episodes")
    parser.add_argument("--pos_bins", type=int, default=40, help="position split number")
    parser.add_argument("--vel_bins", type=int, default=40, help="velocity split number")
    parser.add_argument("--n_runs", type=int, default=5, help="number of runs to averae over")
    parser.add_argument("--alpha", type=float, default=0.1, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount parameter (gamma)")
    parser.add_argument("--initial_tau", type=float, default=1.0, help="initial temp softmax")
    parser.add_argument("--tau_decay", type=float, default=0.9, help="temp decay softmax")
    parser.add_argument("--min_tau", type=float, default=0.005, help="min temp softmax")
    parser.add_argument("--initial_epsilon", type=float, default=1.0, help="initial epsilon epsilon greedy")
    parser.add_argument("--epsilon_decay", type=float, default=0.9, help="epsilon decay epsilon greedy")
    parser.add_argument("--min_epsilon", type=float, default=0.005, help="min epsilon epilon greedy")
    parser.add_argument("--numTilings", type=int, default=15, help="min epsilon epilon greedy")
    parser.add_argument("--numTiles", type=int, default=10, help="min epsilon epilon greedy")
    
    np.random.seed(42)  
    args = parser.parse_args()
    n_bins = (args.pos_bins, args.vel_bins)
    q_rewards = None
    sarsa_rewards = None
    if args.type == "q_table":
        if args.algo in ["q_learning", "both"]:
            print("\nRunning Q-Learning...")
            q_agent, q_rewards, q_lengths, q_std_rewards = train_q_learning(args.env_name, args.n_episodes, n_bins,args.alpha, args.gamma,args.initial_tau, args.tau_decay, args.min_tau,args.n_runs)
            print("Q-Learning Completed!\n")
            plot_mean_std(q_rewards, q_std_rewards)
            plot_running_average(q_rewards)
            plot_cumulative_regret(-110, q_rewards)
        if args.algo in ["sarsa", "both"]:
            print("\nRunning SARSA...")
            sarsa_agent, sarsa_rewards, sarsa_lengths, sarsa_rewards_std = train_sarsa(args.env_name, args.n_episodes, n_bins,args.alpha, args.gamma,args.initial_epsilon, args.epsilon_decay, args.min_epsilon,args.n_runs)
            print("SARSA Completed!\n")
            plot_mean_std(sarsa_rewards, sarsa_rewards_std)
            plot_running_average(sarsa_rewards)
            plot_cumulative_regret(-110, sarsa_rewards)
        if args.algo == "both" and q_rewards is not None and sarsa_rewards is not None:
            plot_running_average_both(q_rewards, sarsa_rewards, window_size=100)
            plot_cumulative_regret_both(-110, q_rewards, sarsa_rewards)
    else:
        if args.algo in ["q_learning", "both"]:
            print("\nRunning Q-Learning...")
            q_rewards, q_std_rewards, Q_mean_steps, Q_std_steps=tain_q_learning_tiling(env_name= args.env_name, num_episodes= args.n_episodes,numTilings=args.numTilings, numTiles=args.numTiles,alpha=args.alpha, gamma=args.gamma,tau_start=args.initial_tau,tau_decay= args.tau_decay,tau_min= args.min_tau,num_runs= args.n_runs)
            print("Q-Learning Completed!\n")
            plot_mean_std(q_rewards, q_std_rewards)
            plot_running_average(q_rewards)
            plot_cumulative_regret(-110, q_rewards)
        if args.algo in ["sarsa", "both"]:
            print("\nRunning SARSA...")
            sarsa_mean_rewards, sarsa_std_rewards, sarsa_mean_steps, sarsa_std_steps = train_sarsa_tiling(env_name= args.env_name, num_episodes= args.n_episodes,numTilings=args.numTilings, numTiles=args.numTiles,alpha=args.alpha, gamma=args.gamma,epsilon_start= args.initial_epsilon,epsilon_decay= args.epsilon_decay,epsilon_min= args.min_epsilon,num_runs= args.n_runs)
            print("SARSA Completed!\n")
            plot_mean_std(sarsa_mean_rewards, sarsa_std_rewards)
            plot_running_average(sarsa_mean_rewards)
            plot_cumulative_regret(-110, sarsa_mean_rewards)
        if args.algo == "both" and q_rewards is not None and sarsa_mean_rewards is not None:
            plot_running_average_both(q_rewards, sarsa_mean_rewards, window_size=100)
            plot_cumulative_regret_both(-110, q_rewards, sarsa_mean_rewards)


if __name__ == "__main__":
    main()
