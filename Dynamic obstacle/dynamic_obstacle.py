import argparse
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import minigrid

def plot_results(rewards, std_rewards, steps, std_steps, successes, std_successes, algorithm):
    plt.figure()
    plt.plot(rewards, alpha=0.6, label='Episode Reward')
    plt.fill_between(range(len(rewards)), rewards - std_rewards, rewards + std_rewards, alpha=0.3, color='b', label='±1 Std Dev')
    plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'), label='100-Episode Average', color='r')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Reward per Episode ({algorithm})')
    plt.legend()
    plt.grid()
    plt.savefig(f"rewards_{algorithm.lower()}.png")
    plt.show()
    plt.figure()
    plt.plot(steps, alpha=0.6, label='Episode Steps')
    plt.fill_between(range(len(steps)), steps - std_steps, steps + std_steps, alpha=0.3, color='b', label='±1 Std Dev')
    plt.plot(np.convolve(steps, np.ones(100)/100, mode='valid'), label='100-Episode Average', color='r')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title(f'Steps per Episode ({algorithm})')
    plt.legend()
    plt.grid()
    plt.savefig(f"steps_{algorithm.lower()}.png")
    plt.show()
    window_size = 100
    success_rates = np.convolve(successes, np.ones(window_size)/window_size, mode='valid') * 100
    std_successes_adj = std_successes[:len(success_rates)] * 100
    plt.figure()
    plt.plot(success_rates, label='Success Rate (%)')
    plt.fill_between(range(len(success_rates)),
                     np.maximum(0, success_rates - std_successes_adj),
                     np.minimum(100, success_rates + std_successes_adj),
                     alpha=0.3, color='b', label='±1 Std Dev')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.title(f'Success Rate (over {window_size} episodes) ({algorithm})')
    plt.legend()
    plt.grid()
    plt.savefig(f"success_rate_{algorithm.lower()}.png")
    plt.show()

class SARSAAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = {}
        self.num_actions = env.action_space.n

    def get_state_key(self, obs):
        env_grid = self.env.unwrapped.grid.encode()
        agent_pos = self.env.unwrapped.agent_pos
        agent_dir = self.env.unwrapped.agent_dir

        x, y = agent_pos
        surrounding = []
        for i in range(max(0, x-1), min(x+2, self.env.unwrapped.width)):
            for j in range(max(0, y-1), min(y+2, self.env.unwrapped.height)):
                cell_type = env_grid[j, i, 0]
                surrounding.append(int(cell_type))

        state_tuple = (x, y, agent_dir) + tuple(surrounding)
        return hash(state_tuple)

    def get_q_values(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.num_actions)
        return self.q_table[state_key]

    def choose_action(self, state_key):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = self.get_q_values(state_key)
            max_q = np.max(q_values)
            actions = np.where(q_values == max_q)[0]
            return np.random.choice(actions)

    def update_q_table(self, state_key, action, reward, next_state_key, next_action, done):
        q_values = self.get_q_values(state_key)
        next_q_values = self.get_q_values(next_state_key)

        target = reward + (self.gamma * next_q_values[next_action] if not done else 0)
        q_values[action] += self.alpha * (target - q_values[action])

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
def train_sarsa(agent_class, env,alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, episodes=3000, num_runs=5):
    all_rewards = np.zeros((num_runs, episodes))
    all_steps = np.zeros((num_runs, episodes))
    all_success_rates = np.zeros((num_runs, episodes))

    for run in range(num_runs):
        print(f"Starting run {run+1}/{num_runs}...")
        agent = agent_class(env,alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay)
        np.random.seed(run)
        
        rewards_per_episode = np.zeros(episodes)
        steps_per_episode = np.zeros(episodes)
        success_per_episode = np.zeros(episodes)

        for episode in range(episodes):
            obs, _ = env.reset()
            state_key = agent.get_state_key(obs)
            action = agent.choose_action(state_key)

            done = False
            total_reward = 0
            steps = 0
            success = False

            while not done:
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state_key = agent.get_state_key(next_obs)
                next_action = agent.choose_action(next_state_key)
                done = terminated or truncated

                agent.update_q_table(state_key, action, reward, next_state_key, next_action, done)
                
                state_key, action = next_state_key, next_action
                total_reward += reward
                steps += 1

                if reward > 0: 
                    success = True

            rewards_per_episode[episode] = total_reward
            steps_per_episode[episode] = steps
            success_per_episode[episode] = 1 if success else 0
            agent.decay_epsilon()
            if (episode + 1) % 100 == 0:
                avg_recent_success = np.mean(success_per_episode[max(0, episode - 99):episode + 1]) * 100
                avg_reward = np.mean(rewards_per_episode[max(0, episode-99):episode+1])
                avg_steps = np.mean(steps_per_episode[max(0, episode-99):episode+1])
                print(f"Run {run+1}, Episode {episode+1}: Avg Reward = {avg_reward:.2f}, Avg Steps = {avg_steps:.2f}, Success Rate = {avg_recent_success:.1f}%, epsilon = {agent.epsilon:.3f}")

        all_rewards[run] = rewards_per_episode
        all_steps[run] = steps_per_episode
        all_success_rates[run] = success_per_episode

    avg_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    avg_steps = np.mean(all_steps, axis=0)
    std_steps = np.std(all_steps, axis=0)
    avg_success_rate = np.mean(all_success_rates, axis=0)
    std_success_rate = np.std(all_success_rates, axis=0)
    
    return avg_rewards, std_rewards, avg_steps, std_steps, avg_success_rate, std_success_rate

class SoftmaxQLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, tau=1.0, tau_min=0.1, tau_decay=0.995):
        self.env = env
        self.alpha = alpha 
        self.gamma = gamma  
        self.tau = tau      
        self.tau_min = tau_min  
        self.tau_decay = tau_decay  
        self.q_table = {}
        self.num_actions = env.action_space.n

    def get_state_key(self, obs):
        env_grid = self.env.unwrapped.grid.encode()
        agent_pos = self.env.unwrapped.agent_pos
        agent_dir = self.env.unwrapped.agent_dir
        x, y = agent_pos
        surrounding = []
        for i in range(max(0, x-1), min(x+2, self.env.unwrapped.width)):
            for j in range(max(0, y-1), min(y+2, self.env.unwrapped.height)):
                cell_type = env_grid[j, i, 0]  
                surrounding.append(int(cell_type))

        state_tuple = (x, y, agent_dir) + tuple(surrounding)
        return hash(state_tuple)

    def get_q_values(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.num_actions)
        return self.q_table[state_key]

    def softmax(self, q_values):
        q_values = q_values - np.max(q_values)
        exp_values = np.exp(q_values / self.tau)
        probabilities = exp_values / np.sum(exp_values)
        return probabilities

    def choose_action(self, state_key):
        q_values = self.get_q_values(state_key)
        probabilities = self.softmax(q_values)
        action = np.random.choice(self.num_actions, p=probabilities)
        return action

    def update_q_table(self, state_key, action, reward, next_state_key, done):
        q_values = self.get_q_values(state_key)
        next_q_values = self.get_q_values(next_state_key)
        best_next_q = np.max(next_q_values) if not done else 0
        q_values[action] += self.alpha * (reward + self.gamma * best_next_q - q_values[action])

    def decay_temperature(self):
        if self.tau > self.tau_min:
            self.tau *= self.tau_decay


def train_q_learning(agent_class,  env,alpha=0.1, gamma=0.99, tau=1.0, tau_min=0.1, tau_decay=0.995, episodes=3000, num_runs=5):
    all_rewards = np.zeros((num_runs, episodes))
    all_steps = np.zeros((num_runs, episodes))
    all_successes = np.zeros((num_runs, episodes))

    for run in range(num_runs):
        print(f"Starting run {run+1}/{num_runs}...")
        agent = agent_class(env,alpha,gamma,tau,tau_min,tau_decay)
        np.random.seed(run)

        rewards_per_episode = np.zeros(episodes)
        steps_per_episode = np.zeros(episodes)
        success_rate = np.zeros(episodes)

        for episode in range(episodes):
            obs, _ = env.reset()
            state_key = agent.get_state_key(obs)

            done = False
            total_reward = 0
            steps = 0
            success = False

            while not done:
                action = agent.choose_action(state_key)
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_state_key = agent.get_state_key(next_obs)
                done = terminated or truncated
                agent.update_q_table(state_key, action, reward, next_state_key, done)
                state_key = next_state_key
                total_reward += reward
                steps += 1

                if reward > 0:
                    success = True

            rewards_per_episode[episode] = total_reward
            steps_per_episode[episode] = steps
            success_rate[episode] = 1 if success else 0
            agent.decay_temperature()

            if (episode + 1) % 100 == 0:
                recent_success_rate = np.mean(success_rate[max(0, episode-99):episode+1]) * 100
                avg_reward = np.mean(rewards_per_episode[max(0, episode-99):episode+1])
                avg_steps = np.mean(steps_per_episode[max(0, episode-99):episode+1])
                print(f"Run {run+1}, Episode {episode+1}: Avg Reward = {avg_reward:.2f}, Avg Steps = {avg_steps:.2f}, Success Rate = {recent_success_rate:.1f}%, Tau = {agent.tau:.3f}")

        all_rewards[run] = rewards_per_episode
        all_steps[run] = steps_per_episode
        all_successes[run] = success_rate

    avg_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    avg_steps = np.mean(all_steps, axis=0)
    std_steps = np.std(all_steps, axis=0)
    avg_success_rate = np.mean(all_successes, axis=0)
    std_success_rate = np.std(all_successes, axis=0)

    return avg_rewards, std_rewards, avg_steps, std_steps, avg_success_rate, std_success_rate




def main():
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent using SARSA or Q-learning.")
    parser.add_argument("--algorithm", choices=["sarsa", "q-learning"], required=True, help="Algorithm to use: SARSA or Q-learning")
    parser.add_argument("--env", type=str, default="MiniGrid-Dynamic-Obstacles-5x5-v0", help="Gym environment name")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate (only for SARSA)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon for SARSA")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum epsilon for SARSA")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Epsilon decay factor for SARSA")
    parser.add_argument("--tau", type=float, default=1.0, help="Initial epsilon for SARSA")
    parser.add_argument("--tau_min", type=float, default=0.05, help="Minimum epsilon for SARSA")
    parser.add_argument("--tau_decay", type=float, default=0.995, help="Epsilon decay factor for SARSA")
    parser.add_argument("--episodes", type=int, default=3000, help="Number of episodes")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of independent runs")
    
    args = parser.parse_args()
    
    env = gym.make(args.env)
    
    if args.algorithm == "sarsa":
        agent_class = SARSAAgent
        sarsa_avg_rewards, sarsa_std_rewards, sarsa_avg_steps, sarsa_std_steps, sarsa_avg_success_rate, sarsa_std_success_rate = train_sarsa(agent_class, env, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, epsilon_min=args.epsilon_min, epsilon_decay=args.epsilon_decay, episodes=args.episodes, num_runs=args.num_runs)
        plot_results(sarsa_avg_rewards, sarsa_std_rewards, sarsa_avg_steps, sarsa_std_steps, sarsa_avg_success_rate, sarsa_std_success_rate,algorithm="SARSA")
    else:  # Q-learning
        agent_class = SoftmaxQLearningAgent
        Q_avg_rewards, Q_std_rewards, Q_avg_steps, Q_std_steps, Q_avg_success_rate, Q_std_success_rate = train_q_learning(agent_class, env,alpha=args.alpha, gamma=args.gamma, tau=args.tau, tau_min=args.tau_min, tau_decay=args.tau_decay, episodes=args.episodes, num_runs=args.num_runs)

        plot_results(Q_avg_rewards, Q_std_rewards, Q_avg_steps, Q_std_steps, Q_avg_success_rate, Q_std_success_rate,algorithm="Q_learning")

    print("Training complete!")
    
if __name__ == "__main__":
    main()