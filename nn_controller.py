import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as td
from systems import InvertedPendulum, CartPole


class ActorNetwork(nn.Module):
    """
    Actor network for continuous control with tanh squashing.

    Uses Gaussian distribution but squashes output to [-1, 1] with tanh.
    Properly computes log probabilities accounting for squashing.
    """

    def __init__(self, state_dim, action_dim=1, hidden_dim=128):
        super(ActorNetwork, self).__init__()

        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Output mean of unsquashed policy (NOT tanh here)
        self.mean_head = nn.Linear(hidden_dim, action_dim)

        # Output log_std (learnable parameter, initialized small for stability)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, state):
        """Return mean and log_std of policy (unsquashed)"""
        features = self.feature_layers(state)
        mean = self.mean_head(features)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        return mean, std, log_std

    def get_action(self, state, deterministic=False):
        """Get action (deterministic or stochastic)"""
        mean, std, _ = self.forward(state)

        if deterministic:
            # Deterministic: just return tanh(mean)
            return torch.tanh(mean)
        else:
            # Stochastic: sample and squash
            dist = td.Normal(mean, std)
            unsquashed_action = dist.rsample()
            action = torch.tanh(unsquashed_action)

            # Compute log probability with squashing correction
            # log_prob(squashed) = log_prob(unsquashed) - sum(log(1 - tanh^2(a)))
            log_prob = dist.log_prob(unsquashed_action)
            log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1)

            return action, log_prob


class CriticNetwork(nn.Module):
    """
    Critic network for value function estimation.

    Input: state
    Output: estimated value of state
    """

    def __init__(self, state_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        """Estimate value of state"""
        return self.net(state).squeeze(-1)


class ActorCriticController:
    """
    Actor-Critic controller combining policy gradient and value function learning.

    Algorithm: Advantage Actor-Critic (A2C)
    - Actor: learns policy π(a|s)
    - Critic: learns value function V(s)
    - Advantage: A(s,a) = r + γV(s') - V(s)
    - Actor loss: -log(π(a|s)) * A(s,a)
    - Critic loss: MSE(V(s), target)
    """

    def __init__(
        self, system, actor_lr=3e-4, critic_lr=1e-3, gamma=0.99, gae_lambda=0.95
    ):
        """
        Initialize Actor-Critic controller.

        Parameters:
        -----------
        system : InvertedPendulum or CartPole
            The system to control
        actor_lr : float
            Learning rate for actor network
        critic_lr : float
            Learning rate for critic network
        gamma : float
            Discount factor for future rewards
        gae_lambda : float
            Lambda for Generalized Advantage Estimation (GAE)
            Higher lambda = lower variance but higher bias
        """
        self.system = system
        self.state_dim = 2 if isinstance(system, InvertedPendulum) else 4
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = torch.device("cpu")

        # Initialize networks
        self.actor = ActorNetwork(self.state_dim).to(self.device)
        self.critic = CriticNetwork(self.state_dim).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Training history
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []

    def compute_reward(self, state, action, next_state, done):
        """
        Compute reward signal based on safety and control objectives.

        Reward structure:
        - Safety: penalty for leaving safe region
        - Convergence: reward for moving toward origin
        - Control: small penalty for large actions
        - Survival: reward for each step survived
        """
        if not self.system.is_safe(next_state):
            return -100.0  # Terminal state penalty

        reward = 0.0

        # Convergence reward: encourage moving toward origin
        state_norm = np.linalg.norm(state)
        next_state_norm = np.linalg.norm(next_state)

        # Strong reward for getting closer
        if next_state_norm < state_norm:
            reward += 5.0
        else:
            reward -= 1

        # Control smoothness
        action_magnitude = np.abs(action[0])
        reward -= 0.1 * action_magnitude

        # Survival bonus per step
        reward += 2.0

        return reward

    def select_action(self, state, training=True):
        """
        Select action from policy.

        During training: sample stochastic action
        During evaluation: deterministic action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if training:
                action, log_prob = self.actor.get_action(
                    state_tensor, deterministic=False
                )
                action = action.cpu().numpy().flatten()
            else:
                action = self.actor.get_action(state_tensor, deterministic=True)
                action = action.cpu().numpy().flatten()

        return action

    def compute_gae(self, states, rewards, values, next_values, dones):
        """
        Compute Generalized Advantage Estimation (GAE).

        GAE reduces variance of advantage estimates while keeping bias low.
        A_t = r_t + γV(s_{t+1}) - V(s_t) + (γλ)A_{t+1}
        """
        batch_size = len(rewards)
        advantages = np.zeros(batch_size)

        gae = 0.0
        for t in reversed(range(batch_size)):
            if dones[t]:
                next_value = 0.0
            else:
                next_value = next_values[t]

            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages, returns

    def train_step(self, states, actions, rewards, next_states, dones):
        """
        Single training step for actor and critic.

        Collects trajectory, computes advantages, and updates both networks.
        """
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # Compute values
        with torch.no_grad():
            values = self.critic(states_tensor).cpu().numpy()
            next_values = self.critic(next_states_tensor).cpu().numpy()

        # Compute advantages and returns using GAE
        advantages, returns = self.compute_gae(
            states, rewards, values, next_values, dones
        )

        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Update Critic
        critic_values = self.critic(states_tensor).squeeze(-1)
        critic_loss = nn.MSELoss()(critic_values, returns_tensor)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Update Actor
        # Get mean and std from policy network
        mean, std, _ = self.actor.forward(states_tensor)

        # Sample from policy
        dist = td.Normal(mean, std)
        unsquashed = dist.rsample()
        sampled_actions = torch.tanh(unsquashed)

        # Compute log probabilities
        log_prob = dist.log_prob(unsquashed)
        log_prob = log_prob - torch.log(1.0 - sampled_actions.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1)

        # Policy gradient loss
        actor_loss = -(log_prob * advantages_tensor).mean()

        # Entropy bonus for exploration
        entropy = dist.entropy().sum(dim=-1).mean()
        entropy_coef = 0.01
        actor_loss = actor_loss - entropy_coef * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())

        return actor_loss.item(), critic_loss.item()

    def train_episode(self, max_steps=200):
        """
        Train for one episode.

        Collects full trajectory then updates networks once.
        """
        state = np.zeros(self.state_dim)

        # Random starting state within 80% of safe region
        safe_region = self.system.safe_region
        for j, (dim_name, (min_val, max_val)) in enumerate(safe_region.items()):
            state[j] = np.random.uniform(min_val * 0.8, max_val * 0.8)

        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_next_states = []
        episode_dones = []

        total_reward = 0.0

        for step in range(max_steps):
            # Select action
            action = self.select_action(state, training=True)

            # Step system
            next_state = self.system.step(state, action)

            # Compute reward
            done = not self.system.is_safe(next_state)
            reward = self.compute_reward(state, action, next_state, done)

            # Store transition
            episode_states.append(state.copy())
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_next_states.append(next_state.copy())
            episode_dones.append(float(done))

            total_reward += reward
            state = next_state

            if done:
                break

        # Train on collected trajectory
        if len(episode_states) > 0:
            states_array = np.array(episode_states)
            actions_array = np.array(episode_actions).reshape(-1, 1)
            rewards_array = np.array(episode_rewards)
            next_states_array = np.array(episode_next_states)
            dones_array = np.array(episode_dones)

            self.train_step(
                states_array,
                actions_array,
                rewards_array,
                next_states_array,
                dones_array,
            )

        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(len(episode_states))

        return total_reward, len(episode_states)

    def train(self, num_episodes=500, eval_interval=50):
        """
        Train the Actor-Critic controller for multiple episodes.

        Parameters:
        -----------
        num_episodes : int
            Number of training episodes
        eval_interval : int
            Evaluate every N episodes
        """
        print(f"Training Actor-Critic controller for {num_episodes} episodes...")

        for episode in range(num_episodes):
            reward, length = self.train_episode()

            if (episode + 1) % eval_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-eval_interval:])
                avg_length = np.mean(self.episode_lengths[-eval_interval:])
                avg_actor_loss = (
                    np.mean(self.actor_losses[-eval_interval:])
                    if self.actor_losses
                    else 0
                )
                avg_critic_loss = (
                    np.mean(self.critic_losses[-eval_interval:])
                    if self.critic_losses
                    else 0
                )

                print(
                    f"Episode {episode+1}: Avg Reward = {avg_reward:.2f}, "
                    f"Avg Length = {avg_length:.1f}, "
                    f"Actor Loss = {avg_actor_loss:.4f}, "
                    f"Critic Loss = {avg_critic_loss:.4f}"
                )

    def evaluate(self, num_trials=100, max_steps=200):
        """
        Evaluate trained controller on random trials (deterministic).

        Returns:
        --------
        success_rate : float
            Percentage of trials that stayed safe
        """
        successes = 0

        for trial in range(num_trials):
            state = np.zeros(self.state_dim)
            safe_region = self.system.safe_region

            for j, (dim_name, (min_val, max_val)) in enumerate(safe_region.items()):
                state[j] = np.random.uniform(min_val * 0.8, max_val * 0.8)

            success = True
            for step in range(max_steps):
                action = self.select_action(state, training=False)
                state = self.system.step(state, action)

                if not self.system.is_safe(state):
                    success = False
                    break

            if success:
                successes += 1

        success_rate = 100 * successes / num_trials
        return success_rate

    def test_single_trajectory(self, initial_state, n_steps=200, verbose=False):
        """
        Test on a single trajectory (deterministic).

        Returns:
        --------
        success : bool
        final_state : np.array
        trajectory : np.array
        """
        state = initial_state.copy()
        trajectory = [state.copy()]

        for step in range(n_steps):
            action = self.select_action(state, training=False)
            state = self.system.step(state, action)
            trajectory.append(state.copy())

            if not self.system.is_safe(state):
                break

        trajectory = np.array(trajectory)
        is_safe = np.all([self.system.is_safe(s) for s in trajectory])

        if verbose:
            print(f"Initial: {initial_state}")
            print(f"Final: {trajectory[-1]}")
            print(f"Safe: {is_safe}")

        return is_safe, trajectory[-1], trajectory
