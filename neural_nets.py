import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from systems import InvertedPendulum, CartPole


import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List


class Pruner:
    """
    Add pruning capabilities to PolicyNetwork.
    Includes magnitude pruning and activation pruning.
    """

    def magnitude_pruning(self, sparsity: float = 0.5, verbose: bool = True) -> Dict:
        """
        Magnitude-based pruning: removes weights with smallest absolute values.

        Parameters:
        -----------
        sparsity : float
            Target sparsity (0.5 = remove 50% of weights)
        verbose : bool
            Print pruning statistics

        Returns:
        --------
        stats : dict
            Pruning statistics (layers pruned, weights removed, etc.)
        """
        stats = {"total_weights": 0, "pruned_weights": 0, "layer_stats": []}

        for name, module in self.network.named_modules():
            if isinstance(module, nn.Linear):
                weights = module.weight.data.abs()
                threshold = torch.quantile(weights, sparsity)
                mask = weights > threshold
                module.weight.data = module.weight.data * mask.float()

                # Track statistics
                total = weights.numel()
                pruned = (~mask).sum().item()

                stats["total_weights"] += total
                stats["pruned_weights"] += pruned
                stats["layer_stats"].append(
                    {
                        "layer": name,
                        "total": total,
                        "pruned": pruned,
                        "sparsity": pruned / total,
                    }
                )

        if verbose:
            print("MAGNITUDE PRUNING RESULTS")
            for layer_stat in stats["layer_stats"]:
                print(f"{layer_stat['layer']}")
                print(
                    f"  Pruned: {layer_stat['pruned']}/{layer_stat['total']} ({layer_stat['sparsity']*100:.1f}%)"
                )
            print(
                f"\nTotal pruned: {stats['pruned_weights']}/{stats['total_weights']} ({100*stats['pruned_weights']/stats['total_weights']:.1f}%)"
            )

        return stats

    def activation_pruning(
        self,
        dataloader: torch.utils.data.DataLoader,
        sparsity: float = 0.5,
        verbose: bool = True,
    ) -> Dict:
        """
        Activation-based pruning: removes neurons that are rarely activated.

        Parameters:
        -----------
        dataloader : DataLoader
            Training or validation dataloader to estimate activations
        sparsity : float
            Target sparsity (0.5 = remove 50% of neurons)
        verbose : bool
            Print pruning statistics

        Returns:
        --------
        stats : dict
            Pruning statistics
        """
        self.network.eval()
        stats = {"total_neurons": 0, "pruned_neurons": 0, "layer_stats": []}

        # Collect activation statistics
        activation_stats = {}

        def get_activation_stats(module, input, output):
            """Hook to collect activation statistics."""
            if isinstance(module, nn.ReLU):
                # Count how often ReLU is active (output > 0)
                activation_count = (output > 0).sum(dim=0)  # Sum across batch
                activation_stats[id(module)] = activation_count

        # Register hooks
        hooks = []
        for module in self.network.modules():
            if isinstance(module, nn.ReLU):
                hook = module.register_forward_hook(get_activation_stats)
                hooks.append(hook)

        # Forward pass through data to collect stats
        with torch.no_grad():
            total_samples = 0
            for batch_idx, (states_batch, _) in enumerate(dataloader):
                states_batch = states_batch.to(self.device)
                _ = self.network(states_batch)
                total_samples += states_batch.shape[0]

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Prune neurons based on activation frequency
        neuron_idx = 0
        for name, module in self.network.named_modules():
            if isinstance(module, nn.Linear):
                activation_freq = activation_stats.get(id(module), None)

                if activation_freq is not None:
                    activation_freq = activation_freq / total_samples
                    threshold = torch.quantile(activation_freq, sparsity)
                    inactive_mask = activation_freq < threshold

                    # Prune connections to inactive neurons
                    module.weight.data[inactive_mask, :] = 0
                    if module.bias is not None:
                        module.bias.data[inactive_mask] = 0

                    # Track statistics
                    total = module.weight.shape[0]
                    pruned = inactive_mask.sum().item()

                    stats["total_neurons"] += total
                    stats["pruned_neurons"] += pruned
                    stats["layer_stats"].append(
                        {
                            "layer": name,
                            "total": total,
                            "pruned": pruned,
                            "sparsity": pruned / total,
                            "activation_freq": activation_freq.cpu().numpy(),
                        }
                    )

        if verbose:
            print("ACTIVATION PRUNING RESULTS")
            for layer_stat in stats["layer_stats"]:
                print(f"{layer_stat['layer']}")
                print(
                    f"  Pruned: {layer_stat['pruned']}/{layer_stat['total']} ({layer_stat['sparsity']*100:.1f}%)"
                )
                print(
                    f"  Avg activation frequency: {layer_stat['activation_freq'].mean():.4f}"
                )
            if stats["total_neurons"] > 0:
                print(
                    f"\nTotal pruned: {stats['pruned_neurons']}/{stats['total_neurons']} ({100*stats['pruned_neurons']/stats['total_neurons']:.1f}%)"
                )

        return stats

    def get_sparsity(self) -> float:
        """
        Calculate current network sparsity (percentage of zero weights).

        Returns:
        --------
        sparsity : float
            Current sparsity (0.0 = dense, 1.0 = all zeros)
        """
        total_params = 0
        zero_params = 0

        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                total_params += module.weight.numel()
                zero_params += (module.weight == 0).sum().item()
                if module.bias is not None:
                    total_params += module.bias.numel()
                    zero_params += (module.bias == 0).sum().item()

        if total_params == 0:
            return 0.0

        return zero_params / total_params

    def print_network_stats(self):
        """Print detailed network statistics."""
        total_params = 0
        total_nonzero = 0

        print("NETWORK STATISTICS")

        for name, module in self.network.named_modules():
            if isinstance(module, nn.Linear):
                params = module.weight.numel()
                nonzero = (module.weight != 0).sum().item()

                total_params += params
                total_nonzero += nonzero

                sparsity = 1.0 - (nonzero / params)
                print(f"{name}")
                print(
                    f"  Params: {params:,} | Non-zero: {nonzero:,} | Sparsity: {sparsity*100:.1f}%"
                )

        total_sparsity = (
            1.0 - (total_nonzero / total_params) if total_params > 0 else 0.0
        )
        print(
            f"\nTotal: {total_params:,} params | {total_nonzero:,} non-zero | Sparsity: {total_sparsity*100:.1f}%"
        )


class PolicyNetwork(nn.Module, Pruner):
    """
    Neural network that learns to mimic LQR expert controller.

    Architecture:
        Input (state) -> Hidden layers -> Output (action)
    """

    def __init__(self, system, learning_rate=1e-3, device="cuda"):
        """
        Parameters:
        -----------
        state_dim : int
            Dimension of state space
        action_dim : int
            Dimension of action space
        """
        super().__init__()
        state_dim = 0
        action_dim = 0

        if isinstance(system, InvertedPendulum):
            state_dim = 2
            action_dim = 1
        elif isinstance(system, CartPole):
            state_dim = 4
            action_dim = 1

        self.network = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, action_dim),
        )

        self.learning_rate = learning_rate
        self.device = device
        self.to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None
        self.train_loader = None
        self.val_loader = None

    def __call__(self, state):
        """
        Compute action for a given state.

        Parameters:
        -----------
        state : np.array of shape (state_dim,)

        Returns:
        --------
        action : np.array of shape (action_dim,)
        """
        with torch.no_grad():
            # Convert state to tensor and normalize
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            state_normalized = (
                state_tensor - self.state_mean.to(self.device)
            ) / self.state_std.to(self.device)

            # Get normalized action tensor
            action_tensor_normalized = self.network(state_normalized)

            # Denormalize action tensor
            action = action_tensor_normalized * self.action_std.to(
                self.device
            ) + self.action_mean.to(self.device)
            action = action.cpu().numpy().squeeze()

        return np.array([action])

    def prepare_data(self, states, actions, train_split=0.8, batch_size=32):
        """
        Prepare data for training.

        Parameters:
        -----------
        states : np.array of shape (N, state_dim)
        actions : np.array of shape (N, action_dim)
        train_split : float
            Fraction of data for training
        batch_size : int
            Batch size for training

        Returns:
        --------
        train_loader : DataLoader
        val_loader : DataLoader
        """
        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(actions)

        # Train-val split
        n_samples = len(states)
        n_train = int(n_samples * train_split)
        indices = np.random.permutation(n_samples)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_states = states_tensor[train_indices]
        train_actions = actions_tensor[train_indices]
        val_states = states_tensor[val_indices]
        val_actions = actions_tensor[val_indices]

        # Normalize using train statistics
        self.state_mean = train_states.mean(dim=0)
        self.state_std = train_states.std(dim=0) + 1e-8
        self.action_mean = train_actions.mean(dim=0)
        self.action_std = train_actions.std(dim=0) + 1e-8

        train_states_norm = (train_states - self.state_mean) / self.state_std
        val_states_norm = (val_states - self.state_mean) / self.state_std

        train_actions_norm = (train_actions - self.action_mean) / self.action_std
        val_actions_norm = (val_actions - self.action_mean) / self.action_std

        # Create dataloaders
        train_dataset = TensorDataset(train_states_norm, train_actions_norm)
        val_dataset = TensorDataset(val_states_norm, val_actions_norm)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.train_loader = train_loader
        self.val_loader = val_loader

        return train_loader, val_loader

    def train_epoch(self, train_loader):
        """
        Train for one epoch.

        Parameters:
        -----------
        train_loader : DataLoader

        Returns:
        --------
        avg_loss : float
            Average training loss
        """
        self.network.train()
        total_loss = 0.0

        for states_batch, actions_batch in train_loader:
            states_batch = states_batch.to(self.device)
            actions_batch = actions_batch.to(self.device)

            # Forward pass
            predicted_actions = self.network(states_batch)
            loss = self.loss_fn(predicted_actions, actions_batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, val_loader):
        """
        Validate on validation set.

        Parameters:
        -----------
        val_loader : DataLoader

        Returns:
        --------
        avg_loss : float
            Average validation loss
        """
        self.network.eval()
        total_loss = 0.0

        with torch.no_grad():
            for states_batch, actions_batch in val_loader:
                states_batch = states_batch.to(self.device)
                actions_batch = actions_batch.to(self.device)

                predicted_actions = self.network(states_batch)
                loss = self.loss_fn(predicted_actions, actions_batch)

                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss

    def train(self, train_loader=None, val_loader=None, epochs=100, patience=20):
        """
        Train policy network with early stopping.

        Parameters:
        -----------
        train_loader : DataLoader
        val_loader : DataLoader
        epochs : int
            Maximum number of epochs
        patience : int
            Stop training if validation loss doesn't improve for this many epochs
        """
        if val_loader is None:
            val_loader = self.val_loader
        if train_loader is None:
            train_loader = self.train_loader

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

    def plot_losses(self):
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Train Loss", linewidth=2)
        plt.plot(self.val_losses, label="Validation Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def save_model(self, filepath):
        """Save trained policy network."""
        torch.save(
            {
                "model_state_dict": self.network.state_dict(),
                "state_mean": self.state_mean,
                "state_std": self.state_std,
                "action_mean": self.action_mean,
                "action_std": self.action_std,
            },
            filepath,
        )
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained policy network."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.state_mean = checkpoint["state_mean"]
        self.state_std = checkpoint["state_std"]
        self.action_mean = checkpoint["action_mean"]
        self.action_std = checkpoint["action_std"]
        print(f"Model loaded from {filepath}")
