import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from systems import InvertedPendulum, CartPole


class PolicyNetwork(nn.Module):
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
        if isinstance(system, InvertedPendulum):
            state_dim = 2
            action_dim = 1
        elif isinstance(system, CartPole):
            state_dim = 4
            action_dim = 1

        layers = []
        layers.append(nn.Linear(state_dim, 64))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(64, 32))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(32, action_dim))

        self.network = nn.Sequential(*layers)
        self.learning_rate = learning_rate
        self.device = device
        self.to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        self.state_mean = None
        self.state_std = None
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
            # Convert to tensor and normalize
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            state_normalized = (
                state_tensor - self.state_mean.to(self.device)
            ) / self.state_std.to(self.device)

            # Get action
            action_tensor = self.network(state_normalized)
            action = action_tensor.cpu().numpy().squeeze()

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

        # Normalize states
        self.state_mean = states_tensor.mean(dim=0)
        self.state_std = states_tensor.std(dim=0) + 1e-8
        states_normalized = (states_tensor - self.state_mean) / self.state_std

        # Train-val split
        n_samples = len(states)
        n_train = int(n_samples * train_split)
        indices = np.random.permutation(n_samples)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        # Create datasets
        train_dataset = TensorDataset(
            states_normalized[train_indices], actions_tensor[train_indices]
        )
        val_dataset = TensorDataset(
            states_normalized[val_indices], actions_tensor[val_indices]
        )

        # Create dataloaders
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
        print(f"Model loaded from {filepath}")
