import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from systems import InvertedPendulum, CartPole
import time

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm


class verifier:
    def __init__(self, system, network):
        self.system = system
        self.network = network
        self.device = network.device
        self.input_range = self._get_input_range()

    def _get_input_range(self):
        safe_region = self.system.safe_region
        if isinstance(self.system, InvertedPendulum):
            input_range = np.array([safe_region["theta"], safe_region["theta_dot"]])
        elif isinstance(self.system, CartPole):
            input_range = np.array(
                [
                    safe_region["x"],
                    safe_region["x_dot"],
                    safe_region["theta"],
                    safe_region["theta_dot"],
                ]
            )

        return input_range

    def output_range_sampling(self, num_samples) -> np.ndarray:
        model = self.network.network
        input_range = self.input_range

        model.eval()
        n_inputs = input_range.shape[0]

        # Sampling
        input_samples = np.random.uniform(
            low=input_range[:, 0],
            high=input_range[:, 1],
            size=(int(num_samples), n_inputs),
        )

        # Forward Pass
        input_tensor = torch.Tensor(input_samples).to(self.device)
        with torch.no_grad():
            output_tensor = model(input_tensor)
        output_samples = output_tensor.cpu().numpy()

        # Compute min and max for each output dimension
        output_min = np.min(output_samples, axis=0)
        output_max = np.max(output_samples, axis=0)
        output_range = np.column_stack([output_min, output_max])

        return output_range, output_samples

    def plot_output_range_sampling(
        self,
        output_range,
        nominal_output,
        output_samples=None,
        x_lim=None,
        y_lim=None,
    ):
        fig, ax = plt.subplots(figsize=(12, 10))

        ax.scatter(output_samples[:, 0], output_samples[:, 1], alpha=0.3, s=1, c="blue")

        # Plot the rectangular bound
        rect_x = [
            output_range[0, 0],
            output_range[0, 1],
            output_range[0, 1],
            output_range[0, 0],
            output_range[0, 0],
        ]
        rect_y = [
            output_range[1, 0],
            output_range[1, 0],
            output_range[1, 1],
            output_range[1, 1],
            output_range[1, 0],
        ]
        ax.plot(rect_x, rect_y, "r-", linewidth=2, label="Rectangular bound")

        # Plot nominal output
        ax.plot(
            nominal_output[0, 0].item(),
            nominal_output[0, 1].item(),
            "go",
            markersize=10,
            label="Nominal output",
            markeredgecolor="black",
            markeredgewidth=1.5,
        )

        ax.set_xlabel("Output Dimension 1")
        ax.set_ylabel("Output Dimension 2")
        if x_lim:
            ax.set_xlim(x_lim[0], x_lim[1])
        if y_lim:
            ax.set_ylim(y_lim[0], y_lim[1])
        ax.legend(loc="upper left")
        plt.show()

    def interval_bound_propagation(self) -> np.ndarray:
        """
        Compute certified bounds using interval arithmetic.

        Returns:
        --------
        output_range : np.ndarray of shape (n_outputs, 2)
            Certified bounds [[lower1, upper1], [lower2, upper2], ...]
        """
        start_time = time.time()
        input_lowers = self.input_range[:, 0]
        input_uppers = self.input_range[:, 1]

        # Normalize inputs
        state_mean = self.network.state_mean.cpu().numpy()
        state_std = self.network.state_std.cpu().numpy()

        lower = (input_lowers - state_mean) / state_std
        upper = (input_uppers - state_mean) / state_std

        # Extract weights from network
        model = self.network.network
        layers_info = []

        for layer in model:
            if isinstance(layer, nn.Linear):
                w = layer.weight.data.cpu().numpy()
                b = layer.bias.data.cpu().numpy()
                layers_info.append(("linear", w, b))
            elif isinstance(layer, nn.ReLU):
                layers_info.append(("relu", None, None))

        # Propagate bounds through network
        for layer_type, w, b in layers_info:
            if layer_type == "linear":
                # For linear layer: y = Wx + b
                lower_new = np.zeros(w.shape[0])
                upper_new = np.zeros(w.shape[0])

                for i in range(w.shape[0]):
                    # Positive weights use upper bound, negative use lower bound
                    max_val = (
                        np.sum(np.where(w[i] > 0, w[i] * upper, w[i] * lower)) + b[i]
                    )
                    min_val = (
                        np.sum(np.where(w[i] > 0, w[i] * lower, w[i] * upper)) + b[i]
                    )
                    lower_new[i] = min_val
                    upper_new[i] = max_val

                lower = lower_new
                upper = upper_new

            elif layer_type == "relu":
                # ReLU: max(0, x)
                lower = np.maximum(lower, 0)
                upper = np.maximum(upper, 0)

        output_range = np.column_stack([lower, upper])
        elapsed = time.time() - start_time

        return output_range, elapsed

    def auto_lirpa(self, method: str = "crown") -> Tuple[np.ndarray, float]:
        """
        Verification using auto-LiRPA library.

        Parameters:
        -----------
        method : str
            Verification method: "crown", "ibp", or "alpha-crown"
            - "crown": Fast CROWN propagation
            - "ibp": Interval Bound Propagation (fastest, loosest)
            - "alpha-crown": Optimized CROWN (slower, tighter bounds)

        Returns:
        --------
        output_range : np.ndarray of shape (n_outputs, 2)
            Certified bounds [[lower1, upper1], ...]
        elapsed : float
            Time taken
        """
        start_time = time.time()

        # Get input bounds
        input_lowers = self.input_range[:, 0]
        input_uppers = self.input_range[:, 1]
        n_inputs = len(input_lowers)

        # Convert to tensors and reshape for batch
        lower_bound = (
            torch.FloatTensor(input_lowers).unsqueeze(0).to(self.device)
        )  # (1, n_inputs)
        upper_bound = (
            torch.FloatTensor(input_uppers).unsqueeze(0).to(self.device)
        )  # (1, n_inputs)

        # Create dummy input for model initialization
        dummy_input = torch.zeros(1, n_inputs).to(self.device)

        # Create bounded module
        bounded_model = BoundedModule(self.network.network, dummy_input)

        # Create perturbation with input bounds
        ptb = PerturbationLpNorm(
            norm=np.inf,
            eps=None,
            x_L=lower_bound,  # Shape: (1, n_inputs)
            x_U=upper_bound,  # Shape: (1, n_inputs)
        )

        # Create bounded input tensor with perturbation
        bounded_input = BoundedTensor(dummy_input, ptb)

        # Compute bounds
        if method == "ibp":
            lower, upper = bounded_model.compute_bounds(
                x=(bounded_input,),
                IBP=True,
                method=None,
            )
        elif method == "crown":
            lower, upper = bounded_model.compute_bounds(
                x=(bounded_input,),
                IBP=False,
                method="backward",
            )
        elif method == "alpha-crown":
            lower, upper = bounded_model.compute_bounds(
                x=(bounded_input,),
                IBP=False,
                method="alpha-crown",
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Extract numpy arrays
        lower_np = lower.squeeze().cpu().detach().numpy()
        upper_np = upper.squeeze().cpu().detach().numpy()

        # Handle scalar outputs
        if lower_np.ndim == 0:
            lower_np = np.array([lower_np])
        if upper_np.ndim == 0:
            upper_np = np.array([upper_np])

        output_range = np.column_stack([lower_np, upper_np])
        elapsed = time.time() - start_time

        return output_range, elapsed
