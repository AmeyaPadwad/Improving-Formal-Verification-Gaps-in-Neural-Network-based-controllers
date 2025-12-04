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

    def output_bounds_sampling(self, num_samples) -> np.ndarray:
        start_time = time.time()
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
        output_bounds = np.column_stack([output_min, output_max])
        elapsed = time.time() - start_time

        return output_bounds, output_samples, round(elapsed, 3)

    def _compute_lipschitz_constant(
        self, input_range, output_range, input_uppers, input_lowers
    ):
        # Compute Lipschitz constant
        input_range = input_range.squeeze().cpu().numpy()
        lipschitz_constant = np.max(output_range / (input_range + 1e-8))

        # Get nominal output
        with torch.no_grad():
            nominal_state = (input_lowers + input_uppers) / 2
            nominal_tensor = (
                torch.FloatTensor(nominal_state).unsqueeze(0).to(self.device)
            )
            nominal_output = (
                self.network.network(nominal_tensor).squeeze().cpu().detach().numpy()
            )

        # Lipschitz bound: output ∈ [nominal - L·ε, nominal + L·ε]
        max_input_perturbation = np.max(input_range) / 2
        lipschitz_margin = lipschitz_constant * max_input_perturbation

        lower_np = np.atleast_1d(nominal_output - lipschitz_margin)
        upper_np = np.atleast_1d(nominal_output + lipschitz_margin)

        return lower_np, upper_np

    def auto_lirpa(self, method: str = "crown") -> Tuple[np.ndarray, float]:
        """
        Verification using auto-LiRPA library.

        Parameters:
        -----------
        method : str
            Verification method: "crown", "ibp", or "alpha-crown"
            - "crown": Fast CROWN propagation
            - "ibp": Interval Bound Propagation
            - "alpha-crown": Optimized CROWN
            - "lipschitz": Lipschitz-based bounds

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
        if method == "ibp" or method == "lipschitz":
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

        if method == "lipschitz":
            input_range = upper_bound - lower_bound
            output_range = upper_np - lower_np
            lower_np, upper_np = self._compute_lipschitz_constant(
                input_range, output_range, input_uppers, input_lowers
            )

        output_bounds = np.column_stack([lower_np, upper_np])
        elapsed = time.time() - start_time

        return output_bounds, round(elapsed, 3)

    def compare(self, n_samples, nominal_state):
        output_bounds = {}

        nominal_output = self.network(nominal_state)

        output_bounds_sampling, output_samples, time_elapsed_sampling = (
            self.output_bounds_sampling(n_samples)
        )

        output_bounds_ibp, time_elapsed_ibp = self.auto_lirpa("ibp")
        output_bounds_crown, time_elapsed_crown = self.auto_lirpa("crown")
        output_bounds_alpha_crown, time_elapsed_alpha_crown = self.auto_lirpa(
            "alpha-crown"
        )
        output_bounds_lipschitz, time_elapsed_lipschitz = self.auto_lirpa("lipschitz")

        print(f"{nominal_output=}")
        print(f"{output_bounds_sampling=}, {time_elapsed_sampling=}s")
        print(f"{output_bounds_ibp=}, {time_elapsed_ibp=}s")
        print(f"{output_bounds_crown=}, {time_elapsed_crown=}s")
        print(f"{output_bounds_alpha_crown=}, {time_elapsed_alpha_crown=}s")
        print(f"{output_bounds_lipschitz=}, {time_elapsed_lipschitz=}s")

        output_bounds["nominal_output"] = nominal_output
        output_bounds["sampling"] = (
            output_bounds_sampling,
            output_samples,
            time_elapsed_sampling,
        )
        output_bounds["ibp"] = (
            output_bounds_ibp,
            time_elapsed_ibp,
        )
        output_bounds["crown"] = (
            output_bounds_crown,
            time_elapsed_crown,
        )
        output_bounds["alpha_crown"] = (
            output_bounds_alpha_crown,
            time_elapsed_alpha_crown,
        )
        output_bounds["lipschitz"] = (output_bounds_lipschitz, time_elapsed_lipschitz)

        return output_bounds
