"""
Dynamical Systems Module

This module implements two classic control systems:
1. Inverted Pendulum: A stick balanced upright on a fixed pivot
2. CartPole: A cart moving on a track with a pole balanced on top

Each system uses Euler integration for numerical simulation of continuous dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================
# Universal constants
# ==========================
G = 9.81  # Gravity

# ==========================
# Inverted Pendulum parameters
# ==========================
l = 1  # Length of the pendulum pole (meters)
m = 1  # Mass at the end of the pendulum (kg)
mu = 0.3  # Friction coefficient (damping term)
max_torque = 4  # Maximum torque the motor can apply (N*m)

# ==========================
# Cartpole parameters
# ==========================
mass_cart = 1  # Mass of the cart (kg)
mass_pole = 0.1  # Mass of the pole (kg)
length = 0.5  # Length of the pole (meters)
force_mag = 10  # Maximum force the cart motor can apply (Newtons)


# ==========================
# Inverted Pendulum class
# ==========================
class InvertedPendulum:
    """
    Inverted pendulum on fixed pivot.

    Physics:
    --------
    The pendulum is governed by the rotational equation of motion:
        I * θ̈ = τ_applied - m*g*L*sin(θ) - μ*θ̇

    where:
        I = moment of inertia = m*L²
        θ = angle from vertical (radians; 0 = upright, π = hanging down)
        θ̇ = angular velocity (rad/s)
        θ̈ = angular acceleration (rad/s²)
        τ_applied = control torque from motor (input)
        m*g*L*sin(θ) = gravitational torque (destabilizing)
        μ*θ̇ = friction/damping torque (stabilizing)

    Simplifying by dividing by I = m*L²:
        θ̈ = (g/L)*sin(θ) + u/(m*L²) - (μ/(m*L²))*θ̇

    """

    def __init__(self, dt=0.02):
        """
        Parameters:
        -----------
        dt (float): Timestep for numerical integration (default 0.02 seconds)
        """
        self.g = G
        self.L = l
        self.m = m
        self.mu = mu
        self.dt = dt
        self.max_torque = max_torque

    def step(self, state, action):
        """
        Simulate one timestep of pendulum dynamics using Euler integration.

        This method numerically integrates the pendulum differential equation
        from the current state using a simple Euler method:
            x_{n+1} = x_n + dx/dt * dt

        Parameters:
        -----------
        state (numpy array): Current state [θ, θ']
        action (array): Control torque to apply

        Returns:
        --------
        next_state (numpy array): State at time t+dt, shape (2,) [θ_{next}, θ̇_{next}]
        """
        theta, theta_dot = state

        # Clip action to physical limits
        u = np.clip(action[0], -self.max_torque, self.max_torque)

        # Compute acceleration: theta_ddot = (g/L)*sin(theta) + u/(m*L^2) - mu*theta_dot
        theta_ddot = (
            (self.g / self.L) * np.sin(theta)
            + u / (self.m * self.L**2)
            - (self.mu / (self.m * self.L**2)) * theta_dot
        )

        # Euler step
        next_theta_dot = theta_dot + theta_ddot * self.dt
        next_theta = theta + theta_dot * self.dt

        return [next_theta, next_theta_dot]

    def is_safe(self, state):
        """
        Check if a state is within the safe operating region.

        Safety Bounds:
        ---------------
        θ ∈ [-0.5, 0.5] radians ≈ [-28.6°, 28.6°]
        θ̇ ∈ [-1.0, 1.0] radians/second ≈ [-57.3°/s, 57.3°/s]

        Parameters:
        -----------
        state (numpy array): Current state [θ, θ̇]

        Returns:
        --------
        is_safe (bool): True if state is within safe region, False otherwise
        """

        theta, theta_dot = state
        return abs(theta) <= 0.5 and abs(theta_dot) <= 1.0

    @property
    def safe_region(self):
        """
        Return the bounds of the safe operating region.

        Returns:
        --------
        safe_region (dict): Maps dimension names to [min, max] bounds
        """
        return {"theta": [-0.5, 0.5], "theta_dot": [-1.0, 1.0]}

    def plot_trajectory(self, trajectory=None):
        safe_region = self.safe_region
        theta_min = safe_region["theta"][0]
        theta_max = safe_region["theta"][1]
        theta_dot_min = safe_region["theta_dot"][0]
        theta_dot_max = safe_region["theta_dot"][1]

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

        # Draw safe region as a rectangle
        safe_region_rect = patches.Rectangle(
            (theta_min, theta_dot_min),
            theta_max - theta_min,
            theta_dot_max - theta_dot_min,
            linewidth=2,
            edgecolor="green",
            facecolor="lightgreen",
            alpha=0.3,
            label="Safe Region",
        )
        ax.add_patch(safe_region_rect)

        if trajectory is not None:
            ax.scatter(trajectory[0][0], trajectory[0][1], label="Trajectory Start")
            ax.scatter(
                [i[0] for i in trajectory[1:-1]],
                [i[1] for i in trajectory[1:-1]],
                label="Trajectory",
            )
            ax.scatter(trajectory[-1][0], trajectory[-1][1], label="Trajectory End")

        # Set labels and title
        ax.set_xlabel("θ (theta)", fontsize=12)
        ax.set_ylabel("θ̇ (theta_dot)", fontsize=12)
        ax.set_title("Trajectory plot", fontsize=14, fontweight="bold")

        ax.legend(fontsize=11)

        # Equal aspect ratio for better visualization
        ax.set_aspect("equal")

        plt.tight_layout()
        plt.show()

        return fig, ax


# ============================================================================
# CLASS: CARTPOLE
# ============================================================================


class CartPole:
    """
    CartPole Dynamical System.

    A cart moves left/right on a frictionless track. A pole is balanced on top
    of the cart (via a hinge). The goal is to apply horizontal force to the cart
    such that the pole stays upright.

    Physics:
    --------
    The system has two coupled differential equations:

    1. Cart motion: m_cart * ẍ = F - m_pole * L * θ̈ * cos(θ) + m_pole * L * θ̇² * sin(θ)
    2. Pole motion: θ̈ = (g*sin(θ) + cos(θ)*(...)) / (L*(4/3 - m_pole*cos²(θ)/m_total))
    """

    def __init__(self, dt=0.02):
        """
        Initialize the CartPole system.

        Parameters:
        -----------
        dt (float): Timestep for numerical integration (default 0.02 seconds)
        """
        self.gravity = G
        self.masscart = mass_cart
        self.masspole = mass_pole
        self.total_mass = self.masspole + self.masscart
        self.length = length
        self.polemass_length = self.masspole * self.length
        self.force_mag = force_mag
        self.tau = dt

    def step(self, state, action):
        """
        Simulate one timestep of CartPole dynamics using Euler integration.
        The implementation follows the standard CartPole dynamics from
        OpenAI Gym (which follows Barto, Sutton, & Anderson 1983).

        Dynamics Derivation:
        --------------------
        From Lagrangian mechanics:

        Variables:
            F = force applied to cart (N)
            m_c = mass of cart (kg)
            m_p = mass of pole (kg)
            M = m_c + m_p (total mass)
            x = cart position (m)
            θ = pole angle from vertical (rad)
            L = length of pole (m)

        Derived equations (from Lagrangian):
            temp = (F + m_p*L*θ̇²*sin(θ)) / M
            θ̈ = (g*sin(θ) - cos(θ)*temp) / (L*(4/3 - m_p*cos²(θ)/M))
            ẍ = temp - m_p*L*θ̈*cos(θ) / M

        These are nonlinear differential equations.

        Parameters:
        -----------
        state (numpy array): Current state [x, ẋ, θ, θ̇]
            x (float): cart position on track (m)
            ẋ (float): cart velocity (m/s)
            θ (float): pole angle from vertical (rad)
            θ̇ (float): pole angular velocity (rad/s)

        action (float or array): Normalized force command
            Can be scalar or 1-element array
            Actual force = action * force_mag, clipped to [-1, 1]

        Returns:
        --------
        next_state (numpy array): State at time t+dt, shape (4,)
            [x_{next}, ẋ_{next}, θ_{next}, θ̇_{next}]
        """
        x, x_dot, theta, theta_dot = state
        force = self.force_mag * np.clip(action[0], -1, 1)

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (
            force + self.polemass_length * theta_dot * theta_dot * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x_dot = x_dot + xacc * self.tau
        theta_dot = theta_dot + thetaacc * self.tau

        x = x + x_dot * self.tau
        theta = theta + theta_dot * self.tau

        return np.array([x, x_dot, theta, theta_dot])

    def is_safe(self, state):
        x, x_dot, theta, theta_dot = state
        return abs(x) <= 2.4 and abs(theta) <= 0.209

    @property
    def safe_region(self):
        """
        Check if a state is within the safe operating region.
        Safety Bounds:
        ---------------
        x ∈ [-2.4, 2.4] meters
        θ ∈ [-0.209, 0.209] radians ≈ [-12°, 12°]

        Parameters:
        -----------
        state (numpy array): Current state [x, ẋ, θ, θ']

        Returns:
        --------
        is_safe (bool): True if state is within safe region, False otherwise
        """
        return {
            "x": [-2.4, 2.4],
            "x_dot": [-5.0, 5.0],
            "theta": [-0.209, 0.209],
            "theta_dot": [-5.0, 5.0],
        }
