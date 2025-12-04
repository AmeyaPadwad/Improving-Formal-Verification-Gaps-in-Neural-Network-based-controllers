import numpy as np
from systems import InvertedPendulum, CartPole
from scipy.linalg import solve_continuous_are


class LQRExpert:
    """
    LQR-based expert controller using feedback gains.

    The control law is:
        u = -K @ x

    where x is the state and K is the gain matrix.

    For Inverted Pendulum:
        u = -K_θ * θ - K_θ̇ * θ̇

    For CartPole:
        u = -K_x * x - K_ẋ * ẋ - K_θ * θ - K_θ̇ * θ̇
    """

    def __init__(self, system):
        """
        Initialize LQR controller for a system.

        Parameters:
        -----------
        system : InvertedPendulum or CartPole
            The system to control
        """
        self.system = system
        self.state_dim = 2 if isinstance(system, InvertedPendulum) else 4
        self.A, self.B = self._linearize_system()
        self.Q, self.R = self._define_cost_matrices()
        self.K = self._solve_lqr()

    def _linearize_system(self):
        """
        Compute linearized A and B matrices around equilibrium (x=0).

        For continuous-time systems: ẋ = A @ x + B @ u
        """
        if isinstance(self.system, InvertedPendulum):
            return self._linearize_pendulum()
        elif isinstance(self.system, CartPole):
            return self._linearize_cartpole()
        else:
            raise ValueError("Unknown system type")

    def _linearize_pendulum(self):
        """
        Linearize inverted pendulum around upright equilibrium.

        Continuous dynamics (from systems.py):
            θ̈ = (g/L)*sin(θ) + u/(m*L²) - (μ/(m*L²))*θ̇

        State: x = [θ, θ̇]
        Control: u (torque)

        Linearized around θ=0, θ̇=0:
            θ̈ ≈ (g/L)*θ + u/(m*L²) - (μ/(m*L²))*θ̇

        In state-space form ẋ = A @ x + B @ u:
            [θ̇  ]   [0    1  ] [θ ]   [0      ]
            [θ̈  ] = [g/L  -μ/mL²] [θ̇] + [1/mL²] @ u
        """
        g = self.system.g
        L = self.system.L
        m = self.system.m
        mu = self.system.mu

        A = np.array([[0, 1], [g / L, -mu / (m * L**2)]])

        B = np.array([[0], [1 / (m * L**2)]])

        return A, B

    def _linearize_cartpole(self):
        """
        Linearize CartPole around upright equilibrium.

        State: x = [x, ẋ, θ, θ̇]
        Control: F (force)

        Linearized dynamics around θ=0, ẋ=0, θ̇=0:
            [ẋ    ]   [0  1  0                    0] [x  ]   [0]
            [ẍ    ] = [0  0  -m_p*g/m_c           0] [ẋ  ] + [1/m_c] @ F
            [θ̇    ]   [0  0  0                    1] [θ  ]   [0]
            [θ̈    ]   [0  0  g*(m_c+m_p)/(m_c*L)  0] [θ̇ ]   [-1/(m_c*L)]
        """
        m_c = self.system.masscart
        m_p = self.system.masspole
        L = self.system.length
        g = self.system.gravity

        A = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, -m_p * g / m_c, 0],
                [0, 0, 0, 1],
                [0, 0, g * (m_c + m_p) / (m_c * L), 0],
            ]
        )

        B = np.array([[0], [1 / m_c], [0], [-1 / (m_c * L)]])

        return A, B

    def _define_cost_matrices(self):
        """
        Define Q and R cost matrices for the LQR problem.

        Q penalizes state deviation (larger Q → tighter control of state)
        R penalizes control effort (larger R → conservative control)

        For safety: increase penalties on angle/position to keep within bounds.
        """
        if isinstance(self.system, InvertedPendulum):
            # Pendulum: state = [θ, θ̇]
            # Penalize angle heavily (safety), moderate velocity
            Q = np.diag([500.0, 50.0])
            R = np.array([[0.1]])

        elif isinstance(self.system, CartPole):
            # CartPole: state = [x, ẋ, θ, θ̇]
            # Penalize angle most (primary safety concern)
            # Then position, then velocities
            Q = np.diag([10.0, 1.0, 100.0, 10.0])
            R = np.array([[1]])

        return Q, R

    def _solve_lqr(self):
        """
        Solve the continuous-time algebraic Riccati equation (ARE).

        The ARE is: A^T @ P + P @ A - P @ B @ R^(-1) @ B^T @ P + Q = 0

        scipy.linalg.solve_continuous_are computes P, then we compute K.
        Optimal gain: K = R^(-1) @ B^T @ P
        """
        P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        K = np.linalg.inv(self.R) @ self.B.T @ P
        return K

    def __call__(self, state):
        """
        Compute control action for a given state.

        Parameters:
        -----------
        state : numpy array
            Current state

        Returns:
        --------
        action : array
            Control action(s)
        """
        action = -self.K @ state
        theta, theta_dot = state

        # Approaching danger zone
        if abs(theta) > 0.3:
            action += -1.5 * np.sign(theta_dot) * abs(theta_dot) ** 1.5

        return action
