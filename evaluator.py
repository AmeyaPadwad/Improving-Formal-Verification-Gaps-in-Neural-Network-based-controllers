import numpy as np
import pickle


def test_single_trajectory(
    controller, system, initial_state, n_steps=100, verbose=False
):
    """
    Test a single trajectory with detailed diagnostics.

    Parameters:
        controller
        system
        initial_state
        n_steps
        verbose

    Returns:
        success: bool - whether trajectory stayed safe
        final_state: numpy array - final state after simulation
        max_state: numpy array - maximum absolute state values
        trajectory: numpy array - full trajectory
    """
    state = initial_state.copy()
    trajectory = [state.copy()]

    for step in range(n_steps):
        action = controller(state)
        state = system.step(state, action)
        trajectory.append(state.copy())

    trajectory = np.array(trajectory)

    # Check if safe
    is_safe = np.all([system.is_safe(s) for s in trajectory])

    # Get statistics
    final_state = trajectory[-1]
    max_state = np.max(np.abs(trajectory), axis=0)

    if verbose:
        print(f"  Initial: {initial_state}")
        print(f"  Final:   {final_state}")
        print(f"  Safe:    {is_safe}")
        print(f"  Max abs: {max_state}")

    return is_safe, final_state, max_state, trajectory


def test_random_trajectories(
    controller, system, n_trials=50, n_steps=100, verbose=False, safe_region_limit=0.8
):
    """Test on random starting states within safe region."""

    successes = 0
    failures = 0

    safe_region = system.safe_region
    state_dim = len(safe_region)

    for trial in range(n_trials):
        # Random starting state (within safe region)
        state = np.zeros(state_dim)
        for j, (dim_name, (min_val, max_val)) in enumerate(safe_region.items()):
            # Start within 80% of safe region
            state[j] = np.random.uniform(
                min_val * safe_region_limit, max_val * safe_region_limit
            )

        trajectory = [state.copy()]
        is_safe = True

        for step in range(n_steps):
            action = controller(state)
            state = system.step(state, action)
            trajectory.append(state.copy())

            if not system.is_safe(state):
                is_safe = False
                break

        if is_safe:
            successes += 1
        else:
            failures += 1

    success_rate = 100 * successes / n_trials

    if verbose:
        print(f"  Trials: {n_trials}")
        print(f"  Successes: {successes}/{n_trials} ({success_rate:.1f}%)")
        print(f"  Failures: {failures}/{n_trials}")

    return success_rate


def generate_lqr_dataset(
    system, expert, n_trajectories=1000, max_steps=200, save_path=None
):
    """
    Generate dataset of (state, action) pairs from LQR expert trajectories.

    Parameters:
    -----------
    system : InvertedPendulum or CartPole
        The system to collect data from
    expert : LQRExpert
        The expert controller
    n_trajectories : int
        Number of trajectories to generate
    max_steps : int
        Maximum steps per trajectory
    save_path : str
        Path to save the dataset

    Returns:
    --------
    states : np.array of shape (N, state_dim)
    actions : np.array of shape (N, action_dim)
    """

    states_list = []
    actions_list = []

    safe_region = system.safe_region
    state_dim = len(safe_region)

    for traj_idx in range(n_trajectories):
        # Sample initial state
        state = np.zeros(state_dim)
        for j, (dim_name, (min_val, max_val)) in enumerate(safe_region.items()):
            state[j] = np.random.uniform(min_val, max_val)

        # Collect trajectory
        for step in range(max_steps):
            # Store state-action pair
            states_list.append(state.copy())

            # Get action from expert
            action = expert(state)
            actions_list.append(action)

            # Step the system
            state = system.step(state, action)

    # Convert to numpy arrays
    states = np.array(states_list)
    actions = np.concatenate(actions_list).reshape(-1, actions_list[0].size)

    print(f"Generated dataset with {len(states)} samples")
    print(f"State shape: {states.shape}")
    print(f"Action shape: {actions.shape}")

    if save_path:
        data = {"states": states, "actions": actions}
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved to {save_path}")

    return states, actions
