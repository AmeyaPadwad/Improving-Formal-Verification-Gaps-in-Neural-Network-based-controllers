import numpy as np


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
    controller, system, n_trials=50, n_steps=100, verbose=False
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
            state[j] = np.random.uniform(min_val * 0.8, max_val * 0.8)

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
