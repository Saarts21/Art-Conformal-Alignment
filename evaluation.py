import numpy as np
import matplotlib.pyplot as plt

# Evaluate the experiment and plot the results


def fdp(test_set_A, rejection_set_indices, c):
    """
    Compute the FDP (False discovery proportion) of a single experiment (like equation 1.1 but without the mean)
    test_set_A - ground truth alignment score for each sample in the test set.
    rejection_set_indices - a boolean vector giving a true value to the selected test samples.
    c - alignment level.
    """
    selected_but_not_aligned = np.logical_and(test_set_A <= c, rejection_set_indices)
    S_size = np.sum(rejection_set_indices)
    return np.sum(selected_but_not_aligned) / max(S_size, 1)

def power(test_set_A, rejection_set_indices, c):
    """
    Compute the POWER of a single experiment (not the mean like in the problem setup)
    test_set_A - ground truth alignment score for each sample in the test set.
    rejection_set_indices - a boolean vector giving a true value to the selected test samples.
    c - alignment level.
    """
    aligned = test_set_A > c
    aligned_and_selected = np.logical_and(aligned, rejection_set_indices)
    return np.sum(aligned_and_selected) / max(np.sum(aligned), 1)

def plot_fdp_vs_power(alphas, fdrs, powers):
    """
    Draws the plot like in figure 3:
    Realized FDP and Power for conformal alignment at various FDR target levels.
    """
    assert alphas.size == fdrs.size == powers.size

    plt.plot(alphas, fdrs, '--bo', label='FDP')
    plt.plot(alphas, powers, '--ro', label='Power')
    plt.plot(alphas, alphas, '--g')
    plt.title("Conformal Alignment FDP and Power w.r.t FDR Target Level")
    plt.xlabel('Target level of FDR', fontdict={'size': 12})
    plt.ylabel('FDP and Power', fontdict={'size': 12})
    plt.legend()
    plt.xticks([0.0, 0.5, 1.0])
    plt.yticks([0.0, 0.5, 1.0])
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plot_fdp_vs_power.png")

# Unit tests
f = fdp(np.array([1, 0, 1, 1]), np.array([False, True, True, True]), 0)
assert np.isclose(f, 0.33333)

p = power(np.array([1, 0, 1, 1]), np.array([False, True, True, True]), 0)
assert np.isclose(p, 0.66666)

plot_fdp_vs_power(np.arange(0.0, 1.0, 0.1), np.arange(0.0, 1.0, 0.1) - 0.05, np.arange(0.0, 1.0, 0.1) + 0.05)