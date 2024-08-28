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

def plot_fdr_vs_power(alphas, fdr_list, power_list):
    """
    fdr_list: a list of tuples (mean_fdp, std_fdp)
    power_list: a list of tuples (mean_power, std_power)
    Draws the plot like in figure 3:
    Realized FDR and Power for conformal alignment at various FDR target levels.
    """
    fdrs, std_fdps = zip(*fdr_list)
    powers, std_powers = zip(*power_list)

    plt.figure(figsize=(7, 5))

    plt.plot(alphas, fdrs, '--bo', label='FDR')
    plt.plot(alphas, powers, '--ro', label='Power')
    plt.plot(alphas, alphas, '--g', label='Alpha')

    # shaded area representing the standard deviation
    plt.fill_between(alphas, np.array(fdrs) - np.array(std_fdps), np.array(fdrs) + np.array(std_fdps),
                     color='blue', alpha=0.1, label='FDR STD')
    plt.fill_between(alphas, np.array(powers) - np.array(std_powers), np.array(powers) + np.array(std_powers),
                     color='red', alpha=0.1, label='Power STD')
    
    plt.title("Conformal Alignment FDR and Power w.r.t FDR Target Level", fontdict={'size': 15})
    plt.xlabel('Target level of FDR', fontdict={'size': 12})
    plt.ylabel('FDR and Power', fontdict={'size': 12})
    plt.legend()
    plt.xticks([0.0, 0.5, 1.0])
    plt.yticks([0.0, 0.5, 1.0])
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plot_fdr_vs_power.png")
