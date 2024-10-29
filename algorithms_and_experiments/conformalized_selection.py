import numpy as np

# Compute the conformal p-values p_j according to Equation (3.2).
# Apply BH to the conformal p-values: S ← BH(p_1 , . . . , p_m).


def conformal_p_value(calib_set_A, calib_set_A_hat, a_hat_j, c):
    """
    Given a test sample X_n+j, computes its conformal p-value according to equation 3.2:
    calib_set_A - ground truth alignment score for each sample in the calibration set.
    calib_set_A_hat - predicted alignment score for each sample in the calibration set.
    a_hat_j - predicted alignment score of X_n+j.
    c - alignment level.
    return: p_j
    """
    assert calib_set_A.size == calib_set_A_hat.size
    calib_set_size = calib_set_A.size

    indicator = np.logical_and(calib_set_A <= c, calib_set_A_hat >= a_hat_j)
    p_j = (1 + np.sum(indicator)) / (calib_set_size + 1)
    return p_j

def benjamini_hochberg_procedure(test_set_p_values, alpha):
    """
    Given the test set conformal p-values, and the target FDR level (alpha), compute a
    boolean vector of the rejection set indices, according to the order of test_set_p_values.
    """
    sorted_p_values = np.sort(test_set_p_values)
    m = test_set_p_values.size # test set size

    k_array = np.arange(1, m + 1) # [1, 2, ... , m]
    thresholds = k_array * (alpha / m) # ak/m
    indicator = sorted_p_values <= thresholds
    filtered_k_values = np.where(indicator, k_array, 0)
    k_star = np.max(filtered_k_values)

    rejection_set_indices = test_set_p_values <= ((k_star * alpha) / m)
    return rejection_set_indices

def conformalized_selection(calib_set_A, calib_set_A_hat, test_set_A_hat, alpha, c):
    """
    Select the aligned test samples according to their predicted alignment scores,
    by performing the BH procedure. If a test sample's null-hypothesis is rejected,
    it is selected to the aligned samples set.
    calib_set_A - ground truth alignment score for each sample in the calibration set.
    calib_set_A_hat - predicted alignment score for each sample in the calibration set.
    test_set_A_hat - test set predicted alignment values.
    alpha - target FDR level.
    c - alignment level.
    return: a boolean vector giving a true value to the selected test samples.
    """
    test_set_p_values = np.fromiter(
        (conformal_p_value(calib_set_A, calib_set_A_hat, a_hat_j, c) for a_hat_j in test_set_A_hat),
        float)
    rejection_set_indices = benjamini_hochberg_procedure(test_set_p_values, alpha)
    return rejection_set_indices

# Unit tests
p = conformal_p_value(np.array([1, 1, 1, 0, 0]), np.array([1, 0, 1, 0, 1]), 1, 0)
assert np.isclose(p, 0.33333)

r = benjamini_hochberg_procedure(np.array([0.011, 0.3, 0.029]), 0.1)
assert np.array_equal(r, np.array([True, False, True]))
