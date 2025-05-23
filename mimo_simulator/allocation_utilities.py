import numpy as np

def water_filling1(singular_values_squared, total_power, noise_power):
    """
    Perform water-filling power allocation.

    INPUTS:
    singular_values: 2D array of singular values
    total_power: Total available power for transmission.
    noise_power: Noise power.

    OUTPUT:
    Allocated power for each channel (2D array).
    """

    num_rows, num_cols = singular_values_squared.shape
    power_allocation_final = np.zeros((num_rows, num_cols), dtype=float)

    for i in range(num_rows):
        inverse_snr = np.zeros(num_cols, dtype=float)
        non_zero_indices = singular_values_squared[i] > 0
        inverse_snr[non_zero_indices] = noise_power / singular_values_squared[i, non_zero_indices]

        sorted_indices = np.argsort(inverse_snr)
        sorted_inverse_snr = inverse_snr[sorted_indices]

        water_level = (total_power + np.sum(sorted_inverse_snr)) / num_cols
        power_allocation = np.maximum(water_level - sorted_inverse_snr, 0)

        while np.sum(power_allocation) > total_power:
            excess_power = np.sum(power_allocation) - total_power
            water_level -= excess_power / num_cols
            power_allocation = np.maximum(water_level - sorted_inverse_snr, 0)

        power_allocation_final[i, sorted_indices] = power_allocation

    return power_allocation_final


def uniform_allocation(singular_values_squared, total_power):
    num_rows, num_cols = singular_values_squared.shape
    power_allocation_final = np.zeros((num_rows, num_cols), dtype=float)

    power_allocation = np.full((num_rows, num_cols), total_power / (num_rows * num_cols))
    power_allocation_final = power_allocation

    return power_allocation_final


def uniform_power(singular_vals: np.ndarray,
                  total_power: float):

    mask = singular_vals > 0
    n_active = np.sum(mask, axis=-1, keepdims=True)
    p_per_mode = np.where(mask, total_power / np.maximum(n_active,1),0.)
    return p_per_mode


def water_filling(singular_vals: np.ndarray,
                  total_power: float,
                  noise_psd: float):

    sv2 = singular_vals**2
    n_links, n_eig = sv2.shape
    P_alloc = np.zeros_like(sv2)

    for k in range(n_links):
        g_k = sv2[k, :]
        mask = g_k >  0
        g = g_k[mask]
        m = g.size
        if m == 0:
            continue

        inv_snr = noise_psd / g
        inv_snr_sorted = np.sort(inv_snr)
        prefix_sum = np.cumsum(inv_snr_sorted)

        j = np.argmax((total_power + prefix_sum) / np.arange(1, m+1) > inv_snr_sorted)
        water_level = (total_power + prefix_sum[j]) / (j+1)

        p_i = np.maximum(water_level - inv_snr, 0.)
        P_alloc[k, mask] = p_i

    return P_alloc





