import numpy as np

def water_filling(singular_values_squared, total_power, noise_power):
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


