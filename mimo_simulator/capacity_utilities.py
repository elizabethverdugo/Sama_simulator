import numpy as np


def calculate_capacity(singularValues, power_allocated, noise):

    """
    INPUTS:
    singularValues (array, 2D): from the Channel matrix, size UxS
    power_allocated (value, float): after an algorithm for allocation
    Noise (value, float): Noise spectral density

    Output:
    total_capacity(value, float): the capacity of the MIMO channel for this combination: BS-MS
    """

    # Ensure singular_values and power_allocation are arrays
    singularValues = np.array(singularValues)
    power_allocated = np.array(power_allocated)

    # Calculate the capacity
    capacity = np.sum(np.log2(1 + (singularValues ** 2 * power_allocated) / noise), axis=1)
    return capacity