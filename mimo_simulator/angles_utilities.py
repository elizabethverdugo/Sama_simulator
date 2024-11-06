import numpy as np

def calculate_aod(N, r_AS, sigma_AS):

    # Calculate sigma_AoD for each path
    #sigma_AoD = (r_AS * sigma_AS).reshape(BS, 1) To use when considering multiple BS
    sigma_AoD = r_AS * sigma_AS

    # Generate i.i.d. zero-mean Gaussian random variables for each multipath
    AoD_random_vars = np.random.randn(N, 1) * sigma_AoD   #temporal variable:  array of generated Gaussian random variables.
    #column vector of random variables drawn from the normal distribution with mean 0 and the calculated standard deviation

    #working only with real part
    AoD_random_vars = AoD_random_vars.real

    print("Generated AoD random variables:", AoD_random_vars)

    # Order these variables in increasing absolute value
    ordered_indices = np.argsort(np.abs(AoD_random_vars), axis=0).flatten()     #contains the indices that would sort the array by the absolute values of the elements
    ordered_AoD_vars = AoD_random_vars[ordered_indices]       #uses the indices to sort AoD_random_vars: array of the variables ordered by increasing absolute value.

    print("Ordered AoD variables by absolute value:", ordered_AoD_vars)

    # Assign AoDs to the ordered variables
    AoD_values = ordered_AoD_vars       #is simply the ordered list of AoD_random_vars


    return AoD_values, ordered_indices


def calculate_aoa(N, path_powers):
    """
    Calculate the Angle of Arrival (AoA) for each multipath component based on their power levels

    Args:
    - N (int): Number of multipath components.
    - path_powers (array): Power levels of each path (Pn).

    Returns:
    - np.array: Array of AoAs for each path.
    """

    # Convert path power to dB as internal part of expression of std dev in STEP 9
    path_powers_dB = 10 * np.log10(path_powers)

    # Calculate the standard deviation for AoA of each path
    std_dev_AoA = 104.12 * (1 - np.exp(-0.2175 * np.abs(path_powers_dB)))

    # Mean of the Gaussian distribution for AoAs, assumed to be 0
    mean_AoA = 0

    # Generate Gaussian random variables for AoAs
    AoAs = np.random.normal(mean_AoA, std_dev_AoA)

    return AoAs