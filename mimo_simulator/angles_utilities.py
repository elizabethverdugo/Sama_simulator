import numpy as np

def calculate_aod(d, AoD_bs, N, r_AS, sigma_AS):

    # Calculate sigma_AoD for each path
    """
    Parameters:
    - d: Distance matrix (num_bs x num_ues).
    - AoD_bs: Base station azimuth angles (num_bs x num_ues).
    - N: Number of multipaths.
    - r_AS: Ratio of angular spread (environment parameter).
    - sigma_AS: Angular spread array-like (per BS).

    Output
    - AoD_values: AoD values for each multipath component (num_bs x num_ues x N).
    - ordered_indices: Indices of ordered variables (num_bs x num_ues x N).

    """
    num_bs, num_ues = d.shape
    sigma_AoD = (r_AS * sigma_AS).reshape(num_bs, 1, 1)  #To use when considering multiple BS
    sigma_AoD = np.tile(sigma_AoD, (1, num_ues, 1))  # using ues dimension

    # Generate i.i.d. zero-mean Gaussian random variables for each multipath
    AoD_random_vars = np.random.randn(num_bs, num_ues, N) * sigma_AoD   #temporal variable:  array of generated Gaussian random variables.


    #working only with real part
    AoD_random_vars = AoD_random_vars.real

    #print("Generated AoD random variables:", AoD_random_vars)

    # Order these variables in increasing absolute value
    ordered_indices = np.argsort(np.abs(AoD_random_vars), axis=-1)     #contains the indices that would sort the array by the absolute values of the elements
    ordered_AoD_vars = np.take_along_axis(AoD_random_vars, ordered_indices, axis=-1)      #uses the indices to sort AoD_random_vars: array of the variables ordered by increasing absolute value.

    #print("Ordered AoD variables by absolute value:", ordered_AoD_vars)

    # Assign AoDs to the ordered variables
    #AoD_values = ordered_AoD_vars       #is simply the ordered list of AoD_random_vars
    AoD_values = AoD_bs[:, :, np.newaxis] + ordered_AoD_vars  # Add AoD_bs to each multipath AoD

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