import numpy as np


def acquire_subpath_parameters(N, M, path_powers):
    """
    Calculate the powers, phases, and AoD offsets for sub-paths
    STEP 8

    Inputs:
    - N (int): Number of multipath components.
    - M (int): Number of sub-paths per path.
    - (not as input) offsetAoD (int): offset for the AoD TABLE 5.2
    - path_powers (array): Array containing the power for each path.

    Returns:
    - tuple: A tuple containing arrays for sub-path powers, phases, and AoD offsets.
    """
    # Define the sub-path offset values (adjust based on Table5.2)
    # offset_values = np.array([offsetAoD, -offsetAoD] + [0] * (M - 2))

    all_offset_values = [0.0894, -0.0894, 0.2826, -0.2826, 0.4984, -0.4984, 0.7431, -0.7431, 1.0257, -1.0257,
                     1.3594, -1.3594, 1.7688, -1.7688, 2.2961, -2.2961, 3.0389, -3.0389, 4.3101, -4.3101]


    offset_values = all_offset_values[:M]

    # Create sub-path power matrix: takes the power per path and divide it into the multipaths
    #subpath_powers = np.tile(path_powers[:, np.newaxis] / M, (1, M))
    subpath_powers = np.tile((path_powers / M).reshape(N, 1), (1, M))

    # Create sub-path phases matrix: uniform from 0 to 360
    subpath_phases = np.random.uniform(0, 360, (N, M))

    # Create sub-path AoD offsets matrix
    subpath_offsets = np.tile(offset_values, (N, 1))

    return subpath_powers, subpath_phases, subpath_offsets


def calculate_offset_aoas(N, M):
    """
    Calculate offset AoAs for the sub-paths of each path at the MS using specified values.

    Args:
    - N (int): Number of multipath components.
    - M (int): Number of sub-paths per path. Default is 20.
    - (not as input) offsetAoA (int): offset for the AoA TABLE 5.2

    Returns:
    - tuple: Array of AoA offsets for each sub-path of each path.
    """

    # Initialize the offsets array
    # First two values are given, the rest are temporarily zeros
    #offset_values = np.array([offsetAoA, -offsetAoA] + [0] * (M - 2))
    all_offset_values = [ 1.5649,  -1.5649,  4.9447,  -4.9447,  8.7224,  -8.7224, 13.0045, -13.0045, 17.9492, -17.9492,
                     23.7899, -23.7899, 30.9538, -30.9538, 40.1824, -40.1824, 53.1816, -53.1816, 75.4274, -75.4274]

    offset_values = all_offset_values[:M]

    # Create sub-path AoD offsets matrix
    subpath_AoA_offsets = np.tile(offset_values, (N, 1))

    return subpath_AoA_offsets


def associate_subpath(bs_subpath_offsets, ms_subpath_offsets):
    """
    Associate the BS and MS sub-paths by randomly pairing each BS sub-path with an MS sub-path.

    Args:
    - bs_subpath_offsets (np.array): Array of BS sub-path offsets of shape (N, M). offset of AoD
    - ms_subpath_offsets (np.array): Array of MS sub-path offsets of shape (N, M). offset of AoA

    Returns:
    - np.array: Reassociated MS sub-path offsets after pairing.
    """
    N, M = bs_subpath_offsets.shape         #define the size of the matrix

    # Generate a random permutation for each path
    permuted_indices = np.array([np.random.permutation(M) for _ in range(N)])

    #print("Permuted indices", permuted_indices)

    # Reassociate MS sub-path offsets based on the random pairing
    reassociated_ms_subpath_offsets = np.array([ms_subpath_offsets[n, permuted_indices[n]] for n in range(N)])


    return reassociated_ms_subpath_offsets


def calculate_angles(N, M, theta_BS, delta_AoD, AoD_offsets, theta_MS, delta_AoA, AoA_offsets):
    """
    Calculate the AoDs, AoAs, and corresponding antenna gains for BS and MS sub-paths.

    Args:
    - theta_BS (float): Broadside angle for the BS antenna array.
    - delta_AoD (np.array): Array of AoD deviations for each path (N,).
    - AoD_offsets (np.array): Array of AoD offsets for each sub-path of each path (N, M).
    - theta_MS (float): Broadside angle for the MS antenna array.
    - delta_AoA (np.array): Array of AoA deviations for each path (N,).
    - AoA_offsets (np.array): Array of AoA offsets for each sub-path of each path (N, M).

    Returns:
    - tuple: Arrays of AoDs, AoAs, and antenna gains for BS and MS sub-paths.
    """
    # Reshape delta_AoD and delta_AoA
    delta_AoD = delta_AoD.reshape(N, 1)
    delta_AoA = delta_AoA.reshape(N, 1)

    # Calculate AoDs
    AoDs = theta_BS + delta_AoD + AoD_offsets

    # Calculate AoAs
    AoAs = theta_MS + delta_AoA + AoA_offsets

    # Normalize angles
    AoDs = normalize_angle(AoDs)
    AoAs = normalize_angle(AoAs)

    return AoDs, AoAs

def normalize_angle(angle):
    """
    Normalize the angle to the range [-180, 180] degrees.
    """
    return (angle + 180) % 360 - 180


def calculate_gains(Angle, Antenna_Sectors):

    #This part should be in param.yml
    if Antenna_Sectors == '3':
        theta_3dB = 70       #3dB beamwidth in degrees
        A_m = 20             #Max. attenuation
        gain_dBi = 14  # for 3-sector scenario
    elif Antenna_Sectors == '6':
        theta_3dB = 35
        A_m = 23
        gain_dBi = 17  # for 6-sector scenario
    else:
        raise ValueError("Antenna_Sectors must be either '3' or '6'")

    # Normalize angles to the range [-180, 180]
    Angle = normalize_angle(Angle)

    # Calculate A(theta), eq (4.5-1)
    A_theta = -np.minimum(12 * (Angle / theta_3dB) ** 2, A_m)

    # Calculate G(theta), eq (4.5-3)
    G_theta_linear = 10 ** (0.1 * A_theta)  # Gain in linear scale
    G_theta_dB = 10 * np.log10(G_theta_linear)  # Gain in dB scale

    # Considering also the base gain
    G_total_dB = G_theta_dB + gain_dBi  # Total gain in dB

    # Convert total gain back to linear scale
    G_total_linear = 10 ** (G_total_dB / 10)

    #print("Angles:\n", Angle)  # Debug: Print the angles
    #print("A_theta:\n", A_theta)  # Debug: Print A(theta)
    #print("G_theta_linear:\n", G_theta_linear)  # Debug: Print G(theta) in linear scale
    #print("G_total_dB:\n", G_total_dB)  # Debug: Print total gain in dB
    #print("G_total_linear:\n", G_total_linear)  # Debug: Print total gain in linear scale

    return G_total_linear