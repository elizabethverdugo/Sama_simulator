import numpy as np
import scipy.integrate as integrate


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

    num_bs, num_ues, N = path_powers.shape

    #subpath powers
    subpath_powers = (path_powers[..., np.newaxis]/ M)

    # Create sub-path phases
    subpath_phases = np.random.uniform(0, 360, (num_bs, num_ues, N, M))

    # Create sub-path AoD offsets matrix
    subpath_offsets = np.tile(offset_values, (num_bs, num_ues, N, 1))

    return subpath_powers, subpath_phases, subpath_offsets


def calculate_offset_aoas(M, path_powers):
    """
    Calculate offset AoAs for the sub-paths of each path at the MS using specified values.

    Inputs:
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
    num_bs, num_ues, N = path_powers.shape

    # Create sub-path AoD offsets matrix
    subpath_AoA_offsets_per_path = np.tile(offset_values, (N, 1))

    subpath_AoA_offsets = np.tile(subpath_AoA_offsets_per_path,(num_bs,num_ues, 1, 1))

    return subpath_AoA_offsets


def associate_subpath(bs_subpath_offsets, ms_subpath_offsets):
    """
    Associate the BS and MS sub-paths by randomly pairing each BS sub-path with an MS sub-path.

    Inputs:
    - bs_subpath_offsets (np.array): Array of BS sub-path offsets of shape (N, M). offset of AoD
    - ms_subpath_offsets (np.array): Array of MS sub-path offsets of shape (N, M). offset of AoA

    Returns:
    - np.array: Reassociated MS sub-path offsets after pairing.
    """

    num_bs, num_ue, N, M = bs_subpath_offsets.shape         #define the size of the 4D matrix

    # Generate a random permutation for each combination: BS, UE, N
    permuted_indices = np.random.permutation(M)

    # Reassociate MS sub-path offsets by using the permutation, (acros all dimensions)
    reassociated_ms_subpath_offsets = np.take_along_axis(ms_subpath_offsets,
                                                         np.expand_dims(permuted_indices, axis=(0, 1, 2)),
                                                         axis=-1)

    return reassociated_ms_subpath_offsets


def calculate_angles(theta_BS, delta_AoD, AoD_offsets, theta_MS, delta_AoA, AoA_offsets):
    #calculate_angles(theta_BS, delta_AoD, AoD_offsets, theta_MS, delta_AoA, AoA_offsets):
    """
    Calculate the AoDs, AoAs, and corresponding antenna gains for BS and MS sub-paths.

    Inputs:
    - theta_BS (np.array): Broadside angles for each BS-UE pair (num_bs, num_ue)
    - delta_AoD (np.array): AoD deviations for each BS-UE-path (num_bs, num_ue, N)
    - AoD_offsets (np.array): AoD offsets for each BS-UE-path-subpath (num_bs, num_ue, N, M)
    - theta_MS (np.array): Broadside angles for each MS (num_bs, num_ue)
    - delta_AoA (np.array): AoA deviations for each BS-UE-path (num_bs, num_ue, N)
    - AoA_offsets (np.array): AoA offsets for each BS-UE-path-subpath (num_bs, num_ue, N, M)


    Returns:
    - AoDs (np.array): Angles of departure for all sub-paths (num_bs, num_ue, N, M)
    - AoAs (np.array): Angles of arrival for all sub-paths (num_bs, num_ue, N, M)


    """
    # sub-path structure size
    theta_BS_expanded = theta_BS[:, :, np.newaxis, np.newaxis]  #(num_bs, num_ue, 1, 1)
    delta_AoD_expanded = delta_AoD[:, :, :, np.newaxis]  #(num_bs, num_ue, N, 1)

    theta_MS_expanded = theta_MS[:, :, np.newaxis, np.newaxis]  #(num_bs, num_ue, 1, 1)
    delta_AoA_expanded = delta_AoA[:, :, :, np.newaxis]  #(num_bs, num_ue, N, 1)

    # Calculate AoDs
    AoDs = theta_BS_expanded + delta_AoD_expanded + AoD_offsets
    # Calculate AoAs
    AoAs = theta_MS_expanded + delta_AoA_expanded + AoA_offsets

    # Normalize angles
    AoDs = normalize_angle(AoDs)
    AoAs = normalize_angle(AoAs)

    return AoDs, AoAs

def normalize_angle(angle):
    """
    normalize the angle -180 to 180 degrees.
    """
    return (angle + 180) % 360 - 180


def calculate_gains_BS(Angle, Antenna_Sectors):

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

    # Compute attenuation A(θ), eq (4.5-1)
    A_theta = -np.minimum(12 * (Angle / theta_3dB) ** 2, A_m)

    # Compute G(θ) in dB, eq (4.5-3)
    G_total_dB = gain_dBi + A_theta

    # Convert dB gain to linear scale
    G_total_linear = 10 ** (G_total_dB / 10)

    #print("Min A_theta (dB):", np.min(A_theta))


    return G_total_linear


def calculate_gains_MS(Angle):
    """
    Compute omnidirectional antenna gains for MS.
    - MS antennas are omnidirectional, meaning their gain is fixed at -1 dBi.
    Inputs:
    - Angle (np.array): AoA angles (used only for shape consistency)
    Output:
    - G_total_linear (np.array): Gain values in linear scale (num_bs, num_ue, N, M)
    """
    # MS is omnidirectional -> gain = -1 dBi (see 4.6.1)
    G_total_dB = -1  # dBi
    G_total_linear = 10 ** (G_total_dB / 10)  # Convert to linear scale

    #array with shape as input Angle
    return np.full_like(Angle, G_total_linear)




def compute_N0(theta_mean, sigma, theta_range=(-180, 180)):
    """
    Compute the normalization factor N0 using numerical integration (Eq. 4.5-4).

    Args:
    - theta_mean: Mean AoD (in degrees)
    - sigma: RMS angular spread (in degrees)
    - theta_range: Integration range (default: -180 to 180 degrees)

    Returns:
    - N0: Normalization factor for PAS.
    """

    def integrand(theta):
        """ Function inside the integral """
        return np.exp(-np.sqrt(2) * np.abs(theta - theta_mean) / sigma) * bs_gain(theta)

    # Perform numerical integration
    N0, _ = integrate.quad(integrand, theta_range[0], theta_range[1])

    return N0


def power_azimuth_spectrum(theta, theta_mean, sigma, N0):
    """
    Compute the normalized PAS P(θ,σ,θ̄) using Eq. (4.5-2).

    Args:
    - theta: Angles where PAS is computed.
    - theta_mean: Mean AoD (in degrees).
    - sigma: RMS angular spread (in degrees).

    Returns:
    - PAS values normalized by N0.
    """

    # Compute PAS before normalization
    PAS_unnormalized = np.exp(-np.sqrt(2) * np.abs(theta - theta_mean) / sigma) * bs_gain(theta)

    # Normalize PAS
    PAS_normalized = PAS_unnormalized / N0

    return PAS_normalized


def bs_gain(theta, theta_3dB=70, A_m=20, gain_dBi=14):
    """
    Compute the BS gain pattern G(θ) as per equation (4.5-3).
    """
    A_theta = -np.minimum(12 * (theta / theta_3dB) ** 2, A_m)  # Eq (4.5-1)
    G_theta_dB = gain_dBi + A_theta  # Eq (4.5-3)
    return 10 ** (G_theta_dB / 10)  # Convert to linear scale
