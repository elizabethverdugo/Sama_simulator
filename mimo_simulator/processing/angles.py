import numpy as np
import matplotlib.pyplot as plt

def calculate_aod(d, AoD_bs, N, r_AS, sigma_AS, mode="realistic_gain",
                  beam_pointing=None, sector_map=None, beam_gain=None, downtilts=None):

    # Calculate AoD values considering beamforming sectorizaion
    """
    Parameters:
    - d: Distance matrix (num_bs x num_ues).
    - AoD_bs: Base station azimuth angles (num_bs x num_ues).
    - N: Number of multipaths.
    - r_AS: Ratio of angular spread (environment parameter).
    - sigma_AS: Angular spread array-like (per BS).
    -mode: 'pas_gain_model' or 'realistic_gain'
    - beam_pointing: Array of sector directions (n_sector)
    -sector_map: matrix (num:_bs x num_eu ), sector assigned per link
    -beam_gain: 3D gain pattern (for one array) 1,360,360

    Output
    - AoD_values: AoD values for each multipath component (num_bs x num_ues x N).
    - ordered_indices: Indices of ordered variables (num_bs x num_ues x N).

    """
    num_bs, num_ues = d.shape
    sigma_AoD = (r_AS * sigma_AS).reshape(num_bs, 1, 1)  # To use when considering multiple BS
    # Generate angles using PAS model (Eq 4.5-2)
    theta_offset = np.random.laplace(loc=0, scale=sigma_AoD, size=(num_bs, num_ues, N))

    if mode == "pas_gain_model":
        theta_3dB = 35 #default value if no pattern, see 3GPP for the sectorized cases
        A_m = 23  #default, see 3GPP

        G_theta_dB = -np.minimum(12*(theta_offset/theta_3dB) ** 2, A_m)
        G_theta_linear = 10**(G_theta_dB/10)

        # Re-weight angle selection based on PAS * Gain
        weights = np.exp(-np.sqrt(2) * np.abs(theta_offset) / sigma_AoD) * G_theta_linear
        # Ensure no zero probabilities (avoiding np.random.choice() errors)
        weights = np.clip(weights, 1e-10, None)  # Minimum probability value

        # Re-normalize (ensuring sum is exactly 1)
        weights /= np.sum(weights, axis=-1, keepdims=True)

        # Generate AoD values using batch-wise selection
        # Sample angles based on PAS distribution (see Section 4.5.4)
        AoD_PAS_sampled = np.array([
            np.random.choice(theta_offset[i, j], size=N, p=weights[i, j])
            for i in range(theta_offset.shape[0])
            for j in range(theta_offset.shape[1])
        ]).reshape(theta_offset.shape[:2] + (N,))

        # Generate i.i.d. zero-mean Gaussian random variables for each multipath
        # AoD_random_vars = np.random.randn(num_bs, num_ues, N) * sigma_AoD   #temporal variable:  array of generated Gaussian random variables.
        # Generate AoD variations using Laplacian distribution: v2 EV
        # Add controlled Laplacian variation for realism (see Section 4.5.4)
        AoD_random_vars = AoD_PAS_sampled + np.random.laplace(loc=0, scale=sigma_AoD, size=(num_bs, num_ues, N))


    elif mode == "realistic_gain":
        # use realistic beam pattern to derive gain
        # beam direction for each link
        OmegaBS = beam_pointing[sector_map]
        gain_offset = AoD_bs - OmegaBS

        az_idx = np.round(AoD_bs).astype(int) % 360

        phi_idx = downtilts.squeeze()[sector_map]
        phi_idx = np.mod(np.rint(phi_idx).astype(int),360)

        bs_flat = sector_map.flatten()
        az_flat = az_idx.flatten()
        phi_flat = phi_idx.flatten()


        # 2. get gain values per sector azimuth, elevation
        gain_flat = beam_gain[0, az_flat, phi_flat]

        gain_matrix = gain_flat.reshape(sector_map.shape)
        G_theta_linear = 10 **(gain_matrix/10)

        # Re-weight angle selection based on PAS * Gain
        weights = np.exp(-np.sqrt(2) * np.abs(theta_offset) / sigma_AoD) * G_theta_linear[:, :, np.newaxis]
        # Ensure no zero probabilities (avoiding np.random.choice() errors)
        weights = np.clip(weights, 1e-10, None)  # Minimum probability value

        # Re-normalize (ensuring sum is exactly 1)
        weights /= np.sum(weights, axis=-1, keepdims=True)

        AoD_random_vars = np.random.laplace(loc=0, scale=sigma_AoD, size=(num_bs, num_ues, N))


    #print("Generated AoD random variables:", AoD_random_vars)

    # Order these variables in increasing absolute value
    ordered_indices = np.argsort(np.abs(AoD_random_vars), axis=-1)     #contains the indices that would sort the array by the absolute values of the elements
    ordered_AoD_vars = np.take_along_axis(AoD_random_vars, ordered_indices, axis=-1)      #uses the indices to sort AoD_random_vars: array of the variables ordered by increasing absolute value.

    #print("Ordered AoD variables by absolute value:", ordered_AoD_vars)

    # Assign AoDs to the ordered variables
    #AoD_values = ordered_AoD_vars       #is simply the ordered list of AoD_random_vars
    AoD_values = AoD_bs[:, :, np.newaxis] + ordered_AoD_vars  # Add AoD_bs to each multipath AoD

    return AoD_values, ordered_indices, G_theta_linear

def calculate_aoa1(N, path_powers):
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


def calculate_aoa(num_bs, num_ues, N, path_powers, sigma_AS, AS_type="laplacian"):
    """
    Calculate the Angle of Arrival (AoA) for each multipath component based on
    power levels and PAS distribution (see 4.6.4).

    Args:
    - num_bs (int): Number of base stations.
    - num_ues (int): Number of user equipments.
    - N (int): Number of multipath components.
    - path_powers (array): Power levels of each path (num_bs x num_ues x N).
    - sigma_AS (float or array): RMS Angle Spread (35 or 104 degrees, per BS).
    - AS_type (str): "laplacian" (default) or "uniform" for the type of PAS.

    Returns:
    - np.array: Array of AoAs for each BS-UE pair and path (num_bs x num_ues x N).
    """

    if AS_type == "uniform":
        # Uniform distribution over 360 degrees
        AoAs = np.random.uniform(-180, 180, size=(num_bs, num_ues, N))

    elif AS_type == "laplacian":
        # Expand sigma_AS to match the (num_bs, num_ues, N) shape
        sigma_AS = sigma_AS.reshape(num_bs, 1, 1)  # Ensure proper broadcasting
        sigma_AS = np.tile(sigma_AS, (1, num_ues, N))  # Expand along UE and path dims

        # Generate AoAs using a Laplacian distribution
        AoAs = np.random.laplace(loc=0, scale=sigma_AS, size=(num_bs, num_ues, N))

    else:
        raise ValueError("Invalid AS_type. Choose 'laplacian' or 'uniform'.")

    return AoAs
