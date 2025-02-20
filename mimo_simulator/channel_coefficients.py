import numpy as np

def calculate_channel_coef1(N, M, S, U, theta_BS, theta_MS, delta_AoD, delta_AoA,
                           AoD_offsets, AoA_offsets, subpath_powers, subpath_phases,
                           G_BS, G_MS, sigma_SF, d_bs, d_ms, v, theta_v, f, time):

    """
    Inputs:
    N: Number of paths
    M: Number of sub-paths per path
    S: Number of BS antenna elements
    U: Number of MS antenna elements
    theta_BS: Broadside angle for BS
    theta_MS: Broadside angle for MS
    delta_AoD (N)
    delta_AoA (N)
    AoD_offsets (N, M)
    AoA_offsets (N, M)
    subpath_powers (N, M)
    subpath_phases (N, M) # Phases in degrees
    G_BS (N, M)
    G_MS (N, M)
    sigma_SF: (int) one parameter applied as a bulk parameter
    d_bs: Distance between BS antenna elements in wavelengths ( = 0.5  ?)
    d_ms: Distance between MS antenna elements in wavelengths ( = 0.5  ?)
    v: velocity os the MS?
    theta_v: angle of velocity
    time (int): time ?



    """

    c = 3e8 #light speed

    # Wavelength
    lambda_c = c / f
    # Wavenumber
    k = 2 * np.pi / lambda_c

    # BS and MS distance vectors (Should this be different?)
    d_s = np.arange(S) * d_bs * lambda_c
    d_u = np.arange(U) * d_ms * lambda_c

    # create H of zeros
    H = np.zeros((U, S, N), dtype=complex)

    #Loop1: each path(N) and each sub-path(M)
    for n in range(N):
        P_n2 = np.sqrt(subpath_powers[n])  # Power of the nth path

        for m in range(M):
            theta_n_m_AoD = np.radians(theta_BS + delta_AoD[n] + AoD_offsets[n, m])  # Convert to radians
            theta_n_m_AoA = np.radians(theta_MS + delta_AoA[n] + AoA_offsets[n, m])  # Convert to radians

            # Calculate phases for BS and MS
            a_BS = np.exp(1j * k * d_s * np.sin(theta_n_m_AoD))
            a_MS = np.exp(1j * k * d_u * np.sin(theta_n_m_AoA))

            # Doppler shift due to movement of the MS
            shift = np.exp(1j * k * v * time * np.cos(theta_n_m_AoA - np.radians(theta_v)))

            # Calculate the contribution of this subpath
            subpath_contrib = (P_n2[m] * np.sqrt(sigma_SF) * np.sqrt(G_BS[n, m]) * np.sqrt(G_MS[n, m]) *
                               shift * np.exp(1j * np.radians(subpath_phases[n, m])))


            # Debugging prints
            print(f"Subpath Contribution (n={n}, m={m}): {subpath_contrib}")
            print(f"a_BS (n={n}, m={m}): {a_BS}")
            print(f"a_MS (n={n}, m={m}): {a_MS}")
            print(f"Outer Product (n={n}, m={m}): {np.outer(a_MS, a_BS)}")

            # Add the subpath contribution to H
            H[:, :, n] += subpath_contrib * np.outer(a_MS, a_BS)

    return H


def calculate_channel_coef(num_BS, num_MS, N, M, S, U, theta_BS, theta_MS, delta_AoD, delta_AoA,
                           AoD_offsets, AoA_offsets, subpath_powers, subpath_phases,
                           G_BS, G_MS, sigma_SF, d_bs, d_ms, v, theta_v, f, time):
    """
     MIMO channel coefficient matrix H. Multiple BS and MS

    Args:
    - num_BS (int): Number of base stations
    - num_MS (int): Number of mobile stations
    - N (int): Number of paths
    - M (int): Number of sub-paths per path
    - S (int): Number of BS antenna elements
    - U (int): Number of MS antenna elements
    - theta_BS (num_BS, num_MS): Broadside angle for each BS-MS pair
    - theta_MS (num_BS, num_MS): Broadside angle for each MS-BS pair
    - delta_AoD (num_BS, num_MS, N): AoD deviations per path per BS-MS
    - delta_AoA (num_BS, num_MS, N): AoA deviations per path per BS-MS
    - AoD_offsets (num_BS, num_MS, N, M): AoD offsets per BS-MS per path-subpath
    - AoA_offsets (num_BS, num_MS, N, M): AoA offsets per BS-MS per path-subpath
    - subpath_powers (num_BS, num_MS, N, M): Power for each subpath
    - subpath_phases (num_BS, num_MS, N, M): Phase for each subpath
    - G_BS (num_BS, num_MS, N, M): Gain for each BS-MS per subpath
    - G_MS (num_BS, num_MS, N, M): Gain for each MS per subpath
    - sigma_SF: Shadow fading parameter
    - d_bs: Distance between BS antenna elements in wavelengths
    - d_ms: Distance between MS antenna elements in wavelengths
    - v: Velocity of MS
    - theta_v: Angle of velocity
    - f: Frequency in Hz
    - time: Time index

    Returns:
    - H (num_BS, num_MS, U, S, N, M): MIMO channel coefficient matrix
    """

    c = 3e8  # Speed of light
    lambda_c = c / f  # Wavelength
    k = 2 * np.pi / lambda_c  # Wavenumber

    # Compute element positions
    d_s = np.arange(S).reshape(1, 1, 1, 1, S) * d_bs * lambda_c
    d_u = np.arange(U).reshape(1, 1, 1, 1, U) * d_ms * lambda_c


    # Compute full angles
    theta_n_m_AoD = np.radians(theta_BS[:, :, None, None] + delta_AoD[:, :, :, None] + AoD_offsets)
    theta_n_m_AoA = np.radians(theta_MS[:, :, None, None] + delta_AoA[:, :, :, None] + AoA_offsets)

    # Compute BS and MS response vectors
    a_BS = np.exp(1j * k * d_s * np.sin(theta_n_m_AoD[:, :, :, :, None]))
    a_MS = np.exp(1j * k * d_u * np.sin(theta_n_m_AoA[:, :, :, :, None]))


    # Doppler shift
    shift = np.exp(1j * k * v * time * np.cos(theta_n_m_AoA - np.radians(theta_v)))  # Shape: (num_BS, num_MS, N, M)

    # Compute subpath contribution
    subpath_contrib = (np.sqrt(subpath_powers) * np.sqrt(sigma_SF) * np.sqrt(G_BS) *
                       np.sqrt(G_MS) * shift * np.exp(1j * np.radians(subpath_phases)))  # Shape: (num_BS, num_MS, N, M)

    # Compute H using broadcasting
    H = subpath_contrib[:, :, :, :, None, None] * (a_MS @ a_BS)  # Shape: (num_BS, num_MS, U, S, N, M)

    return H
