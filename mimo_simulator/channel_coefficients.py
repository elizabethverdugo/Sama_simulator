import numpy as np

def calculate_channel_coef(N, M, S, U, theta_BS, theta_MS, delta_AoD, delta_AoA,
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