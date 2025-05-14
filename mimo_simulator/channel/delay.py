import numpy as np

def delay(N, r_DS, sigma_DS, dist_map=None, c=3e8, delay_method="3GPP"):
    """
    Compute delay based on one of three methods

    Parameters:
    - N: # multipath components.
    - r_DS: Ratio of RMS delay to delay spread.
    - sigma_DS: Delay spread (array-like).
    - dist_map: Distance matrix (optional for distance-based or combined methods).
    - c: Speed of light (3e8 m/s).
    - delay_method: "3GPP", "distance", or "both".

    Output:
    - tn: Delays (s)
    """
    if delay_method == "3GPP":
        # 3GPP Delay Model
        num_bs = sigma_DS.shape[0]
        Zn = np.random.rand(num_bs, N)  # Generate N multipaths per BS
        tn = -r_DS * sigma_DS[:, np.newaxis] * np.log(Zn)  # Delay spread for each BS
        tn = np.sort(tn, axis=1)  # Sort delays for each BS
        tn = tn - tn[:, [0]]  # Normalize delays for each BS
        tn = tn[:, np.newaxis, :]  # Expand dims to match shape (num_bs, num_ues, N)

    elif delay_method == "distance":
        if dist_map is None:
            raise ValueError("dist_map must be provided for distance-based delay calculation.")
            # Expand dist_map to include N multipaths per BS-UE pair

            # Generate multipath distances with variability based on sigma_DS
            num_bs, num_ues = dist_map.shape
            multipath_offsets = np.random.randn(num_bs, num_ues, N) * sigma_DS[:, np.newaxis, np.newaxis]
            multipath_distances = dist_map[:, :, np.newaxis] + multipath_offsets
            multipath_distances = np.maximum(multipath_distances, 0)  # No negative distances
            tn = multipath_distances / c  # Convert distances to delays

    elif delay_method == "both":
        if dist_map is None:
            raise ValueError("dist_map must be provided for combined delay calculation.")
        # Calculate both delays and combine them
        # 3GPP delays
        num_bs = sigma_DS.shape[0]
        Zn = np.random.rand(num_bs, N)
        tn_3gpp = -r_DS * sigma_DS[:, np.newaxis] * np.log(Zn)
        tn_3gpp = np.sort(tn_3gpp, axis=1)
        tn_3gpp = tn_3gpp - tn_3gpp[:, [0]]
        tn_3gpp = tn_3gpp[:, np.newaxis, :]  # Expand dims to match shape (num_bs, num_ues, N)

        # Distance-based delays
        num_bs, num_ues = dist_map.shape
        multipath_offsets = np.random.randn(num_bs, num_ues, N) * sigma_DS[:, np.newaxis, np.newaxis]
        multipath_distances = dist_map[:, :, np.newaxis] + multipath_offsets
        multipath_distances = np.maximum(multipath_distances, 0)
        tn_distance = multipath_distances / c

        # Combine 3GPP and distance-based delays
        tn = tn_3gpp + tn_distance


    else:
        raise ValueError(f"Unknown method '{delay_method}'. Choose '3GPP', 'distance', or 'both'.")

    return tn