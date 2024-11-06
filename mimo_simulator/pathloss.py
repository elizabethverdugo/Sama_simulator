import numpy as np

def calculate_path_loss(h_bs, h_ms, d, f_c, C):
    """
    Calculate the path loss (PL) in dB based on the given parameters.

    Args:
    - h_bs (float): Height of the base station (in meters).
    - h_ms (float): Height of the mobile station (in meters).
    - d (float): Distance between the base station and the mobile station (in meters).
    - f_c (float): Carrier frequency (in MHz).
    - C: Constant factor.

    Returns:
    - float: Path loss (PL) in dB.
    """
    PL = (44.9 - 6.55 * np.log10(h_bs)) * np.log10(d / 1000) + 45.5 + \
         (35.46 - 1.1 * h_ms) * np.log10(f_c) - 13.82 * np.log10(h_bs) + \
         0.7 * h_ms + C

    return PL