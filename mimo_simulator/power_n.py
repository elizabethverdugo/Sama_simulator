import numpy as np

def calculate_power(r_DS, tn, sigma_DS):

    num_bs, num_ues, n_paths = tn.shape  # Extract dimensions from tn
    sigma_DS = sigma_DS[:, np.newaxis, np.newaxis]  # Reshape to (num_bs, 1, 1)

    temp1_num = (1-r_DS)*(tn-tn[:, :, 0:1])     #temporal variable, numerator in (5.3-3)
    temp1_den = r_DS* sigma_DS

    temp1 = np.exp(temp1_num/temp1_den)

    std_linear = 10 ** (3 / 20.0)       #N Gaussian random variables with standard deviation  = 3 dB
    xi_n = np.random.randn(num_bs, num_ues, n_paths) * std_linear    #shadowing randomization effect on the per-path powers

    Pn1 = temp1 * 10 ** (-xi_n/10)

    Pn = Pn1/np.sum(Pn1, axis=2, keepdims=True)        #Normalization!
    #Important: sum of Pn should be one because of the normalization

    return Pn