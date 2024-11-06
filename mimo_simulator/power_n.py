import numpy as np

def calculate_power(r_DS, tn, sigma_DS):

    temp1_num = (1-r_DS)*(tn-tn[0])     #temporal variable, numerator in (5.3-3)
    temp1_den = r_DS* sigma_DS

    temp1 = np.exp(temp1_num/temp1_den[:, np.newaxis])

    std_linear = 10 ** (3 / 20.0)       #N Gaussian random variables with standard deviation  = 3 dB
    xi_n = np.random.randn(len(tn),1) * std_linear    #shadowing randomization effect on the per-path powers

    Pn1 = temp1 * 10 ** (-xi_n/10)

    Pn = Pn1/np.sum(Pn1)        #Normalization!
    #Important: sum of Pn should be one because of the normalization

    return Pn