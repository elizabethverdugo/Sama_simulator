import numpy as np

def delay(N, r_DS, sigma_DS):


    Zn = np.random.rand(N,1)
    tn = -r_DS * sigma_DS[:, np.newaxis] * np.log(Zn)
    tn = np.sort(tn, axis=0)
    tn = tn - tn[0]


    return tn