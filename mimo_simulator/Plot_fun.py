from random import uniform



import numpy as np


def ecdf(data):
    d = np.sort(data.ravel())
    y = np.linspace(0, 1, d.size, endpoint=False)
    return d,y

