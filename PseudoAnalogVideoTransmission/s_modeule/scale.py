#-*- coding: utf-8 -*-
import numpy as np
from numba import jit

# @jit
def scaling(va, n, P):
    sq_va = va ** 0.5
    sd = []
    for i in range(3):
      sd.append(np.sum(sq_va[:,i]))

    g = np.empty([n**2,3])
    for j in range(3):
      for i in range(len(va)):
        g[i,j] = va[i,j] ** (-1/4) * (P / sd[j]) ** 0.5
    return g