#-*- coding: utf-8 -*-
import numpy as np
from numba import jit

# @jit
def hadamard(n):
  ha =  np.array([[1,1],[1,-1]])

  def sylvester(n):
      b = np.array(([1,1],[1,-1]))
      for i in range(n-1):
          b = np.kron(ha,b)
      return b
  ha = sylvester(n)

  return ha