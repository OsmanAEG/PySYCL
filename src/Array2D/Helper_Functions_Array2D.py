import sys
sys.path.insert(0, '../build')
import unittest
import random
import numpy as np

from pysycl import device
from pysycl import array2D

###########################################################
# Array2D Shared Specific Helper Functions
###########################################################
def array2D_shared_init(rows, cols, Q):
  A = array2D.array2D_shared(rows, cols, Q)
  return A

def create_4_arrays_shared(M, N, Q):
  A_arr2D = array2D_shared_init(M, N, Q)
  B_arr2D = array2D_shared_init(M, N, Q)

  A_np = np.zeros((M, N))
  B_np = np.zeros((M, N))

  return A_arr2D, B_arr2D, A_np, B_np


###########################################################
# Array2D Specific Helper Functions
###########################################################
def element_fill(A_arr2D, B_arr2D, A_np, B_np, M, N):
  for i in range(M):
    for j in range(N):
      rand_1 = random.uniform(0.0, 100.0)
      rand_2 = random.uniform(0.0, 100.0)

      A_arr2D.set_value(i, j, rand_1)
      B_arr2D.set_value(i, j, rand_2)

      A_np[i][j] = rand_1
      B_np[i][j] = rand_2

def check_elements_equal(get_C_arr2D, C_np, M, N):
  for i in range(M):
    for j in range(N):
      assert abs(get_C_arr2D[i][j] - C_np[i][j]) < 1e-2