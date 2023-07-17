import sys
sys.path.insert(0, '../build')
import unittest
import random
import numpy as np

from pysycl import device
from pysycl import array2D

###########################################################
# Helper Functions for testing Array2D Shared
###########################################################
def array2D_init(rows, cols, Q):
  A = array2D.array2D_shared(rows, cols, Q)
  return A

def create_test_arrays_element(M, N, Q):
  A_arr2D = array2D_init(M, N, Q)
  B_arr2D = array2D_init(M, N, Q)

  A_np = np.zeros((M, N))
  B_np = np.zeros((M, N))

  for i in range(M):
    for j in range(N):
      rand_1 = random.uniform(0.0, 100.0)
      rand_2 = random.uniform(0.0, 100.0)
      A_arr2D.set_value(i, j, rand_1)
      B_arr2D.set_value(i, j, rand_2)
      A_np[i][j] = rand_1
      B_np[i][j] = rand_2

  return A_arr2D, B_arr2D, A_np, B_np

def create_test_arrays(M, N, Q):
  A_arr2D = array2D_init(M, N, Q)
  A_np = np.zeros((M, N))

  for i in range(M):
    for j in range(N):
      rand = random.uniform(0.0, 100.0)
      A_arr2D.set_value(i, j, rand)
      A_np[i][j] = rand

  return A_arr2D, A_np,

def create_test_arrays_matmul(M, N, P, Q):
  A_arr2D = array2D_init(M, N, Q)
  B_arr2D = array2D_init(N, P, Q)

  A_np = np.zeros((M, N))
  B_np = np.zeros((N, P))

  for i in range(M):
    for j in range(N):
      rand = random.uniform(0.0, 100.0)
      A_arr2D.set_value(i, j, rand)
      A_np[i][j] = rand

  for i in range(N):
    for j in range(P):
      rand = random.uniform(0.0, 100.0)
      B_arr2D.set_value(i, j, rand)
      B_np[i][j] = rand

  return A_arr2D, B_arr2D, A_np, B_np
