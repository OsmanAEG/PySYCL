import sys
sys.path.insert(0, '../build')
import unittest
import random
import numpy as np

from pysycl import device
from pysycl import array2D

###########################################################
# Helper Functions for testing Array2D
###########################################################

def check_elements_equal(C_arr2D, C_np, M, N):
  for i in range(M):
    for j in range(N):
      assert abs(C_arr2D[i][j] - C_np[i][j]) < 1e-2