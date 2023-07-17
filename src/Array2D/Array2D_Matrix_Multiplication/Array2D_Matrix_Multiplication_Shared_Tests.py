import sys
sys.path.append('../build')
sys.path.append('../src/Array2D/Helper_Functions_Array2D')
import unittest
import math
import numpy as np

from pysycl import device
from pysycl import array2D

from Helper_Functions_Array2D import *
from Helper_Functions_Array2D_Shared import *

class TestArray2D_Matrix_Multiplication_Shared(unittest.TestCase):
  """
  Test Matrix Multiplication of Array2D Shared
  """

  def test_matrix_multiplication(self):
    print("\nTESTING THE MATRIX MULTIPLICATION OF TWO ARRAY2D SHARED OBJECTS")
    """
    Test matrix multiplication for Array2D_Shared Objects
    :returns: None
    :raises: AssertionError
    """

    ###########################################################
    # Device for testing
    Q = device.device_object(0, 0)

    ###########################################################
    # Tests for matrix multiplication
    try:
      M = 5
      N = 3
      P = 4
      A_arr2D, A_np = create_test_arrays(M, N, Q)
      B_arr2D, B_np = create_test_arrays(N, P, Q)
      C_arr2D_obj = array2D.matmul(A_arr2D, B_arr2D)
      C_np = np.matmul(A_np, B_np)
      Q.wait()
      C_arr2D = C_arr2D_obj.get_data()
      check_elements_equal(C_arr2D, C_np, M, P)
    except:
      raise AssertionError("Matrix Multiplication Operation Failed for Array2D_Shared Objects!")

    try:
      M = 6
      N = 6
      P = 6
      A_arr2D, A_np = create_test_arrays(M, N, Q)
      B_arr2D, B_np = create_test_arrays(N, P, Q)
      C_arr2D_obj = array2D.matmul(A_arr2D, B_arr2D)
      C_np = np.matmul(A_np, B_np)
      Q.wait()
      C_arr2D = C_arr2D_obj.get_data()
      check_elements_equal(C_arr2D, C_np, M, P)
    except:
      raise AssertionError("Matrix Multiplication Operation Failed for Array2D_Shared Objects!")

    try:
      M = 5
      N = 3
      P = 4
      A_arr2D, A_np = create_test_arrays(M, N, Q)
      B_arr2D, B_np = create_test_arrays(N, P, Q)
      C_arr2D_obj = array2D.matmul(A_arr2D, B_arr2D)
      C_np = np.matmul(A_np, B_np)
      Q.wait()
      C_arr2D = C_arr2D_obj.get_data()
      check_elements_equal(C_arr2D, C_np, M, P)
    except:
      raise AssertionError("Matrix Multiplication Operation Failed for Array2D_Shared Objects!")

if __name__ == '__main__':
  unittest.main()