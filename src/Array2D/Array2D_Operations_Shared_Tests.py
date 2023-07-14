import sys
sys.path.insert(0, '../build')
import unittest
import math
import numpy as np

from pysycl import device
from pysycl import array2D

from Helper_Functions_Array2D import *
from Helper_Functions_Array2D_Shared import *

class TestArray2D_Operations_Shared(unittest.TestCase):
  """
  Test Array2D Operations Shared
  """

  def test_add(self):
    print("\nTESTING THE ADDITION OF TWO ARRAY2D SHARED OBJECTS")
    """
    Test add for Array2D_Shared Objects
    :returns: None
    :raises: AssertionError
    """

    ###########################################################
    # Device for testing
    Q = device.device_object(0, 0)

    ###########################################################
    # Tests for add
    try:
      M = 5
      N = 3
      A_arr2D, B_arr2D, A_np, B_np = create_test_arrays_element(M, N, Q)
      C_arr2D_obj = array2D.add(A_arr2D, B_arr2D)
      C_np = A_np + B_np
      Q.wait()
      C_arr2D = C_arr2D_obj.get_data()
      check_elements_equal(C_arr2D, C_np, M, N)
    except:
      raise AssertionError("Addition Operation Failed for Array2D_Shared Objects!")

    try:
      M = 6
      N = 6
      A_arr2D, B_arr2D, A_np, B_np = create_test_arrays_element(M, N, Q)
      C_arr2D_obj = array2D.add(A_arr2D, B_arr2D)
      C_np = A_np + B_np
      Q.wait()
      C_arr2D = C_arr2D_obj.get_data()
      check_elements_equal(C_arr2D, C_np, M, N)
    except:
      raise AssertionError("Addition Operation Failed for Array2D_Shared Objects!")

  def test_sub(self):
    print("\nTESTING THE SUBTRACTION OF TWO ARRAY2D SHARED OBJECTS")
    """
    Test sub for Array2D_Shared Objects
    :returns: None
    :raises: AssertionError
    """

    ###########################################################
    # Device for testing
    Q = device.device_object(0, 0)

    ###########################################################
    # Tests for sub
    try:
      M = 5
      N = 3
      A_arr2D, B_arr2D, A_np, B_np = create_test_arrays_element(M, N, Q)
      C_arr2D_obj = array2D.sub(A_arr2D, B_arr2D)
      C_np = A_np - B_np
      Q.wait()
      C_arr2D = C_arr2D_obj.get_data()
      check_elements_equal(C_arr2D, C_np, M, N)
    except:
      raise AssertionError("Subtraction Operation Failed for Array2D_Shared Objects!")

    try:
      M = 6
      N = 6
      A_arr2D, B_arr2D, A_np, B_np = create_test_arrays_element(M, N, Q)
      C_arr2D_obj = array2D.sub(A_arr2D, B_arr2D)
      C_np = A_np - B_np
      Q.wait()
      C_arr2D = C_arr2D_obj.get_data()
      check_elements_equal(C_arr2D, C_np, M, N)
    except:
      raise AssertionError("Subtraction Operation Failed for Array2D_Shared Objects!")

  def test_mul(self):
    print("\nTESTING THE MULTIPLICATION OF TWO ARRAY2D SHARED OBJECTS")
    """
    Test mul for Array2D_Shared Objects
    :returns: None
    :raises: AssertionError
    """

    ###########################################################
    # Device for testing
    Q = device.device_object(0, 0)

    ###########################################################
    # Tests for mul
    try:
      M = 5
      N = 3
      A_arr2D, B_arr2D, A_np, B_np = create_test_arrays_element(M, N, Q)
      C_arr2D_obj = array2D.mul(A_arr2D, B_arr2D)
      C_np = A_np * B_np
      Q.wait()
      C_arr2D = C_arr2D_obj.get_data()
      check_elements_equal(C_arr2D, C_np, M, N)
    except:
      raise AssertionError("Multiplication Operation Failed for Array2D_Shared Objects!")

    try:
      M = 6
      N = 6
      A_arr2D, B_arr2D, A_np, B_np = create_test_arrays_element(M, N, Q)
      C_arr2D_obj = array2D.mul(A_arr2D, B_arr2D)
      C_np = A_np * B_np
      Q.wait()
      C_arr2D = C_arr2D_obj.get_data()
      check_elements_equal(C_arr2D, C_np, M, N)
    except:
      raise AssertionError("Multiplication Operation Failed for Array2D_Shared Objects!")

  def test_div(self):
    print("\nTESTING THE DIVISION OF TWO ARRAY2D SHARED OBJECTS")
    """
    Test sub for Array2D_Shared Objects
    :returns: None
    :raises: AssertionError
    """

    ###########################################################
    # Device for testing
    Q = device.device_object(0, 0)

    ###########################################################
    # Tests for div
    try:
      M = 5
      N = 3
      A_arr2D, B_arr2D, A_np, B_np = create_test_arrays_element(M, N, Q)
      C_arr2D_obj = array2D.div(A_arr2D, B_arr2D)
      C_np = A_np / B_np
      Q.wait()
      C_arr2D = C_arr2D_obj.get_data()
      check_elements_equal(C_arr2D, C_np, M, N)
    except:
      raise AssertionError("Division Operation Failed for Array2D_Shared Objects!")

    try:
      M = 6
      N = 6
      A_arr2D, B_arr2D, A_np, B_np = create_test_arrays_element(M, N, Q)
      C_arr2D_obj = array2D.div(A_arr2D, B_arr2D)
      C_np = A_np / B_np
      Q.wait()
      C_arr2D = C_arr2D_obj.get_data()
      check_elements_equal(C_arr2D, C_np, M, N)
    except:
      raise AssertionError("Divison Operation Failed for Array2D_Shared Objects!")

if __name__ == '__main__':
  unittest.main()
