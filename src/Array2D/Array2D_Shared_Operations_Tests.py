import sys
sys.path.insert(0, '../build')
import unittest
import random
import numpy as np
from Helper_Functions_Array2D import *

from pysycl import device
from pysycl import array2D

class TestArray2D_Element_Wise_Operations_Shared(unittest.TestCase):
  """
  Test Array2D Operations
  """

  ###########################################################
  ###########################################################
  ###########################################################
  # Tests for add_el_Array2D (Shared) addition

  def test_add_el_array2D(self):
    print("\nTESTING THE ELEMENT-WISE ADDITION OF TWO ARRAY2D SHARED OBJECTS")
    """
    Test add_Array2D for Array2D_Shared Objects
    :returns: None
    :raises: AssertionError
    """

    ###########################################################
    # Device for testing
    Q = device.device_object(0, 0)

    ###########################################################
    # Tests for add_el_Array2D (Shared) addition

    # These tests should pass
    ###########################################################
    try:
      M = 100
      N = 50

      A_arr2D, B_arr2D, A_np, B_np = create_4_arrays_shared(M, N, Q)

      element_fill(A_arr2D, B_arr2D, A_np, B_np, M, N)

      C_arr2D = array2D.add_el_array2D(A_arr2D, B_arr2D)
      C_np = A_np + B_np

      Q.wait()
      get_C_arr2D = C_arr2D.get_data()
      check_elements_equal(get_C_arr2D, C_np, M, N)
    except:
      raise AssertionError("add_el_Array2D failed to add two Array2D_Shared objects.")

    ###########################################################
    try:
      M = 393
      N = 528

      A_arr2D, B_arr2D, A_np, B_np = create_4_arrays_shared(M, N, Q)

      element_fill(A_arr2D, B_arr2D, A_np, B_np, M, N)

      C_arr2D = array2D.add_el_array2D(A_arr2D, B_arr2D)
      C_np = A_np + B_np

      Q.wait()
      get_C_arr2D = C_arr2D.get_data()
      check_elements_equal(get_C_arr2D, C_np, M, N)
    except:
      raise AssertionError("add_el_Array2D failed to add two Array2D_Shared objects.")

    ###########################################################
    try:
      M = 25
      N = 25

      A_arr2D, B_arr2D, A_np, B_np = create_4_arrays_shared(M, N, Q)

      element_fill(A_arr2D, B_arr2D, A_np, B_np, M, N)

      C_arr2D = array2D.add_el_array2D(A_arr2D, B_arr2D)
      C_np = A_np + B_np

      Q.wait()
      get_C_arr2D = C_arr2D.get_data()
      check_elements_equal(get_C_arr2D, C_np, M, N)
    except:
      raise AssertionError("add_el_Array2D failed to add two Array2D_Shared objects.")

    # These tests should fail
    ###########################################################
    with self.assertRaises(RuntimeError):
      A_arr2D = array2D_shared_init(500, 200, Q)
      B_arr2D = array2D_shared_init(500, 201, Q)
      C_arr2D = array2D.add_el_array2D(A_arr2D, B_arr2D)

    with self.assertRaises(RuntimeError):
      A_arr2D = array2D_shared_init(400, 200, Q)
      B_arr2D = array2D_shared_init(500, 200, Q)
      C_arr2D = array2D.add_el_array2D(A_arr2D, B_arr2D)

  ###########################################################
  ###########################################################
  ###########################################################
  # Tests for subtract_el_Array2D (Shared) subtraction

  def test_subtract_el_array2D(self):
    print("\nTESTING THE ELEMENT-WISE SUBTRACTION OF TWO ARRAY2D SHARED OBJECTS")
    """
    Test subtract_el_Array2D for Array2D_Shared Objects
    :returns: None
    :raises: AssertionError
    """

    ###########################################################
    # Device for testing
    Q = device.device_object(0, 0)

    ###########################################################
    # Tests for subtract_el_Array2D (Shared) subtraction

    # These tests should pass
    ###########################################################
    try:
      M = 100
      N = 50

      A_arr2D, B_arr2D, A_np, B_np = create_4_arrays_shared(M, N, Q)

      element_fill(A_arr2D, B_arr2D, A_np, B_np, M, N)

      C_arr2D = array2D.subtract_el_array2D(A_arr2D, B_arr2D)
      C_np = A_np - B_np

      Q.wait()
      get_C_arr2D = C_arr2D.get_data()
      check_elements_equal(get_C_arr2D, C_np, M, N)
    except:
      raise AssertionError("subtract_el_Array2D failed to subtract two Array2D_Shared objects.")

    ###########################################################
    try:
      M = 393
      N = 528

      A_arr2D, B_arr2D, A_np, B_np = create_4_arrays_shared(M, N, Q)

      element_fill(A_arr2D, B_arr2D, A_np, B_np, M, N)

      C_arr2D = array2D.subtract_el_array2D(A_arr2D, B_arr2D)
      C_np = A_np - B_np

      Q.wait()
      get_C_arr2D = C_arr2D.get_data()
      check_elements_equal(get_C_arr2D, C_np, M, N)
    except:
      raise AssertionError("subtract_el_Array2D failed to subtract two Array2D_Shared objects.")

    ###########################################################
    try:
      M = 25
      N = 25

      A_arr2D, B_arr2D, A_np, B_np = create_4_arrays_shared(M, N, Q)

      element_fill(A_arr2D, B_arr2D, A_np, B_np, M, N)

      C_arr2D = array2D.subtract_el_array2D(A_arr2D, B_arr2D)
      C_np = A_np - B_np

      Q.wait()
      get_C_arr2D = C_arr2D.get_data()
      check_elements_equal(get_C_arr2D, C_np, M, N)
    except:
      raise AssertionError("subtract_el_Array2D failed to subtract two Array2D_Shared objects.")

    # These tests should fail
    ###########################################################
    with self.assertRaises(RuntimeError):
      A_arr2D = array2D_shared_init(500, 200, Q)
      B_arr2D = array2D_shared_init(500, 201, Q)
      C_arr2D = array2D.subtract_el_array2D(A_arr2D, B_arr2D)

    with self.assertRaises(RuntimeError):
      A_arr2D = array2D_shared_init(400, 200, Q)
      B_arr2D = array2D_shared_init(500, 200, Q)
      C_arr2D = array2D.subtract_el_array2D(A_arr2D, B_arr2D)

  ###########################################################
  ###########################################################
  ###########################################################
  # Tests for subtract_el_Array2D (Shared) addition

  def test_multiply_el_array2D(self):
    print("\nTESTING THE ELEMENT-WISE MULTIPLICATION OF TWO ARRAY2D SHARED OBJECTS")
    """
    Test multiply_el_Array2D for Array2D_Shared Objects
    :returns: None
    :raises: AssertionError
    """

    ###########################################################
    # Device for testing
    Q = device.device_object(0, 0)

    ###########################################################
    # Tests for multiply_el_Array2D (Shared) multiplication

    # These tests should pass
    ###########################################################
    try:
      M = 100
      N = 50

      A_arr2D, B_arr2D, A_np, B_np = create_4_arrays_shared(M, N, Q)

      element_fill(A_arr2D, B_arr2D, A_np, B_np, M, N)

      C_arr2D = array2D.multiply_el_array2D(A_arr2D, B_arr2D)
      C_np = A_np * B_np

      Q.wait()

      get_C_arr2D = C_arr2D.get_data()
      check_elements_equal(get_C_arr2D, C_np, M, N)
    except:
      raise AssertionError("multiply_el_Array2D failed to multiply two Array2D_Shared objects.")

    ###########################################################
    try:
      M = 393
      N = 528

      A_arr2D, B_arr2D, A_np, B_np = create_4_arrays_shared(M, N, Q)

      element_fill(A_arr2D, B_arr2D, A_np, B_np, M, N)

      C_arr2D = array2D.multiply_el_array2D(A_arr2D, B_arr2D)
      C_np = A_np * B_np

      Q.wait()
      get_C_arr2D = C_arr2D.get_data()
      check_elements_equal(get_C_arr2D, C_np, M, N)
    except:
      raise AssertionError("multiply_el_Array2D failed to multiply two Array2D_Shared objects.")

    ###########################################################
    try:
      M = 25
      N = 25

      A_arr2D, B_arr2D, A_np, B_np = create_4_arrays_shared(M, N, Q)

      element_fill(A_arr2D, B_arr2D, A_np, B_np, M, N)

      C_arr2D = array2D.multiply_el_array2D(A_arr2D, B_arr2D)
      C_np = A_np * B_np

      Q.wait()
      get_C_arr2D = C_arr2D.get_data()
      check_elements_equal(get_C_arr2D, C_np, M, N)
    except:
      raise AssertionError("multiply_el_Array2D failed to multiply two Array2D_Shared objects.")

    # These tests should fail
    ###########################################################
    with self.assertRaises(RuntimeError):
      A_arr2D = array2D_shared_init(500, 200, Q)
      B_arr2D = array2D_shared_init(500, 201, Q)
      C_arr2D = array2D.multiply_el_array2D(A_arr2D, B_arr2D)

    with self.assertRaises(RuntimeError):
      A_arr2D = array2D_shared_init(400, 200, Q)
      B_arr2D = array2D_shared_init(500, 200, Q)
      C_arr2D = array2D.multiply_el_array2D(A_arr2D, B_arr2D)

  def test_divide_el_array2D(self):
    print("\nTESTING THE ELEMENT-WISE DIVISION OF TWO ARRAY2D SHARED OBJECTS")
    """
    Test multiply_el_Array2D for Array2D_Shared Objects
    :returns: None
    :raises: AssertionError
    """

    ###########################################################
    # Device for testing
    Q = device.device_object(0, 0)

    ###########################################################
    # Tests for divide_el_Array2D (Shared) division

    # These tests should pass
    ###########################################################
    try:
      M = 5
      N = 5

      A_arr2D, B_arr2D, A_np, B_np = create_4_arrays_shared(M, N, Q)

      element_fill(A_arr2D, B_arr2D, A_np, B_np, M, N)

      C_arr2D = array2D.divide_el_array2D(A_arr2D, B_arr2D)
      C_np = A_np / B_np

      Q.wait()

      get_C_arr2D = C_arr2D.get_data()
      check_elements_equal(get_C_arr2D, C_np, M, N)
    except:
      raise AssertionError("divide_el_Array2D failed to divide two Array2D_Shared objects.")

    ###########################################################
    try:
      M = 393
      N = 528

      A_arr2D, B_arr2D, A_np, B_np = create_4_arrays_shared(M, N, Q)

      element_fill(A_arr2D, B_arr2D, A_np, B_np, M, N)

      C_arr2D = array2D.divide_el_array2D(A_arr2D, B_arr2D)
      C_np = A_np / B_np

      Q.wait()
      get_C_arr2D = C_arr2D.get_data()
      check_elements_equal(get_C_arr2D, C_np, M, N)
    except:
      raise AssertionError("divide_el_Array2D failed to divide two Array2D_Shared objects.")

    ###########################################################
    try:
      M = 25
      N = 25

      A_arr2D, B_arr2D, A_np, B_np = create_4_arrays_shared(M, N, Q)

      element_fill(A_arr2D, B_arr2D, A_np, B_np, M, N)

      C_arr2D = array2D.divide_el_array2D(A_arr2D, B_arr2D)
      C_np = A_np / B_np

      Q.wait()
      get_C_arr2D = C_arr2D.get_data()
      check_elements_equal(get_C_arr2D, C_np, M, N)
    except:
      raise AssertionError("divide_el_Array2D failed to divide two Array2D_Shared objects.")

    # These tests should fail
    ###########################################################
    with self.assertRaises(RuntimeError):
      A_arr2D = array2D_shared_init(500, 200, Q)
      B_arr2D = array2D_shared_init(500, 201, Q)
      C_arr2D = array2D.divide_el_array2D(A_arr2D, B_arr2D)

    with self.assertRaises(RuntimeError):
      A_arr2D = array2D_shared_init(400, 200, Q)
      B_arr2D = array2D_shared_init(500, 200, Q)
      C_arr2D = array2D.divide_el_array2D(A_arr2D, B_arr2D)

if __name__ == '__main__':
  unittest.main()