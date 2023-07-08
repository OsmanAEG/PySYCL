import sys
sys.path.insert(0, '../build')
import unittest
import math

from pysycl import device
from pysycl import array2D

class TestArray2D_Explicit(unittest.TestCase):
  """
  Test Array2D class
  """
  def array2D_explicit_init(self, rows, cols, Q):
    """
    Test Array2D init
    :param rows: The number of rows in the array.
    :param cols: The number of columns in the array.
    :param Q: The initial value of the array.

    :type rows: int
    :type cols: int
    :type Q: float

    :returns: None
    """
    A = array2D.array2D_explicit(rows, cols, Q)
    return A
  def test_array2D_explicit_init(self):
    print("\nTESTING THE ARRAY2D_EXPLICIT CONSTRUCTOR")
    """
    Test Array2D_Explicit init
    :returns: None
    :raises: AssertionError
    """

    ###########################################################
    # Device for testing
    Q = device.device_object(0, 0)

    ###########################################################
    # Test Array2D_Explicit init (constructor tests)
    #   The following tests should pass
    try:
      self.array2D_explicit_init(100, 100, Q)
    except:
      raise AssertionError("Array2D_Explicit init failed to create an Array2D_Explicit.")

    try:
      self.array2D_explicit_init(100, 50, Q)
    except:
      raise AssertionError("Array2D_Explicit init failed to create an Array2D_Explicit.")

    try:
      self.array2D_explicit_init(50, 100, Q)
    except:
      raise AssertionError("Array2D_Explicit init failed to create an Array2D_Explicit.")

    ###########################################################
    # Test Array2D_Explicit init (constructor tests)
    #   The following tests should fail
    with self.assertRaises(TypeError):
      self.array2D_explicit_init(58, 20, 1)

    with self.assertRaises(TypeError):
      self.array2D_explicit_init(340, 10, "Q")

    with self.assertRaises(TypeError):
      self.array2D_explicit_init(8.9, 10, Q)

    with self.assertRaises(TypeError):
      self.array2D_explicit_init(8, 9.87, Q)

    with self.assertRaises(TypeError):
      self.array2D_explicit_init(99.1, 10.9, Q)

    with self.assertRaises(RuntimeError):
      self.array2D_explicit_init(0, 59, Q)

    with self.assertRaises(RuntimeError):
      self.array2D_explicit_init(68, 0, Q)

    with self.assertRaises(RuntimeError):
      self.array2D_explicit_init(0, 0, Q)

    with self.assertRaises(RuntimeError):
      self.array2D_explicit_init(38, -124, Q)

    with self.assertRaises(RuntimeError):
      self.array2D_explicit_init(-100, 80, Q)

    with self.assertRaises(RuntimeError):
      self.array2D_explicit_init(-158, -94, Q)

  def test_array2D_explicit_get_rows(self):
    print("\nTESTING THE ARRAY2D_EXPLICIT GET ROWS")
    """
    Test Array2D_Explicit get_rows
    :returns: None
    :raises: AssertionError
    """

    ###########################################################
    # Device for testing
    Q = device.device_object(0, 0)

    ###########################################################
    # Test Array2D_Explicit init (return rows tests)
    #   The following tests should pass
    try:
      A = self.array2D_explicit_init(100, 100, Q)
      self.assertEqual(A.number_of_rows(), 100)
    except:
      raise AssertionError("Array2D_Explicit failed to return rows correctly.")
    try:
      A = self.array2D_explicit_init(50, 100, Q)
      self.assertEqual(A.number_of_rows(), 50)
    except:
      raise AssertionError("Array2D_Explicit failed to return rows correctly.")
    try:
      A = self.array2D_explicit_init(567, 23, Q)
      self.assertEqual(A.number_of_rows(), 567)
    except:
      raise AssertionError("Array2D_Explicit failed to return rows correctly.")

  def test_array2D_explicit_get_cols(self):
    print("\nTESTING THE ARRAY2D_EXPLICIT GET COLS")
    """
    Test Array2D_Explicit get_cols
    :returns: None
    :raises: AssertionError
    """

    ###########################################################
    # Device for testing
    Q = device.device_object(0, 0)

    ###########################################################
    # Test Array2D_Explicit init (return cols tests)
    #   The following tests should pass
    try:
      A = self.array2D_explicit_init(100, 100, Q)
      self.assertEqual(A.number_of_cols(), 100)
    except:
      raise AssertionError("Array2D_Explicit failed to return columns correctly.")
    try:
      A = self.array2D_explicit_init(50, 100, Q)
      self.assertEqual(A.number_of_cols(), 100)
    except:
      raise AssertionError("Array2D_Explicit failed to return columns correctly.")
    try:
      A = self.array2D_explicit_init(567, 23, Q)
      self.assertEqual(A.number_of_cols(), 23)
    except:
      raise AssertionError("Array2D_Explicit failed to return columns correctly.")

  def test_array2D_explicit_get_device(self):
    print("\nTESTING THE ARRAY2D_EXPLICIT GET DEVICE")
    """
    Test Array2D_Explicit get_device
    :returns: None
    :raises: AssertionError
    """

    ###########################################################
    # Device for testing
    Q = device.device_object(0, 0)
    Q_name = Q.device_name()

    ###########################################################
    # Test Array2D_Explicit i (return device tests)
    #   The following tests should pass
    try:
      A = self.array2D_explicit_init(100, 100, Q)
      A_Q_name = A.get_device().device_name()
      self.assertEqual(A_Q_name, Q_name)
    except:
      raise AssertionError("Array2D_Explicit failed to return device correctly.")

  def test_array2D_explicit_data_movement(self):
    print("\nTESTING THE ARRAY2D_EXPLICIT DATA MOVEMENT")
    """
    Test Array2D_Explicit data movement
    :returns: None
    :raises: AssertionError
    """

    ###########################################################
    # Device for testing
    Q = device.device_object(0, 0)

    ###########################################################
    # Test Array2D_Explicit data movement
    #   The following tests should pass
    try:
      A = self.array2D_explicit_init(100, 100, Q)
      A.copy_host_to_device()
    except:
      raise AssertionError("Array2D_Explicit failed to copy host to device.")

    try:
      A = self.array2D_explicit_init(100, 100, Q)
      A.copy_device_to_host()
    except:
      raise AssertionError("Array2D_Explicit failed to copy device to host.")

  def test_array2D_explicit_set_return(self):
    print("\nTESTING THE ARRAY2D_EXPLICIT SET AND GET")
    """
    Test Array2D_Explicit set and get
    :returns: None
    :raises: AssertionError
    """

    ###########################################################
    # Device for testing
    Q = device.device_object(0, 0)

    ###########################################################
    # Test Array2D_Explicit set and get
    #   The following tests should pass
    try:
      A = self.array2D_explicit_init(100, 100, Q)
      A.set_host_value(10, 10, 1)
      A_py_ij = A.get_host_value(10, 10)
      self.assertEqual(A_py_ij, 1)
    except:
      raise AssertionError("Array2D_Explicit failed to set and get correctly.")

    try:
      A = self.array2D_explicit_init(78, 34, Q)
      A.set_host_value(12, 19, -1.83)
      A_py_ij = A.get_host_value(12, 19)
      self.assertAlmostEqual(A_py_ij, -1.83)
    except:
      raise AssertionError("Array2D_Explicit failed to set and get correctly.")

  def test_array2D_explicit_set_return(self):
    print("\nTESTING THE ARRAY2D_EXPLICIT SET AND RETURN")
    """
    Test Array2D_Explicit set and return
    :returns: None
    :raises: AssertionError
    """

    ###########################################################
    # Device for testing
    Q = device.device_object(0, 0)

    ###########################################################
    # Test Array2D_Explicit set and return
    #   The following tests should pass
    try:
      A = self.array2D_explicit_init(100, 100, Q)
      A.set_host_value(10, 10, 1)
      A_py = A.get_host_data()
      self.assertEqual(A_py[10][10], 1)
    except:
      raise AssertionError("Array2D_Explicit failed to set and return correctly.")

    try:
      A = self.array2D_explicit_init(78, 34, Q)
      A.set_host_value(12, 19, -1.83)
      A_py = A.get_host_data()
      self.assertAlmostEqual(A_py[12][19], -1.83)
    except:
      raise AssertionError("Array2D_Explicit failed to set and return correctly.")

if __name__ == '__main__':
  unittest.main()