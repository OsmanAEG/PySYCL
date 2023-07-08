import sys
sys.path.insert(0, '../build')
import unittest
import math

from pysycl import device
from pysycl import array2D

class TestArray2D_Shared(unittest.TestCase):
  """
  Test Array2D class
  """
  def array2D_shared_init(self, rows, cols, Q):
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
    A = array2D.array2D_shared(rows, cols, Q)
    return A
  def test_array2D_shared_init(self):
    print("\nTESTING THE ARRAY2D_Shared CONSTRUCTOR")
    """
    Test Array2D_Shared  init
    :returns: None
    :raises: AssertionError
    """

    ###########################################################
    # Device for testing
    Q = device.device_object(0, 0)

    ###########################################################
    # Test Array2D_Shared init (constructor tests)
    #   The following tests should pass
    try:
      self.array2D_shared_init(100, 100, Q)
    except:
      raise AssertionError("Array2D_Shared init failed to create an Array2D_Shared.")

    try:
      self.array2D_shared_init(100, 50, Q)
    except:
      raise AssertionError("Array2D_Shared init failed to create an Array2D_Shared.")

    try:
      self.array2D_shared_init(50, 100, Q)
    except:
      raise AssertionError("Array2D_Shared init failed to create an Array2D_Shared.")

    ###########################################################
    # Test Array2D_Shared init (constructor tests)
    #   The following tests should fail
    with self.assertRaises(TypeError):
      self.array2D_shared_init(58, 20, 1)

    with self.assertRaises(TypeError):
      self.array2D_shared_init(340, 10, "Q")

    with self.assertRaises(TypeError):
      self.array2D_shared_init(8.9, 10, Q)

    with self.assertRaises(TypeError):
      self.array2D_shared_init(8, 9.87, Q)

    with self.assertRaises(TypeError):
      self.array2D_shared_init(99.1, 10.9, Q)

    with self.assertRaises(RuntimeError):
      self.array2D_shared_init(0, 59, Q)

    with self.assertRaises(RuntimeError):
      self.array2D_shared_init(68, 0, Q)

    with self.assertRaises(RuntimeError):
      self.array2D_shared_init(0, 0, Q)

    with self.assertRaises(RuntimeError):
      self.array2D_shared_init(38, -124, Q)

    with self.assertRaises(RuntimeError):
      self.array2D_shared_init(-100, 80, Q)

    with self.assertRaises(RuntimeError):
      self.array2D_shared_init(-158, -94, Q)

  def test_array2D_shared_get_rows(self):
    print("\nTESTING THE ARRAY2D_SHARED GET ROWS")
    """
    Test Array2D_Shared get_rows
    :returns: None
    :raises: AssertionError
    """

    ###########################################################
    # Device for testing
    Q = device.device_object(0, 0)

    ###########################################################
    # Test Array2D_Shared init (return rows tests)
    #   The following tests should pass
    try:
      A = self.array2D_shared_init(100, 100, Q)
      self.assertEqual(A.number_of_rows(), 100)
    except:
      raise AssertionError("Array2D_Shared failed to return rows correctly.")
    try:
      A = self.array2D_shared_init(50, 100, Q)
      self.assertEqual(A.number_of_rows(), 50)
    except:
      raise AssertionError("Array2D_Shared failed to return rows correctly.")
    try:
      A = self.array2D_shared_init(567, 23, Q)
      self.assertEqual(A.number_of_rows(), 567)
    except:
      raise AssertionError("Array2D_Shared failed to return rows correctly.")

  def test_array2D_shared_get_cols(self):
    print("\nTESTING THE ARRAY2D_SHARED GET COLS")
    """
    Test Array2D_Shared get_cols
    :returns: None
    :raises: AssertionError
    """

    ###########################################################
    # Device for testing
    Q = device.device_object(0, 0)

    ###########################################################
    # Test Array2D_Shared init (return cols tests)
    #   The following tests should pass
    try:
      A = self.array2D_shared_init(100, 100, Q)
      self.assertEqual(A.number_of_cols(), 100)
    except:
      raise AssertionError("Array2D_Shared failed to return columns correctly.")
    try:
      A = self.array2D_shared_init(50, 100, Q)
      self.assertEqual(A.number_of_cols(), 100)
    except:
      raise AssertionError("Array2D_Shared failed to return columns correctly.")
    try:
      A = self.array2D_shared_init(567, 23, Q)
      self.assertEqual(A.number_of_cols(), 23)
    except:
      raise AssertionError("Array2D_Shared failed to return columns correctly.")

  def test_array2D_shared_get_device(self):
    print("\nTESTING THE ARRAY2D_SHARED GET DEVICE")
    """
    Test Array2D_Shared get_device
    :returns: None
    :raises: AssertionError
    """

    ###########################################################
    # Device for testing
    Q = device.device_object(0, 0)
    Q_name = Q.device_name()

    ###########################################################
    # Test Array2D_Shared i (return device tests)
    #   The following tests should pass
    try:
      A = self.array2D_shared_init(100, 100, Q)
      A_Q_name = A.get_device().device_name()
      self.assertEqual(A_Q_name, Q_name)
    except:
      raise AssertionError("Array2D_Shared failed to return device correctly.")

  def test_array2D_shared_set_get(self):
    print("\nTESTING THE ARRAY2D_SHARED SET AND GET")
    """
    Test Array2D_Shared set and get
    :returns: None
    :raises: AssertionError
    """

    ###########################################################
    # Device for testing
    Q = device.device_object(0, 0)

    ###########################################################
    # Test Array2D_Shared set and return
    #   The following tests should pass
    try:
      A = self.array2D_shared_init(100, 100, Q)
      A.set_value(10, 10, 1)
      A_py_ij = A.get_value(10, 10)
      self.assertEqual(A_py_ij, 1)
    except:
      raise AssertionError("Array2D_Shared failed to set and get correctly.")

    try:
      A = self.array2D_shared_init(78, 34, Q)
      A.set_value(12, 19, -1.83)
      A_py_ij = A.get_value(12, 19)
      self.assertAlmostEqual(A_py_ij, -1.83)
    except:
      raise AssertionError("Array2D_Shared failed to set and get correctly.")

  def test_array2D_shared_set_return(self):
    print("\nTESTING THE ARRAY2D_SHARED SET AND RETURN")
    """
    Test Array2D_Shared set and return
    :returns: None
    :raises: AssertionError
    """

    ###########################################################
    # Device for testing
    Q = device.device_object(0, 0)

    ###########################################################
    # Test Array2D_Shared set and return
    #   The following tests should pass
    try:
      A = self.array2D_shared_init(100, 100, Q)
      A.set_value(10, 10, 1)
      A_py = A.get_data()
      self.assertEqual(A_py[10][10], 1)
    except:
      raise AssertionError("Array2D_Shared failed to set and return correctly.")

    try:
      A = self.array2D_shared_init(78, 34, Q)
      A.set_value(12, 19, -1.83)
      A_py = A.get_data()
      self.assertAlmostEqual(A_py[12][19], -1.83)
    except:
      raise AssertionError("Array2D_Shared failed to set and return correctly.")

if __name__ == '__main__':
  unittest.main()