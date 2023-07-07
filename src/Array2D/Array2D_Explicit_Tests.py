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
    # Test Array2D_Explicit init (return value tests)
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
    # Test Array2D_Explicit init (return value tests)
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
    try:
      Q = device.device_object(0, 0)
      Q_name = Q.device_name()
      A = self.array2D_explicit_init(100, 100, Q)
      self.assertEqual(A.get_device().device_name(), Q.device_name())
    except:
      raise AssertionError("Array2D_Explicit failed to return device correctly.")

  def test_array2D_explicit_get_value(self):

if __name__ == '__main__':
  unittest.main()