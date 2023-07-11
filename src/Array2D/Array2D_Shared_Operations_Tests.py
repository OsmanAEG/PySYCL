import sys
sys.path.insert(0, '../build')
import unittest
import math
import numpy as np

from pysycl import device
from pysycl import array2D

class TestArray2D_Operations(unittest.TestCase):
  """
  Test Array2D Operations
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

  def add_array2D_test_loop(self, arr2D_1, arr2D_2, np_1, np_2)

  def test_add_array2D(self):
    print("\nTESTING THE ADDITION OF TWO ARRAY2D SHARED OBJECTS")
    """
    Test add_Array2D for Array2D_Shared Objects
    :returns: None
    :raises: AssertionError
    """

    ###########################################################
    # Device for testing
    Q = device.device_object(0, 0)

    ###########################################################
    # Tests for add_Array2D (Shared) addition
    try:
      M = 100
      N = 50
      A = self.array2D_shared_init(100, 100, Q)
      B = self.array2D_shared_init(100, 100, Q)
      C = A.add_Array2D(B)