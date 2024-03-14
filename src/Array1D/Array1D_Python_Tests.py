import numpy as np
import unittest
import sys

sys.path.insert(1, '../build/')
import pysycl

############################################
############## ADDITION TESTS ##############
############################################
class TestArray1D_Addition(unittest.TestCase):
  def setUp(self):
    self.tolerance_float  = 1e-7
    self.tolerance_double = 1e-15
    self.device = pysycl.device.get_device(0, 0)

  # ADDITION DOUBLE TYPE TESTS
  def test_vector_addition_double(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.double)
      B_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.double)
      A_pysycl.fill(86.74)
      B_pysycl.fill(12.79)

      C_pysycl = A_pysycl + B_pysycl
      C_pysycl.mem_to_cpu()

      A_np = np.full(N, 86.74, dtype= np.float64)
      B_np = np.full(N, 12.79, dtype= np.float64)
      C_np = A_np + B_np

      for i in range(N):
        self.assertAlmostEqual(C_pysycl[i], C_np[i], delta= self.tolerance_double)

  def test_in_place_vector_addition_double(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.double)
      B_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.double)
      A_pysycl.fill(86.74)
      B_pysycl.fill(12.79)

      A_pysycl += B_pysycl
      A_pysycl.mem_to_cpu()

      A_np = np.full(N, 86.74, dtype= np.float64)
      B_np = np.full(N, 12.79, dtype= np.float64)
      A_np += B_np

      for i in range(N):
        self.assertAlmostEqual(A_pysycl[i], A_np[i], delta= self.tolerance_double)

  # ADDITION FLOAT TYPE TESTS
  def test_vector_addition_float(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.float)
      B_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.float)
      A_pysycl.fill(86.74)
      B_pysycl.fill(12.79)

      C_pysycl = A_pysycl + B_pysycl
      C_pysycl.mem_to_cpu()

      A_np = np.full(N, 86.74, dtype= np.float32)
      B_np = np.full(N, 12.79, dtype= np.float32)
      C_np = A_np + B_np

      for i in range(N):
        self.assertAlmostEqual(C_pysycl[i], C_np[i], delta= self.tolerance_float)

  def test_in_place_vector_addition_float(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.float)
      B_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.float)
      A_pysycl.fill(86.74)
      B_pysycl.fill(12.79)

      A_pysycl += B_pysycl
      A_pysycl.mem_to_cpu()

      A_np = np.full(N, 86.74, dtype= np.float32)
      B_np = np.full(N, 12.79, dtype= np.float32)
      A_np += B_np

      for i in range(N):
        self.assertAlmostEqual(A_pysycl[i], A_np[i], delta= self.tolerance_float)

  # ADDITION INTEGER TYPE TESTS
  def test_vector_addition_int(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.int)
      B_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.int)
      A_pysycl.fill(8)
      B_pysycl.fill(3)

      C_pysycl = A_pysycl + B_pysycl
      C_pysycl.mem_to_cpu()

      A_np = np.full(N, 8, dtype= np.int32)
      B_np = np.full(N, 3, dtype= np.int32)
      C_np = A_np + B_np

      for i in range(N):
        self.assertEqual(C_pysycl[i], C_np[i])

  def test_in_place_vector_addition_int(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.int)
      B_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.int)
      A_pysycl.fill(8)
      B_pysycl.fill(3)

      A_pysycl += B_pysycl
      A_pysycl.mem_to_cpu()

      A_np = np.full(N, 8, dtype= np.int32)
      B_np = np.full(N, 3, dtype= np.int32)
      A_np += B_np

      for i in range(N):
        self.assertEqual(A_pysycl[i], A_np[i])

############################################
########### SUBTRACTION TESTS ##############
############################################
class TestArray1D_Subtraction(unittest.TestCase):
  def setUp(self):
    self.tolerance_float  = 1e-7
    self.tolerance_double = 1e-15
    self.device = pysycl.device.get_device(0, 0)

  # SUBTRACTION DOUBLE TYPE TESTS
  def test_vector_subtraction_double(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.double)
      B_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.double)
      A_pysycl.fill(86.74)
      B_pysycl.fill(12.79)

      C_pysycl = A_pysycl - B_pysycl
      C_pysycl.mem_to_cpu()

      A_np = np.full(N, 86.74, dtype= np.float64)
      B_np = np.full(N, 12.79, dtype= np.float64)
      C_np = A_np - B_np

      for i in range(N):
        self.assertAlmostEqual(C_pysycl[i], C_np[i], delta= self.tolerance_double)

  def test_in_place_vector_subtraction_double(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.double)
      B_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.double)
      A_pysycl.fill(86.74)
      B_pysycl.fill(12.79)

      A_pysycl -= B_pysycl
      A_pysycl.mem_to_cpu()

      A_np = np.full(N, 86.74, dtype= np.float64)
      B_np = np.full(N, 12.79, dtype= np.float64)
      A_np -= B_np

      for i in range(N):
        self.assertAlmostEqual(A_pysycl[i], A_np[i], delta= self.tolerance_double)

  # SUBTRACTION FLOAT TYPE TESTS
  def test_vector_subtraction_float(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.float)
      B_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.float)
      A_pysycl.fill(86.74)
      B_pysycl.fill(12.79)

      C_pysycl = A_pysycl - B_pysycl
      C_pysycl.mem_to_cpu()

      A_np = np.full(N, 86.74, dtype= np.float32)
      B_np = np.full(N, 12.79, dtype= np.float32)
      C_np = A_np - B_np

      for i in range(N):
        self.assertAlmostEqual(C_pysycl[i], C_np[i], delta= self.tolerance_float)

  def test_in_place_vector_subtraction_float(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.float)
      B_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.float)
      A_pysycl.fill(86.74)
      B_pysycl.fill(12.79)

      A_pysycl -= B_pysycl
      A_pysycl.mem_to_cpu()

      A_np = np.full(N, 86.74, dtype= np.float32)
      B_np = np.full(N, 12.79, dtype= np.float32)
      A_np -= B_np

      for i in range(N):
        self.assertAlmostEqual(A_pysycl[i], A_np[i], delta= self.tolerance_float)

  # SUBTRACTION INTEGER TYPE TESTS
  def test_vector_subtraction_int(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.int)
      B_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.int)
      A_pysycl.fill(8)
      B_pysycl.fill(3)

      C_pysycl = A_pysycl - B_pysycl
      C_pysycl.mem_to_cpu()

      A_np = np.full(N, 8, dtype= np.int32)
      B_np = np.full(N, 3, dtype= np.int32)
      C_np = A_np - B_np

      for i in range(N):
        self.assertEqual(C_pysycl[i], C_np[i])

  def test_in_place_vector_subtraction_int(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.int)
      B_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.int)
      A_pysycl.fill(8)
      B_pysycl.fill(3)

      A_pysycl -= B_pysycl
      A_pysycl.mem_to_cpu()

      A_np = np.full(N, 8, dtype= np.int32)
      B_np = np.full(N, 3, dtype= np.int32)
      A_np -= B_np

      for i in range(N):
        self.assertEqual(A_pysycl[i], A_np[i])

############################################
########## MULTIPLICATION TESTS ############
############################################
class TestArray1D_Multiplication(unittest.TestCase):
  def setUp(self):
    self.tolerance_float  = 1e-7
    self.tolerance_double = 1e-15
    self.device = pysycl.device.get_device(0, 0)

  # MULTIPLICATION DOUBLE TYPE TESTS
  def test_vector_multiplication_double(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.double)
      B_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.double)
      A_pysycl.fill(86.74)
      B_pysycl.fill(12.79)

      C_pysycl = A_pysycl * B_pysycl
      C_pysycl.mem_to_cpu()

      A_np = np.full(N, 86.74, dtype= np.float64)
      B_np = np.full(N, 12.79, dtype= np.float64)
      C_np = A_np * B_np

      for i in range(N):
        self.assertAlmostEqual(C_pysycl[i], C_np[i], delta= self.tolerance_double)

  def test_in_place_vector_multiplication_double(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.double)
      B_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.double)
      A_pysycl.fill(86.74)
      B_pysycl.fill(12.79)

      A_pysycl *= B_pysycl
      A_pysycl.mem_to_cpu()

      A_np = np.full(N, 86.74, dtype= np.float64)
      B_np = np.full(N, 12.79, dtype= np.float64)
      A_np *= B_np

      for i in range(N):
        self.assertAlmostEqual(A_pysycl[i], A_np[i], delta= self.tolerance_double)

  # MULTIPLICATION FLOAT TYPE TESTS
  def test_vector_multiplication_float(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.float)
      B_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.float)
      A_pysycl.fill(86.74)
      B_pysycl.fill(12.79)

      C_pysycl = A_pysycl * B_pysycl
      C_pysycl.mem_to_cpu()

      A_np = np.full(N, 86.74, dtype= np.float32)
      B_np = np.full(N, 12.79, dtype= np.float32)
      C_np = A_np * B_np

      for i in range(N):
        self.assertAlmostEqual(C_pysycl[i], C_np[i], delta= self.tolerance_float)

  def test_in_place_vector_multiplication_float(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.float)
      B_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.float)
      A_pysycl.fill(86.74)
      B_pysycl.fill(12.79)

      A_pysycl *= B_pysycl
      A_pysycl.mem_to_cpu()

      A_np = np.full(N, 86.74, dtype= np.float32)
      B_np = np.full(N, 12.79, dtype= np.float32)
      A_np *= B_np

      for i in range(N):
        self.assertAlmostEqual(A_pysycl[i], A_np[i], delta= self.tolerance_float)

  # MULTIPLICATION INTEGER TYPE TESTS
  def test_vector_multiplication_int(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.int)
      B_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.int)
      A_pysycl.fill(8)
      B_pysycl.fill(3)

      C_pysycl = A_pysycl * B_pysycl
      C_pysycl.mem_to_cpu()

      A_np = np.full(N, 8, dtype= np.int32)
      B_np = np.full(N, 3, dtype= np.int32)
      C_np = A_np * B_np

      for i in range(N):
        self.assertEqual(C_pysycl[i], C_np[i])

  def test_in_place_vector_multiplication_int(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.int)
      B_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.int)
      A_pysycl.fill(8)
      B_pysycl.fill(3)

      A_pysycl *= B_pysycl
      A_pysycl.mem_to_cpu()

      A_np = np.full(N, 8, dtype= np.int32)
      B_np = np.full(N, 3, dtype= np.int32)
      A_np *= B_np

      for i in range(N):
        self.assertEqual(A_pysycl[i], A_np[i])

if __name__ == '__main__':
    unittest.main()
