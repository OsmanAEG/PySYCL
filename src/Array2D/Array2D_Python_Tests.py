import numpy as np
import unittest
import sys

sys.path.insert(1, '../build/')
import pysycl

print("\033[32m| ----- ARRAY 2D TEST SUITE ----- |\033[0m")

############################################
############## ADDITION TESTS ##############
############################################
class TestArray2D_Addition(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mARRAY 2D TESTS: ADDITION (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mARRAY 2D TESTS: ADDITION (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.tolerance_float  = 1e-7
    self.tolerance_double = 1e-15
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

  # ADDITION DOUBLE TYPE TESTS
  def test_matrix_addition_double(self):
    for M in [10, 100, 1000]:
      for N in [25, 65, 450]:
        A_pysycl = pysycl.array((M, N), device= self.device, dtype= pysycl.double)
        B_pysycl = pysycl.array((M, N), device= self.device, dtype= pysycl.double)
        A_pysycl.fill(86.74)
        B_pysycl.fill(12.79)

        C_pysycl = A_pysycl + B_pysycl
        C_pysycl.mem_to_cpu()

        A_np = np.full((M, N), 86.74, dtype= np.float64)
        B_np = np.full((M, N), 12.79, dtype= np.float64)
        C_np = A_np + B_np

        for i in range(M):
          for j in range(N):
            self.assertAlmostEqual(C_pysycl[i, j], C_np[i, j], delta= self.tolerance_double)

  def test_in_place_matrix_addition_double(self):
    for M in [10, 100, 1000]:
      for N in [25, 65, 450]:
        A_pysycl = pysycl.array((M, N), device= self.device, dtype= pysycl.double)
        B_pysycl = pysycl.array((M, N), device= self.device, dtype= pysycl.double)
        A_pysycl.fill(86.74)
        B_pysycl.fill(12.79)

        A_pysycl += B_pysycl
        A_pysycl.mem_to_cpu()

        A_np = np.full((M, N), 86.74, dtype= np.float64)
        B_np = np.full((M, N), 12.79, dtype= np.float64)
        A_np += B_np

        for i in range(M):
          for j in range(N):
            self.assertAlmostEqual(A_pysycl[i, j], A_np[i, j], delta= self.tolerance_double)

  # ADDITION FLOAT TYPE TESTS
  def test_matrix_addition_float(self):
    for M in [10, 100, 1000]:
      for N in [25, 65, 450]:
        A_pysycl = pysycl.array((M, N), device= self.device, dtype= pysycl.float)
        B_pysycl = pysycl.array((M, N), device= self.device, dtype= pysycl.float)
        A_pysycl.fill(86.74)
        B_pysycl.fill(12.79)

        C_pysycl = A_pysycl + B_pysycl
        C_pysycl.mem_to_cpu()

        A_np = np.full((M, N), 86.74, dtype= np.float32)
        B_np = np.full((M, N), 12.79, dtype= np.float32)
        C_np = A_np + B_np

        for i in range(M):
          for j in range(N):
            self.assertAlmostEqual(C_pysycl[i, j], C_np[i, j], delta= self.tolerance_float)

  def test_in_place_matrix_addition_float(self):
    for M in [10, 100, 1000]:
      for N in [25, 65, 450]:
        A_pysycl = pysycl.array((M, N), device= self.device, dtype= pysycl.float)
        B_pysycl = pysycl.array((M, N), device= self.device, dtype= pysycl.float)
        A_pysycl.fill(86.74)
        B_pysycl.fill(12.79)

        A_pysycl += B_pysycl
        A_pysycl.mem_to_cpu()

        A_np = np.full((M, N), 86.74, dtype= np.float32)
        B_np = np.full((M, N), 12.79, dtype= np.float32)
        A_np += B_np

        for i in range(M):
          for j in range(N):
            self.assertAlmostEqual(A_pysycl[i, j], A_np[i, j], delta= self.tolerance_float)

  # ADDITION INTEGER TYPE TESTS
  def test_matrix_addition_int(self):
    for M in [10, 100, 1000]:
      for N in [25, 65, 450]:
        A_pysycl = pysycl.array((M, N), device= self.device, dtype= pysycl.int)
        B_pysycl = pysycl.array((M, N), device= self.device, dtype= pysycl.int)
        A_pysycl.fill(8)
        B_pysycl.fill(3)

        C_pysycl = A_pysycl + B_pysycl
        C_pysycl.mem_to_cpu()

        A_np = np.full((M, N), 8, dtype= np.int32)
        B_np = np.full((M, N), 3, dtype= np.int32)
        C_np = A_np + B_np

        for i in range(M):
          for j in range(N):
            self.assertEqual(C_pysycl[i, j], C_np[i, j])

  def test_in_place_matrix_addition_int(self):
    for M in [10, 100, 1000]:
      for N in [25, 65, 450]:
        A_pysycl = pysycl.array((M, N), device= self.device, dtype= pysycl.int)
        B_pysycl = pysycl.array((M, N), device= self.device, dtype= pysycl.int)
        A_pysycl.fill(8)
        B_pysycl.fill(3)

        A_pysycl += B_pysycl
        A_pysycl.mem_to_cpu()

        A_np = np.full((M, N), 8, dtype= np.int32)
        B_np = np.full((M, N), 3, dtype= np.int32)
        A_np += B_np

        for i in range(M):
          for j in range(N):
            self.assertEqual(A_pysycl[i, j], A_np[i, j])

############################################
########## ROWS AND COLS TESTS #############
############################################
class TestArray2D_Rows_Cols(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mARRAY 2D TESTS: ROWS AND COLS (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mARRAY 2D TESTS: ROWS AND COLS (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

  # SIZE TYPE TESTS
  def test_vector_size_double(self):
    for M in [10, 100, 1000]:
      for N in [25, 65, 450]:
        A_pysycl = pysycl.array((M, N), device= self.device, dtype= pysycl.int)
        B_pysycl = pysycl.array((M, N), device= self.device, dtype= pysycl.int)

        self.assertEqual(A_pysycl.num_rows(), M)
        self.assertEqual(A_pysycl.num_cols(), N)

if __name__ == '__main__':
  unittest.main()