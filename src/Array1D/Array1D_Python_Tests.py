import numpy as np
import unittest
import sys

sys.path.insert(1, '../build/')
import pysycl

print("\033[32m| ----- ARRAY 1D TEST SUITE ----- |\033[0m")

############################################
############## ADDITION TESTS ##############
############################################
class TestArray1D_Addition(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mARRAY 1D TESTS: ADDITION (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mARRAY 1D TESTS: ADDITION (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.tolerance_float  = 1e-7
    self.tolerance_double = 1e-15
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

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
  @classmethod
  def setUpClass(cls):
    print("\033[34mARRAY 1D TESTS: SUBTRACTION (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mARRAY 1D TESTS: SUBTRACTION (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.tolerance_float  = 1e-7
    self.tolerance_double = 1e-15
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

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
  @classmethod
  def setUpClass(cls):
    print("\033[34mARRAY 1D TESTS: MULTIPLICATION (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mARRAY 1D TESTS: MULTIPLICATION (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.tolerance_float  = 1e-7
    self.tolerance_double = 1e-15
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

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

############################################
############## DIVISION TESTS ##############
############################################
class TestArray1D_Dvision(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mARRAY 1D TESTS: DIVISION (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mARRAY 1D TESTS: DIVISION (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.tolerance_float  = 1e-7
    self.tolerance_double = 1e-15
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

  # DIVISION DOUBLE TYPE TESTS
  def test_vector_division_double(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.double)
      B_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.double)
      A_pysycl.fill(86.74)
      B_pysycl.fill(12.79)

      C_pysycl = A_pysycl / B_pysycl
      C_pysycl.mem_to_cpu()

      A_np = np.full(N, 86.74, dtype= np.float64)
      B_np = np.full(N, 12.79, dtype= np.float64)
      C_np = A_np / B_np

      for i in range(N):
        self.assertAlmostEqual(C_pysycl[i], C_np[i], delta= self.tolerance_double)

  def test_in_place_vector_division_double(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.double)
      B_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.double)
      A_pysycl.fill(86.74)
      B_pysycl.fill(12.79)

      A_pysycl /= B_pysycl
      A_pysycl.mem_to_cpu()

      A_np = np.full(N, 86.74, dtype= np.float64)
      B_np = np.full(N, 12.79, dtype= np.float64)
      A_np /= B_np

      for i in range(N):
        self.assertAlmostEqual(A_pysycl[i], A_np[i], delta= self.tolerance_double)

  # DIVISION FLOAT TYPE TESTS
  def test_vector_division_float(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.float)
      B_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.float)
      A_pysycl.fill(86.74)
      B_pysycl.fill(12.79)

      C_pysycl = A_pysycl / B_pysycl
      C_pysycl.mem_to_cpu()

      A_np = np.full(N, 86.74, dtype= np.float32)
      B_np = np.full(N, 12.79, dtype= np.float32)
      C_np = A_np / B_np

      for i in range(N):
        self.assertAlmostEqual(C_pysycl[i], C_np[i], delta= self.tolerance_float)

  def test_in_place_vector_division_float(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.float)
      B_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.float)
      A_pysycl.fill(86.74)
      B_pysycl.fill(12.79)

      A_pysycl /= B_pysycl
      A_pysycl.mem_to_cpu()

      A_np = np.full(N, 86.74, dtype= np.float32)
      B_np = np.full(N, 12.79, dtype= np.float32)
      A_np /= B_np

      for i in range(N):
        self.assertAlmostEqual(A_pysycl[i], A_np[i], delta= self.tolerance_float)

  # DIVISION INTEGER TYPE TESTS
  def test_vector_division_int(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.int)
      B_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.int)
      A_pysycl.fill(8)
      B_pysycl.fill(3)

      C_pysycl = A_pysycl / B_pysycl
      C_pysycl.mem_to_cpu()

      A_np = np.full(N, 8, dtype= np.int32)
      B_np = np.full(N, 3, dtype= np.int32)
      C_np = A_np // B_np

      for i in range(N):
        self.assertEqual(C_pysycl[i], C_np[i])

  def test_in_place_vector_division_int(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.int)
      B_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.int)
      A_pysycl.fill(8)
      B_pysycl.fill(3)

      A_pysycl /= B_pysycl
      A_pysycl.mem_to_cpu()

      A_np = np.full(N, 8, dtype= np.int32)
      B_np = np.full(N, 3, dtype= np.int32)
      A_np //= B_np

      for i in range(N):
        self.assertEqual(A_pysycl[i], A_np[i])

############################################
############## FILL TESTS ##################
############################################
class TestArray1D_Fill(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mARRAY 1D TESTS: Fill (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mARRAY 1D TESTS: FILL (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.tolerance_float  = 1e-7
    self.tolerance_double = 1e-15
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

  # FILL DOUBLE TYPE TESTS
  def test_vector_fill_double(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.double)
      A_pysycl.fill(86.74)
      A_pysycl.mem_to_cpu()

      A_np = np.full(N, 86.74, dtype= np.float64)

      for i in range(N):
        self.assertAlmostEqual(A_pysycl[i], A_np[i], delta= self.tolerance_double)

  # FILL FLOAT TYPE TESTS
  def test_vector_division_float(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.float)
      A_pysycl.fill(86.74)
      A_pysycl.mem_to_cpu()

      A_np = np.full(N, 86.74, dtype= np.float32)

      for i in range(N):
        self.assertAlmostEqual(A_pysycl[i], A_np[i], delta= self.tolerance_float)

  # FILL INTEGER TYPE TESTS
  def test_vector_division_int(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.int)
      A_pysycl.fill(8)
      A_pysycl.mem_to_cpu()

      A_np = np.full(N, 8, dtype= np.int32)

      for i in range(N):
        self.assertEqual(A_pysycl[i], A_np[i])

############################################
############## SIZE TESTS ##################
############################################
class TestArray1D_Fill(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mARRAY 1D TESTS: SIZE (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mARRAY 1D TESTS: SIZE (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

  # SIZE TYPE TESTS
  def test_vector_size_double(self):
    for N in [10, 100, 1000, 10000]:
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.double)
      self.assertEqual(A_pysycl.get_size(), N)

############################################
######### MAX, MIN, SUM TESTS ##############
############################################
class TestArray1D_Max(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mARRAY 1D TESTS: MAX, MIN, SUM (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mARRAY 1D TESTS: MAX, MIN, SUM (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    self.tolerance_double = 1e-12
    self.device = pysycl.device.get_device(0, 0)
    print("\033[33mrunning test...\033[0m")

  # SIZE TYPE TESTS
  def test_vector_size_double(self):
    for N in [10, 100, 250]:
      A_np = np.random.rand(N)
      A_pysycl = pysycl.array(N, device= self.device, dtype= pysycl.double)

      for i in range(N):
        A_pysycl[i] = A_np[i]

      A_pysycl.mem_to_gpu()

      max_pysycl = A_pysycl.max()
      min_pysycl = A_pysycl.min()
      sum_pysycl = A_pysycl.sum()

      max_np = A_np.max()
      min_np = A_np.min()
      sum_np = A_np.sum()

      self.assertAlmostEqual(max_pysycl, max_np, delta= self.tolerance_double)
      self.assertAlmostEqual(min_pysycl, min_np, delta= self.tolerance_double)
      self.assertAlmostEqual(sum_pysycl, sum_np, delta= self.tolerance_double)

if __name__ == '__main__':
  unittest.main()
