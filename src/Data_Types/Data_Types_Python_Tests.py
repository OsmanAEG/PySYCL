import numpy as np
import unittest
import sys

sys.path.insert(1, '../build/')
import pysycl

print("\033[32m| ----- DATA TYPES TEST SUITE ----- |\033[0m")

############################################
####### SETTING UP DATA TYPE TESTS #########
############################################
class Data_Type_Tests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mDATA TYPES TESTS: SETTING DATA TYPES (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mDATA TYPES TESTS: SETTING DATA TYPES (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    print("\033[33mrunning test...\033[0m")

  # SETTING DATA TYPES
  def setting_data_types(self):
    float64 = pysycl.double
    float32 = pysycl.float
    int_t   = pysycl.int

if __name__ == '__main__':
  unittest.main()