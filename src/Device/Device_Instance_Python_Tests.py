import numpy as np
import unittest
import sys

sys.path.insert(1, '../build/')
import pysycl

print("\033[32m| ----- DEVICE INSTANCE TEST SUITE ----- |\033[0m")

############################################
########## DEVICE INSTANCE TESTS ###########
############################################
class Device_Instance_Tests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mDEVICE INSTANCE TESTS (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mDEVICE INSTANCE TESTS (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    print("\033[33mrunning test...\033[0m")

  # DEVICE INSTANCE
  def device_instance(self):
    Q = pysycl.device.get_device(0,0);
    name = Q.name()
    vencor = Q.vendor()

if __name__ == '__main__':
  unittest.main()