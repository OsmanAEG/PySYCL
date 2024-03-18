import numpy as np
import unittest
import sys

sys.path.insert(1, '../build/')
import pysycl

print("\033[32m| ----- DEVICE MANAGER TEST SUITE ----- |\033[0m")

############################################
########## DEVICE MANAGER TESTS ############
############################################
class Device_Manager_Tests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mDEVICE MANAGER TESTS (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mDEVICE MANAGER TESTS (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    print("\033[33mrunning test...\033[0m")

  # DEVICE INSTANCE
  def device_manager(self):
    device_manager = pysycl.device.get_device_manager()
    dev = pysycl.device.get_device(0,0);
    name = dev.name()
    vendor = dev.vendor()

    self.assertIsNotNone(device_manager, "pysycl.device.get_device_manager() failed")
    self.assertIsNotNone(dev, "pysycl.device.get_device(0,0) failed.")
    self.assertIsNotNone(name, "name function failed.")
    self.assertIsNotNone(vendor, "vendor function failed.")

if __name__ == '__main__':
  unittest.main()