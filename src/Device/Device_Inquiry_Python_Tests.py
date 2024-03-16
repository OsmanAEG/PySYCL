import numpy as np
import unittest
import sys

sys.path.insert(1, '../build/')
import pysycl

print("\033[32m| ----- DEVICE INQUIRY TEST SUITE ----- |\033[0m")

############################################
########## DEVICE INQUIRY TESTS ############
############################################
class Device_Inquiry_Tests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\033[34mDEVICE INQUIRY TESTS (STARTING)\033[0m")

  @classmethod
  def tearDownClass(cls):
    print("\033[32mDEVICE INQUIRY TESTS (COMPLETED)\033[0m")
    print("\033[33m------------------------------------------\033[0m")

  def setUp(self):
    print("\033[33mrunning test...\033[0m")

  # DEVICE INQUIRY
  def device_inquiry(self):
    devices = pysycl.device.get_device_list()

if __name__ == '__main__':
  unittest.main()