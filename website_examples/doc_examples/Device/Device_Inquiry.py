import sys
sys.path.insert(1, '../../../build/')

###############################################
# Device_Inquiry.py
###############################################

from pysycl import device

def platform_list_ex():
  print(device.platform_list())

def device_list_ex():
  print(device.device_list(0))

platform_list_ex()
device_list_ex()

