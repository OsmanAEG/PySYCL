import sys
sys.path.insert(1, '../../../build/')

###############################################
# Device_Object.py
###############################################

from pysycl import device

def device_constructor_ex():
  return device.device_object(0, 0)

def device_name(device_obj):
  print(device_obj.device_name())

def device_vendor(device_obj):
  print(device_obj.device_vendor())

default_device = device_constructor_ex()
device_name(default_device)
device_vendor(default_device)