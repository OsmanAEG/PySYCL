import sys
sys.path.insert(1, '../../build/')

import pysycl

my_devices = pysycl.device.get_device_list()
print(my_devices)