import sys
sys.path.insert(1, '../../build/')

import pysycl

# create a device instance
Q = pysycl.device.device_instance(0, 0)

# output the device name
print(Q.name())

# output the device vendor
print(Q.vendor())

