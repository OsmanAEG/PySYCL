import sys
sys.path.insert(1, '../../build/')

import pysycl

device_manager = pysycl.device.get_device_manager()
cpu = pysycl.device.get_device(0,0);
cpu = device_manager.get_device(0,0);
gpu = device_manager.get_device(1,0);
print(cpu.name(), gpu.name())
