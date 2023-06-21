import sys
sys.path.insert(1, '../../build/')

import random
from pysycl import device
from pysycl import vector

q = device.device_object(0, 0)

vec_obj = vector.vector_object(2000, q)