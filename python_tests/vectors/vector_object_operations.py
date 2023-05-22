import sys
sys.path.insert(1, '../../build/PySYCL')

import random
from pysycl import vector_object as vec_obj

vec_obj = vec_obj.vector_object(2000, 0, 0)