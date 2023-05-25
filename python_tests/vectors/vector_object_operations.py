import sys
sys.path.insert(1, '../../build/')

import random
from pysycl import vector

vec_obj = vector.vector_object(2000, 0, 0)