import sys
sys.path.insert(1, '../../build/')

import pysycl

float64 = pysycl.double
float32 = pysycl.float
int_t   = pysycl.int