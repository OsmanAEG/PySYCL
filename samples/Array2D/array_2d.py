import sys
sys.path.insert(1, '../../build/')

import pysycl

device = pysycl.device.device_instance(0, 0)
A = pysycl.array_2d.array_2d_init(12, 10, device)

# get number of cols
print(A.num_cols())

# get number of rows
print(A.num_rows())

print(A[2, 4])
A[2, 4] = 6.0
print(A[2, 4])
