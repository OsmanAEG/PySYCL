import sys
sys.path.insert(1, '../../build/')

import pysycl

device = pysycl.device.device_instance(0, 0)
A = pysycl.array_2d.array_2d_init(10, 12, device)
B = pysycl.array_2d.array_2d_init(10, 12, device)

# get number of rows
print(A.num_rows())

# get number of cols
print(A.num_cols())

# set element values
A[2, 4] = 6.0
B[2, 4] = 3.0

# matrix addition
C = A + B

print(C[2, 4])

# fill the matrix with a constant value
A.fill(45.0)
print(A[9, 7])
