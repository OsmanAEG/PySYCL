import sys
sys.path.insert(1, '../../build/')

import pysycl

device = pysycl.device.device_instance(0, 0)

M = 5
N = 3

A = pysycl.array_2d.array_2d_init(M, N, device)
B = pysycl.array_2d.array_2d_init(M, N, device)

for i in range(M):
  for j in range(N):
    A[i, j] = i*j

A.mem_to_gpu()

print("MAX VALUE: " + str(A.max()))
print("MIN VALUE: " + str(A.min()))
print("SUM VALUE: " + str(A.sum()))