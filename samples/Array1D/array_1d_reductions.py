import sys
sys.path.insert(1, '../../build/')

import pysycl

device = pysycl.device.device_instance(0, 0)

N = 3

A = pysycl.array_1d(N, device)

for i in range(N):
  A[i] = i

A.mem_to_gpu()

print("MAX VALUE: " + str(A.max()))
print("MIN VALUE: " + str(A.min()))
print("SUM VALUE: " + str(A.sum()))