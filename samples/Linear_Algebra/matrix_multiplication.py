import sys
import time
import numpy as np

sys.path.insert(1, '../../build/')
import pysycl

device = pysycl.device.device_instance(0, 0)

# Matrix dimensions
M = 3000
N = 3000
P = 3000

np.random.seed(35)
A_np = np.random.rand(M, N).astype(np.float32)
B_np = np.random.rand(N, P).astype(np.float32)

B_ps = pysycl.array_2d(N, P, device)
A_ps = pysycl.array_2d(M, N, device)

for i in range(M):
  for j in range(N):
    A_ps[i, j] = A_np[i, j]

for i in range(N):
  for j in range(P):
    B_ps[i, j] = B_np[i, j]

A_ps.mem_to_gpu()
B_ps.mem_to_gpu()

start_time_np = time.time()
C_np = np.matmul(A_np, B_np)
end_time_np = time.time()
numpy_duration = end_time_np - start_time_np

start_time_ps = time.time()
C_ps = pysycl.linalg.matmul(A_ps, B_ps)
end_time_ps = time.time()
pysycl_duration = end_time_ps - start_time_ps

print("numpy time: {:.2f} seconds".format(numpy_duration))
print("pysycl time: {:.2f} seconds".format(pysycl_duration))

C_ps.mem_to_cpu()

print("C_np[30, 50] = ", C_np[30, 50])
print("C_ps[30, 50] = ", C_ps[30, 50])
