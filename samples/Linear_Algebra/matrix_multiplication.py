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

A_np = np.full((M, N), 2.0)
B_np = np.full((N, P), 4.0)

B_ps = pysycl.array_2d(M, N, device)
A_ps = pysycl.array_2d(N, P, device)

A_ps.fill(2.0)
B_ps.fill(4.0)

start_time_np = time.time()
C_np = np.matmul(A_np, B_np)
end_time_np = time.time()
numpy_duration = end_time_np - start_time_np

start_time_ps = time.time()
C_ps = pysycl.linalg.tiled_matmul(A_ps, B_ps, 32)
end_time_ps = time.time()
pysycl_duration = end_time_ps - start_time_ps

C_ps.mem_to_cpu()
print("numpy time: {:.2f} seconds".format(numpy_duration))
print("pysycl time: {:.2f} seconds".format(pysycl_duration))

print("C_np[30, 50] = ", C_np[30, 50])
print("C_ps[30, 50] = ", C_ps[30, 50])
