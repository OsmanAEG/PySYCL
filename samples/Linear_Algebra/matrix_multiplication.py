import sys
import time
import numpy as np

sys.path.insert(1, '../../build/')
import pysycl

device = pysycl.device.get_device(0, 0)

# Matrix dimensions
N = 6000
P = 6000
M = 6000

A_np = np.full((M, N), 2.0, dtype=np.float32)
B_np = np.full((N, P), 4.0, dtype=np.float32)
C_np = np.full((M, P), 0.0, dtype=np.float32)

A_ps = pysycl.array_2d(M, N, device)
B_ps = pysycl.array_2d(N, P, device)
C_ps = pysycl.array_2d(M, P, device)

A_ps.fill(2.0)
B_ps.fill(4.0)

start_time_np = time.time()
C_np = np.matmul(A_np, B_np)
end_time_np = time.time()
numpy_duration = end_time_np - start_time_np

start_time_ps = time.time()
pysycl.linalg.matmul(A_ps, B_ps, C_ps, 32)
end_time_ps = time.time()
pysycl_duration = end_time_ps - start_time_ps
pysycl_duration = pysycl_duration

C_ps.mem_to_cpu()
print("numpy time: {:.2f} seconds".format(numpy_duration))
print("pysycl time: {:.2f} seconds".format(pysycl_duration))

print("C_np[30, 50] = ", C_np[30, 50])
print("C_ps[30, 50] = ", C_ps[30, 50])
