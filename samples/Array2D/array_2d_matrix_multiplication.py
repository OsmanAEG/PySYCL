import sys
import time
import numpy as np

sys.path.insert(1, '../../build/')
import pysycl

device = pysycl.device.device_instance(0, 0)

# matrix dimensions
M = 4000
N = 800
P = 2500

# initialize numpy and pysycl arrays
A_np = np.full((M, N), 8.0)
B_np = np.full((N, P), 3.0)
C_np = np.zeros((M, P))

A_ps = pysycl.array_2d.array_2d_init(M, N, device)
B_ps = pysycl.array_2d.array_2d_init(N, P, device)
C_ps = pysycl.array_2d.array_2d_init(M, P, device)

A_ps.fill(8.0)
B_ps.fill(3.0)

# numpy execution time
start_time_np = time.time()
C_np = np.matmul(A_np, B_np)
end_time_np = time.time()
numpy_duration = end_time_np - start_time_np

# pysycl execution time
start_time_ps = time.time()
C_ps.matmul(A_ps, B_ps)
end_time_ps = time.time()
pysycl_duration = end_time_ps - start_time_ps

# output
print("numpy time: {:.2f} seconds".format(numpy_duration))
print("pysycl time: {:.2f} seconds".format(pysycl_duration))

C_ps.mem_to_cpu()

print("C_np[30, 50] = ", C_np[30, 50])
print("C_ps[30, 50] = ", C_ps[30, 50])
