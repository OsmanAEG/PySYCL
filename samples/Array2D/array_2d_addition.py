import sys
sys.path.insert(1, '../../build/')

import pysycl

device = pysycl.device.device_instance(0, 0)

M = 500
N = 300

A = pysycl.array_2d.array_2d_init(M, N, device)
B = pysycl.array_2d.array_2d_init(M, N, device)

print("Fill A with 1.0 and B with 2.0")
print("Compute C = A + B")
A.fill(1.0)
B.fill(2.0)

C = A + B
C.mem_to_cpu()

print("C[30, 50] = " + str(C[30, 50]) + "\n")

print("Now compute C += A")
C += A
C.mem_to_cpu()

print("C[30, 50] = " + str(C[30, 50]) + "\n")



