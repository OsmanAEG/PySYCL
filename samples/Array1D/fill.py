import sys
sys.path.insert(1, '../../build/')

import pysycl

device = pysycl.device.get_device(0, 0)

N = 10

A = pysycl.array(N, device= device, dtype = pysycl.float)
B = pysycl.array(N, device= device, dtype = pysycl.float)

# get number of elements
print("Number of elements in A: " + str(A.get_size()))

# # set element values
print("Set A[2] = 6.0 and set B[2] = 3.0")
print("----------------------------------------")
A[2] = 6.0
B[2] = 3.0

A.mem_to_cpu()
B.mem_to_cpu()
print("A[2] = " + str(A[2]))
print("B[2] = " + str(B[2]) + "\n")

# fill the matrix with a constant value
print("Fill A with 45.0")
print("----------------------------------------")
A.fill(45.0)
A.mem_to_cpu()
print("A[9] = " + str(A[9]))
