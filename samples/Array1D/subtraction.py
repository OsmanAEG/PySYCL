import sys
sys.path.insert(1, '../../build/')

import pysycl

device = pysycl.device.get_device(0, 0)

N = 300

A = pysycl.array(N, device= device, dtype = pysycl.float)
B = pysycl.array(N, device= device, dtype = pysycl.float)

print("Fill A with 1.0 and B with 2.0")
print("Compute C = A - B")
A.fill(1.0)
B.fill(2.0)

C = A - B

C.mem_to_cpu()
print("C[30] = " + str(C[30]))

print("Now compute C -= A")
C -= A

C.mem_to_cpu()
print("C[30] = " + str(C[30]) + "\n")
