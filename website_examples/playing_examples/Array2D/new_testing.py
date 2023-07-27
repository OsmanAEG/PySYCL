import sys
sys.path.append('../../../build')

import random
import numpy as np
import time

from pysycl import device
from pysycl import array2D

def check_elements_equal(C_arr2D, C_np, M, N):
  for i in range(M):
    for j in range(N):
      val_average = (C_arr2D[i][j] + C_np[i][j])*0.5
      percent_dif = abs((C_arr2D[i][j] - C_np[i][j])/val_average)
      assert percent_dif < 1e-4

  print("The elements are equal!")

M = 2**12
N = 2**12
P = 2**12

Q = device.device_object(0, 0)

arr2D_1 = array2D.array2D_shared(M, N, Q)
arr2D_2 = array2D.array2D_shared(N, P, Q)

np2D_1 = np.zeros((M, N))
np2D_2 = np.zeros((N, P))

'''for i in range(M):
  for j in range(N):
    rand = random.uniform(0.0, 100.0)
    arr2D_1.set_value(i, j, rand)
    np2D_1[i][j] = rand

for i in range(N):
  for j in range(P):
    rand = random.uniform(0.0, 100.0)
    arr2D_2.set_value(i, j, rand)
    np2D_2[i][j] = rand'''

start = time.time()
arr2D_r_obj = array2D.matmul(arr2D_1, arr2D_2, kernel_key = "nd", b = 32)
Q.wait()
end = time.time()
arr2D_r = arr2D_r_obj.get_data()
Q.wait()
arr2D_time = end - start
print("Time for array2D.add: ", arr2D_time)

start = time.time()
np2D_r = np.matmul(np2D_1, np2D_2)
end = time.time()
numpy_time = end - start
print("Time for numpy add: ", numpy_time)

'''check_elements_equal(arr2D_r, np2D_r, M, N)'''

print("Speedup with PySYCL: ", numpy_time/arr2D_time)


