import sys
sys.path.append('../../../build')

import random
import numpy as np

from pysycl import device
from pysycl import array2D

def check_elements_equal(C_arr2D, C_np, M, N):
  for i in range(M):
    for j in range(N):
      val_average = (C_arr2D[i][j] + C_np[i][j])*0.5
      percent_dif = abs((C_arr2D[i][j] - C_np[i][j])/val_average)
      assert percent_dif < 1e-4

  print("The elements are equal!")

M = 2**9
N = 2**9

Q = device.device_object(0, 0)

arr2D_1 = array2D.array2D_shared(M, N, Q)
arr2D_2 = array2D.array2D_shared(M, N, Q)

np2D_1 = np.zeros((M, N))
np2D_2 = np.zeros((M, N))

for i in range(M):
  for j in range(N):
    rand1 = random.uniform(0.0, 100.0)
    rand2 = random.uniform(0.0, 100.0)

    arr2D_1.set_value(i, j, rand1)
    arr2D_2.set_value(i, j, rand2)

    np2D_1[i][j] = rand1
    np2D_2[i][j] = rand2

arr2D_r_obj = array2D.add(arr2D_1, arr2D_2, kernel_key = "nd", b = 32)
Q.wait()

arr2D_r = arr2D_r_obj.get_data()
Q.wait()

np2D_r = np2D_1 + np2D_2

check_elements_equal(arr2D_r, np2D_r, M, N)


