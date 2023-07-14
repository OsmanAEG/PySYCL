import sys
sys.path.insert(1, '../../../build/')

###############################################
# time_tests_array2D.py
###############################################

from pysycl import device
from pysycl import array2D
import numpy as np
import random
import time

###############################################
# Time testing fill
###############################################
def fill_test():
	# set up
	'''M = 10000
	N = 10000

	Q = device.device_object(0, 0)

	arr2D = array2D.array2D_object(M, N, Q)
	np2D = np.zeros((M, N))

	# test
	start = time.time()
	arr2D.fill(5.0)
	sum = arr2D.sum_reduction()
	end = time.time()

	print("Time to fill array2D: ", end - start)

	start = time.time()
	np2D.fill(5.0)
	sum = np2D.sum()
	end = time.time()

	print("Time to fill numpy array: ", end - start)

###############################################
# Time testing add
###############################################
def add_test():
	M = 8000
	N = 7000

	Q = device.device_object(0, 0)

	arr2D_1 = array2D.array2D_object(M, N, Q)
	arr2D_2 = array2D.array2D_object(M, N, Q)

	np2D_1 = np.zeros((M, N))
	np2D_2 = np.zeros((M, N))

	# filling the arrays
	arr2D_1.fill(1.0)
	arr2D_2.fill(2.0)

	np2D_1.fill(1.0)
	np2D_2.fill(2.0)

	# test
	start = time.time()
	arr2D_3 = array2D.add(arr2D_1, arr2D_2)
	end = time.time()

	print("Time to add array2D: ", end - start)

	start = time.time()
	np2D_3 = np2D_1 + np2D_2
	end = time.time()

	print("Time to add numpy array: ", end - start)

###############################################
# Time testing subtract
###############################################
def subtract_test():
	M = 8000
	N = 7000

	Q = device.device_object(0, 0)

	arr2D_1 = array2D.array2D_object(M, N, Q)
	arr2D_2 = array2D.array2D_object(M, N, Q)

	np2D_1 = np.zeros((M, N))
	np2D_2 = np.zeros((M, N))

	# filling the arrays
	arr2D_1.fill(1.0)
	arr2D_2.fill(2.0)

	np2D_1.fill(1.0)
	np2D_2.fill(2.0)

	# test
	start = time.time()
	arr2D_3 = array2D.subtract(arr2D_1, arr2D_2)
	end = time.time()

	print("Time to subtract array2D: ", end - start)

	start = time.time()
	np2D_3 = np2D_1 - np2D_2
	end = time.time()

	print("Time to subtract numpy array: ", end - start)

###############################################
# Time testing multiply
###############################################
def multiply_test():
	M = 800
	N = 700
	P = 600

	Q = device.device_object(0, 0)

	arr2D_1 = array2D.array2D_object(M, N, Q)
	arr2D_2 = array2D.array2D_object(N, P, Q)

	np2D_1 = np.zeros((M, N))
	np2D_2 = np.zeros((N, P))

	# filling the arrays
	arr2D_1.fill(1.0)
	arr2D_2.fill(2.0)

	np2D_1.fill(1.0)
	np2D_2.fill(2.0)

	# test
	start = time.time()
	arr2D_3 = array2D.matmul(arr2D_1, arr2D_2)
	Q.wait()
	end = time.time()

	#arr2D_3.copy_device_to_host()
	#result = arr2D_3.get_host_data()

	#print(result)

	print("Time to multiply array2D: ", end - start)

	#start = time.time()
	#np2D_3 = np.matmul(np2D_1, np2D_2)
	#end = time.time()

	#print(np2D_3)

	#print("Time to multiply numpy array: ", end - start)

###############################################
###############################################
# Executing the tests
###############################################
###############################################
#fill_test()
#add_test()
#subtract_test()
multiply_test()'''

M = 50
N = 30

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

arr2D_r = array2D.add(arr2D_1, arr2D_2)
A_r_np  = np2D_1 + np2D_2

Q.wait()
A_r_sycl = arr2D_r.get_data()


