import sys
sys.path.insert(1, '../../../build/')

###############################################
# Array2D.py
###############################################

from pysycl import device
from pysycl import array2D

def array2D_constructor_ex(M, N, Q):
  return array2D.array2D_object(M, N, Q)

def copy_device_to_host(arr2D):
  arr2D.copy_device_to_host()

def copy_host_to_device(arr2D):
  arr2D.copy_host_to_device()

def fill(arr2D):
  arr2D.fill(5.0)

def fill_element_host(arr2D):
  arr2D.fill_element_host(3, 2, 8.0)

def get_host_data(arr2D):
  return arr2D.get_host_data()

def sum_reduction(arr2D):
  return arr2D.sum_reduction()

def min_reduction(arr2D):
  return arr2D.min_reduction()

def max_reduction(arr2D):
  return arr2D.max_reduction()

M = 5
N = 4
Q = device.device_object(0, 0)

arr2D = array2D_constructor_ex(M, N, Q)
copy_host_to_device(arr2D)
copy_device_to_host(arr2D)
fill(arr2D)
copy_device_to_host(arr2D)
fill_element_host(arr2D)
host_array = get_host_data(arr2D)
copy_host_to_device(arr2D)
arr_sum = sum_reduction(arr2D)
arr_min = min_reduction(arr2D)
arr_max = max_reduction(arr2D)

print(host_array)
print(arr_sum)
print(arr_min)
print(arr_max)


