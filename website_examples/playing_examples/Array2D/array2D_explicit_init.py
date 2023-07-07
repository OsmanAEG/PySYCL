import sys
sys.path.insert(1, '../../../build/')

###############################################
# array2D_explicit_init.py
###############################################

from pysycl import device
from pysycl import array2D

def array2D_init(rows, cols, Q):
  A = array2D.array2D_explicit(rows, cols, Q)
  return A

Q = device.device_object(0, 0)
A = array2D_init(21, 76, Q)

print(A.number_of_rows())