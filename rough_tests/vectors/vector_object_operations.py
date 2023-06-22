import sys
sys.path.insert(1, '../../build/')

import random
from pysycl import device
from pysycl import vector

# creating a PySYCL device object for queue
q = device.device_object(0, 0)

# helper function to check the result of a vector operation
#   full vector, full values, or full values with tolerance
def check_vector_full_value(vec, expected_result, tol=1e-6):

  if len(vec) != len(expected_result):
    raise ValueError("Lists must have the same length.")

  for i in range(len(vec)):
    if abs(vec[i] - expected_result[i]) > tol:
      print("Vector element {} is {} but should be {}".format(i, vec[i], expected_result))
      print("ERROR: The test failed")
      assert(False)

  print("SUCCESS: The test passed!")

# helper function to check the result of a vector operation
#   full vector, single value, or single value with tolerance
def check_vector_singe_value(vec, expected_result, tol=1e-6):
  for i in range(len(vec)):
    if abs(vec[i] - expected_result) > tol:
      print("Vector element {} is {} but should be {}".format(i, vec[i], expected_result))
      print("ERROR: The test failed")
      assert(False)

  print("SUCCESS: The test passed!")

# helper function to check the result of a vector operation
#   single value, or single value with tolerance
def check_single_value(result, expected_result, tol=1e-6):
  if abs(result - expected_result) > tol:
    print("Result is {} but should be {}".format(result, expected_result))
    print("ERROR: The test failed")
    assert(False)

  print("SUCCESS: The test passed!")

# helper function to generate a random vector of size N
def generate_random_vector(min_val, max_val, N):
  return [random.uniform(min_val, max_val) for _ in range(N)]

vec_obj = vector.vector_object(2000, q)

########################################
########################################
## tests for vector objects
########################################
########################################

###########################################
# tests for basic vector element operations
###########################################

###########################################
# addition

# test for vector addition
def vector_addition(N):
  print("Running vector addition test...")

  a = generate_random_vector(-100, 100, N)
  b = generate_random_vector(-100, 100, N)

  expected_result = [x + y for x, y in zip(a, b)]

  a_sycl = vector.vector_object(N, q)
  a_sycl.set_data(a)
  a_sycl.add_vector(b)

  c = a_sycl.get_data()
  check_vector_full_value(c, expected_result)

# test for add constant
def vector_add_constant(N):
  print("Running vector add constant test...")

  a = generate_random_vector(-100, 100, N)
  const = 50

  expected_result = [x + const for x in a]

  a_sycl = vector.vector_object(N, q)
  a_sycl.set_data(a)
  a_sycl.add_constant(const)

  c = a_sycl.get_data()
  check_vector_full_value(c, expected_result)

###########################################
# subtraction

def vector_subtraction(N):
  print("Running vector subtraction test...")

  a = generate_random_vector(-100, 100, N)
  b = generate_random_vector(-100, 100, N)

  expected_result = [x - y for x, y in zip(a, b)]

  a_sycl = vector.vector_object(N, q)
  a_sycl.set_data(a)
  a_sycl.subtract_vector(b)

  c = a_sycl.get_data()
  check_vector_full_value(c, expected_result)

def vector_subtract_constant(N):
  print("Running vector subtract constant test...")

  a = generate_random_vector(-100, 100, N)
  const = 50

  expected_result = [x - const for x in a]

  a_sycl = vector.vector_object(N, q)
  a_sycl.set_data(a)
  a_sycl.subtract_constant(const)

  c = a_sycl.get_data()
  check_vector_full_value(c, expected_result)

###########################################
# multiplication
def vector_multiplication(N):
  print("Running vector multiplication test...")

  a = generate_random_vector(-100, 100, N)
  b = generate_random_vector(-100, 100, N)

  expected_result = [x * y for x, y in zip(a, b)]

  a_sycl = vector.vector_object(N, q)
  a_sycl.set_data(a)
  a_sycl.multiply_vector(b)

  c = a_sycl.get_data()
  check_vector_full_value(c, expected_result)

def vector_multiply_constant(N):
  print("Running vector multiply constant test...")

  a = generate_random_vector(-100, 100, N)
  const = 50

  expected_result = [x * const for x in a]

  a_sycl = vector.vector_object(N, q)
  a_sycl.set_data(a)
  a_sycl.multiply_constant(const)

  c = a_sycl.get_data()
  check_vector_full_value(c, expected_result)

###########################################
# division
def vector_division(N):
  print("Running vector division test...")

  a = generate_random_vector(-100, 100, N)
  b = generate_random_vector(-100, 100, N)

  expected_result = [x / y for x, y in zip(a, b)]

  a_sycl = vector.vector_object(N, q)
  a_sycl.set_data(a)
  a_sycl.divide_vector(b)

  c = a_sycl.get_data()
  check_vector_full_value(c, expected_result)

def vector_divide_constant(N):
  print("Running vector divide constant test...")

  a = generate_random_vector(-100, 100, N)
  const = 50

  expected_result = [x / const for x in a]

  a_sycl = vector.vector_object(N, q)
  a_sycl.set_data(a)
  a_sycl.divide_constant(const)

  c = a_sycl.get_data()
  check_vector_full_value(c, expected_result)

def basic_tests(N):
  # running the tests
  vector_addition(2000)
  vector_add_constant(2000)
  vector_subtraction(2000)
  vector_subtract_constant(2000)
  vector_multiplication(2000)
  vector_multiply_constant(2000)
  vector_division(2000)
  vector_divide_constant(2000)

########################################
########################################
## running tests
########################################
########################################
N = 2000
basic_tests(N)