import sys
sys.path.insert(1, '../../build/')

import random
from pysycl import vector

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

########################################
########################################
## basic tests for vector operations
########################################
########################################

# basic test for vector addition
def basic_addition(N):
  print("Running basic addition test...")

  a = generate_random_vector(-100, 100, N)
  b = generate_random_vector(-100, 100, N)

  expected_result = [x + y for x, y in zip(a, b)]

  c = vector.vector_addition(a, b)
  check_vector_full_value(c, expected_result)

# basic test for vector subtraction
def basic_subtraction(N):
  print("Running basic subtraction test...")

  a = generate_random_vector(-100, 100, N)
  b = generate_random_vector(-100, 100, N)

  expected_result = [x - y for x, y in zip(a, b)]

  c = vector.vector_subtraction(a, b)
  check_vector_full_value(c, expected_result)

# basic test for vector element multiplication
def basic_element_multiplication(N):
  print("Running basic element multiplication test...")

  a = generate_random_vector(-100, 100, N)
  b = generate_random_vector(-100, 100, N)

  expected_result = [x * y for x, y in zip(a, b)]

  c = vector.vector_element_multiplication(a, b)
  check_vector_full_value(c, expected_result)

# basic test for vector element division
def basic_element_division(N):
  print("Running basic element division test...")

  a = generate_random_vector(-100, 100, N)
  b = generate_random_vector(-100, 100, N)

  expected_result = [x / y for x, y in zip(a, b)]

  c = vector.vector_element_division(a, b)
  check_vector_full_value(c, expected_result)

# run all basic tests
def basic_tests(N):
  basic_addition(N)
  basic_subtraction(N)
  basic_element_multiplication(N)
  basic_element_division(N)

########################################
########################################
## vector reduction operations
########################################
########################################

# test for vector sum reduction
def vector_sum_reduction(N):
  print("Running vector sum reduction test...")

  a = generate_random_vector(-100, 100, N)

  expected_result = sum(a)

  a_sum = vector.vector_sum_reduction(a)
  check_single_value(a_sum, expected_result)

# test for vector minimum value reduction
def vector_min_reduction(N):
  print("Running vector sum reduction test...")

  a = generate_random_vector(-100, 100, N)

  expected_result = min(a)

  a_min = vector.vector_min_reduction(a)
  check_single_value(a_min, expected_result)

# test for vector maximum value reduction
def vector_max_reduction(N):
  print("Running vector sum reduction test...")

  a = generate_random_vector(-100, 100, N)

  expected_result = max(a)

  a_max = vector.vector_max_reduction(a)
  check_single_value(a_max, expected_result)

# run all reduction tests
def reduction_tests(N):
  vector_sum_reduction(N)
  vector_min_reduction(N)
  vector_max_reduction(N)

########################################
########################################
## advanced vector operations
########################################
########################################

# test for vector dot product
def vector_dot_product(N):
  print("Running vector dot product test...")

  a = generate_random_vector(-100, 100, N)
  b = generate_random_vector(-100, 100, N)

  expected_result = sum([x * y for x, y in zip(a, b)])

  a_dot_b = vector.vector_dot_product(a, b)
  check_single_value(a_dot_b, expected_result)

# run all advanced vector operations
def advanced_vector_operations(N):
  vector_dot_product(N)

########################################
########################################
## running tests
########################################
########################################
N = 3000
basic_tests(N)
reduction_tests(N)
advanced_vector_operations(N)