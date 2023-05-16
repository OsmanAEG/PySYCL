#ifndef VECTOR_OPERATIONS_PYBIND_MODULE_H
#define VECTOR_OPERATIONS_PYBIND_MODULE_H

///////////////////////////////////////////////////////////////////////
// This file is part of the PySYCL software for SYCL development in
// Python.  It is licensed under the MIT licence.  A copy of
// this license, in a file named LICENSE.md, should have been
// distributed with this file.  A copy of this license is also
// currently available at "http://opensource.org/licenses/MIT".
//
// Unless explicitly stated, all contributions intentionally submitted
// to this project shall also be under the terms and conditions of this
// license, without any additional terms or conditions.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
/// \file
/// \brief Python module for vector operations in PySYCL.
///////////////////////////////////////////////////////////////////////

#include "Vector_Operations.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(vector_operations, m) {
  namespace py = pybind11;

  m.doc() = "Vector operations sub-module for PySYCL.";

  m.def("vector_addition", &pysycl::Vector_Addition, R"delim(
    Performs a vector addition on two python lists and returns the sum.

    Parameters
    ----------
    vector_a : list[float]
        The first vector to be added.

    vector_b : list[float]
        The second vector to be added.

    platform_index : int, default = 0
        optional: The index of the device platform to be used.

    device_index : int, default = 0
        optional: The index of the device to be used for a given platform.

    Returns
    ----------
    Returns : list[float]
        The sum of the two input vectors.

    Examples
    ----------
    >>> from pysycl import vector_operations as vec_ops
    >>> vector_a = [1.0, 2.0, 3.0]
    >>> vector_b = [4.0, 5.0, 6.0]
    >>> vec_ops.vector_addition(a, b)
    [5.0, 7.0, 9.0]
  )delim",
  py::arg("vector_a"),
  py::arg("vector_b"),
  py::arg("platform_index") = 0,
  py::arg("device_index") = 0);

  m.def("vector_subtraction", &pysycl::Vector_Subtraction, R"delim(
    Performs a vector subtraction on two python lists and returns the difference.

    Parameters
    ----------
    vector_a : list[float]
        The first minuend vector.

    vector_b : list[float]
        The second subtrahend vector.

    platform_index : int, default = 0
        optional: The index of the device platform to be used.

    device_index : int, default = 0
        optional: The index of the device to be used for a given platform.

    Returns
    ----------
    Returns : list[float]
        The difference of the two input vectors.

    Examples
    ----------
    >>> from pysycl import vector_operations as vec_ops
    >>> vector_a = [1.0, 2.0, 3.0]
    >>> vector_b = [4.0, 5.0, 6.0]
    >>> vec_ops.vector_subtraction(a, b)
    [-3.0, -3.0, -3.0]
  )delim",
  py::arg("vector_a"),
  py::arg("vector_b"),
  py::arg("platform_index") = 0,
  py::arg("device_index") = 0);

  m.def("vector_element_multiplication", &pysycl::Vector_Element_Multiplication, R"delim(
    Performs a vector element-wise multiplication on two python lists and returns the product.

    Parameters
    ----------
    vector_a : list[float]
        The first vector to be multiplied (element-wise).

    vector_b : list[float]
        The second vector to be multiplied (element-wise).

    platform_index : int, default = 0
        optional: The index of the device platform to be used.

    device_index : int, default = 0
        optional: The index of the device to be used for a given platform.

    Returns
    ----------
    Returns : list[float]
        The vector product.

    Examples
    ----------
    >>> from pysycl import vector_operations as vec_ops
    >>> vector_a = [1.0, 2.0, 3.0]
    >>> vector_b = [4.0, 5.0, 6.0]
    >>> vec_ops.vector_element_multiplication(a, b)
    [4.0, 10.0, 18.0]
  )delim",
  py::arg("vector_a"),
  py::arg("vector_b"),
  py::arg("platform_index") = 0,
  py::arg("device_index") = 0);

  m.def("vector_element_division", &pysycl::Vector_Element_Division, R"delim(
    Performs a vector element-wise division on two python lists and returns the product.

    Parameters
    ----------
    vector_a : list[float]
        The first vector for element-wise division (numerator).

    vector_b : list[float]
        The first vector for element-wise division (denominator).

    platform_index : int, default = 0
        optional: The index of the device platform to be used.

    device_index : int, default = 0
        optional: The index of the device to be used for a given platform.

    Returns
    ----------
    Returns : list[float]
        The vector quotient.

    Examples
    ----------
    >>> from pysycl import vector_operations as vec_ops
    >>> vector_a = [1.0, 2.0, 3.0]
    >>> vector_b = [4.0, 5.0, 6.0]
    >>> vec_ops.vector_element_division(a, b)
    [0.25, 0.4, 0.5]
  )delim",
  py::arg("vector_a"),
  py::arg("vector_b"),
  py::arg("platform_index") = 0,
  py::arg("device_index") = 0);

  m.def("vector_sum_reduction", &pysycl::Vector_Sum_Reduction, R"delim(
    Performs a sum reduction on an input vector and returns the total.

    Parameters
    ----------
    vector_a : list[float]
        The input vector for sum reduction.

    platform_index : int, default = 0
        optional: The index of the device platform to be used.

    device_index : int, default = 0
        optional: The index of the device to be used for a given platform.

    Returns
    ----------
    Returns : float
        The sum of every element in the input vector.

    Examples
    ----------
    >>> from pysycl import vector_operations as vec_ops
    >>> vector_a = [1.0, 2.0, 3.0]
    >>> vec_ops.vector_sum_reduction(a)
    6.0
  )delim",
  py::arg("vector_a"),
  py::arg("platform_index") = 0,
  py::arg("device_index") = 0);

  m.def("vector_min_reduction", &pysycl::Vector_Min_Reduction, R"delim(
    Performs a minimum reduction on an input vector and returns the minimum value.

    Parameters
    ----------
    vector_a : list[float]
        The input vector minimum reduction.

    platform_index : int, default = 0
        optional: The index of the device platform to be used.

    device_index : int, default = 0
        optional: The index of the device to be used for a given platform.

    Returns
    ----------
    Returns : float
        The minimum element value in the input vector.

    Examples
    ----------
    >>> from pysycl import vector_operations as vec_ops
    >>> vector_a = [1.0, 2.0, 3.0]
    >>> vec_ops.vector_min_reduction(a)
    1.0
  )delim",
  py::arg("vector_a"),
  py::arg("platform_index") = 0,
  py::arg("device_index") = 0);

  m.def("vector_max_reduction", &pysycl::Vector_Max_Reduction, R"delim(
    Performs a maximum reduction on an input vector and returns the minimum value.

    Parameters
    ----------
    vector_a : list[float]
        The input vector for maximum reduction.

    platform_index : int, default = 0
        optional: The index of the device platform to be used.

    device_index : int, default = 0
        optional: The index of the device to be used for a given platform.

    Returns
    ----------
    Returns : float
        The maximum element value in the input vector.

    Examples
    ----------
    >>> from pysycl import vector_operations as vec_ops
    >>> vector_a = [1.0, 2.0, 3.0]
    >>> vec_ops.vector_max_reduction(a)
    3.0
  )delim",
  py::arg("vector_a"),
  py::arg("platform_index") = 0,
  py::arg("device_index") = 0);

  m.def("vector_dot_product", &pysycl::Vector_Dot_Product, R"delim(
    Performs a vector dot product on two python lists and returns the sum.

    Parameters
    ----------
    vector_a : list[float]
        The first vector for dot product.

    vector_b : list[float]
        The second vector for dot product.

    platform_index : int, default = 0
        optional: The index of the device platform to be used.

    device_index : int, default = 0
        optional: The index of the device to be used for a given platform.

    Returns
    ----------
    Returns : float
        The vector dot product.

    Examples
    ----------
    >>> from pysycl import vector_operations as vec_ops
    >>> vector_a = [1.0, 2.0, 3.0]
    >>> vector_b = [4.0, 5.0, 6.0]
    >>> vec_ops.vector_dot_product(a, b)
    32.0
  )delim",
  py::arg("vector_a"),
  py::arg("vector_b"),
  py::arg("platform_index") = 0,
  py::arg("device_index") = 0);
}

#endif // VECTOR_OPERATIONS_PYBIND_MODULE