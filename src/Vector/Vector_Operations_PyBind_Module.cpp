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
    Adds two vectors.
  )delim",
  py::arg("vector_a"),
  py::arg("vector_b"),
  py::arg("platform_index") = 0,
  py::arg("device_index") = 0);

  m.def("vector_subtraction", &pysycl::Vector_Subtraction, R"delim(
    Subtracts two vectors.
  )delim",
  py::arg("vector_a"),
  py::arg("vector_b"),
  py::arg("platform_index") = 0,
  py::arg("device_index") = 0);

  m.def("vector_element_multiplication", &pysycl::Vector_Element_Multiplication, R"delim(
    Multiplies the respective elements of two vectors.
  )delim",
  py::arg("vector_a"),
  py::arg("vector_b"),
  py::arg("platform_index") = 0,
  py::arg("device_index") = 0);

  m.def("vector_element_division", &pysycl::Vector_Element_Division, R"delim(
    Divides the respective elements of two vectors.
  )delim",
  py::arg("vector_a"),
  py::arg("vector_b"),
  py::arg("platform_index") = 0,
  py::arg("device_index") = 0);

  m.def("vector_sum_reduction", &pysycl::Vector_Sum_Reduction, R"delim(
    Performs a sum reduction on an input vector.
  )delim",
  py::arg("vector_a"),
  py::arg("platform_index") = 0,
  py::arg("device_index") = 0);

  m.def("vector_min_reduction", &pysycl::Vector_Min_Reduction, R"delim(
    Performs a minimum reduction on an input vector.
  )delim",
  py::arg("vector_a"),
  py::arg("platform_index") = 0,
  py::arg("device_index") = 0);

  m.def("vector_max_reduction", &pysycl::Vector_Max_Reduction, R"delim(
    Performs a maximum reduction on an input vector.
  )delim",
  py::arg("vector_a"),
  py::arg("platform_index") = 0,
  py::arg("device_index") = 0);
}

#endif // VECTOR_OPERATIONS_PYBIND_MODULE