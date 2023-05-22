#ifndef VECTOR_OBJECT_PYBIND_MODULE_CPP
#define VECTOR_OBJECT_PYBIND_MODULE_CPP

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
/// \brief Python module for Vector Object in PySYCL.
///////////////////////////////////////////////////////////////////////

#include "Vector_Object.h"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(vector_object, m) {
  namespace py = pybind11;

  m.doc() = R"delim(
    Vector Object sub-module for PySYCL
      This sub-module is a class object for creating and manipulating vectors.
  )delim";

  py::class_<pysycl::Vector_Object>(m, "vector_object")
    .def(py::init<int, int, int>(), R"delim(
      Description
        This is a class for creating and manipulating vectors in PySYCL.

      Constructor Parameters
        N: int
          The length of the vector.
        platform_index: int
          Optional: Index for the sycl platform to select.
        device_index: int
          Optional: Index for the sycl device to select."

      Example
        >>> from pysycl import vector_object
        >>> vector = vector_object.vector_object(3000, 0, 0)
      )delim",
      py::arg("N"),
      py::arg("platform_index") = 0,
      py::arg("device_index") = 0)
    .def("set_data", &pysycl::Vector_Object::set_data, R"delim(
      Description
        This function sets the data of the vector.

      Parameters
        data_in: list[float]
          The data to be set to the vector.

      Example
        >>> from pysycl import vector_object
        >>> vector = vector_object.vector_object(3000, 0, 0)
        >>> vector.set_data([1.0, 2.0, 3.0])
      )delim")
    .def("reset_data", &pysycl::Vector_Object::reset_data, R"delim(
      Description
        This function resets the data of the vector to zero.

      Example
        >>> from pysycl import vector_object
        >>> vector = vector_object.vector_object(3000, 0, 0)
        >>> vector.set_data([1.0, 2.0, 3.0])
        >>> vector.reset_data()
        >>> vector.get_data()
        [0.0, 0.0, 0.0]
      )delim")
    .def("get_data", &pysycl::Vector_Object::get_data, R"delim(
      Description
        This function gets the data of the vector.

      Returns
        list[float]
          The data of the vector.

      Example
        >>> from pysycl import vector_object
        >>> vector = vector_object.vector_object(3000, 0, 0)
        >>> vector.set_data([1.0, 2.0, 3.0])
        >>> vector.get_data()
        [1.0, 2.0, 3.0]
      )delim")
    .def("add_vector", &pysycl::Vector_Object::add_vector, R"delim(
      Description
        This function adds two vectors together.

      Parameters
        vector_b: list[float]
          The vector to be added to the vector object calling the function.

      Example
        >>> from pysycl import vector_object
        >>> vector_a = vector_object.vector_object(3000, 0, 0)
        >>> vector_a.set_data([1.0, 2.0, 3.0])
        >>> vector_b = [4.0, 5.0, 6.0]
        >>> vector_a.add_vector(vector_b)
        >>> vector_a.get_data()
        [5.0, 7.0, 9.0]
      )delim")
    .def("add_constant", &pysycl::Vector_Object::add_constant, R"delim(
      Description
        This function adds two vectors together.

      Parameters
        C: double
          The constant to be added to the vector object calling the function.

      Example
        >>> from pysycl import vector_object
        >>> vector_a = vector_object.vector_object(3000, 0, 0)
        >>> vector_a.set_data([1.0, 2.0, 3.0])
        >>> C = 4.0
        >>> vector_a.add_constant(C)
        >>> vector_a.get_data()
        [5.0, 6.0, 7.0]
      )delim")
    .def("subtract_vector", &pysycl::Vector_Object::subtract_vector, R"delim(
      Description
        This function subtracts two vectors.

      Parameters
        vector_b: list[float]
          The vector to be subtracted from the vector object calling the function.

      Example
        >>> from pysycl import vector_object
        >>> vector_a = vector_object.vector_object(3000, 0, 0)
        >>> vector_a.set_data([1.0, 2.0, 3.0])
        >>> vector_b = [4.0, 5.0, 6.0]
        >>> vector_a.subtract_vector(vector_b)
        >>> vector_a.get_data()
        [-3.0, -3.0, -3.0]
      )delim")
    .def("subtract_constant", &pysycl::Vector_Object::subtract_constant, R"delim(
      Description
        This function subtracts a constant from a vector.

      Parameters
        C: double
          The constant to be subtracted from the vector object calling the function.

      Example
        >>> from pysycl import vector_object
        >>> vector_a = vector_object.vector_object(3000, 0, 0)
        >>> vector_a.set_data([1.0, 2.0, 3.0])
        >>> C = 4.0
        >>> vector_a.subtract_constant(C)
        >>> vector_a.get_data()
        [-3.0, -2.0, -1.0]
      )delim")
    .def("multiply_vector", &pysycl::Vector_Object::multiply_vector, R"delim(
      Description
        This function multiplies two vectors together.

      Parameters
        vector_b: list[float]
          The vector to be multiplied with the vector object calling the function.

      Example
        >>> from pysycl import vector_object
        >>> vector_a = vector_object.vector_object(3000, 0, 0)
        >>> vector_a.set_data([1.0, 2.0, 3.0])
        >>> vector_b = [4.0, 5.0, 6.0]
        >>> vector_a.multiply_vector(vector_b)
        >>> vector_a.get_data()
        [4.0, 10.0, 18.0]
      )delim")
    .def("multiply_constant", &pysycl::Vector_Object::multiply_constant, R"delim(
      Description
        This function multiplies a vector by a constant.

      Parameters
        C: double
          The constant to be multiplied with the vector object calling the function.

      Example
        >>> from pysycl import vector_object
        >>> vector_a = vector_object.vector_object(3000, 0, 0)
        >>> vector_a.set_data([1.0, 2.0, 3.0])
        >>> C = 4.0
        >>> vector_a.multiply_constant(C)
        >>> vector_a.get_data()
        [4.0, 8.0, 12.0]
      )delim")
    .def("divide_vector", &pysycl::Vector_Object::divide_vector, R"delim(
      Description
        This function divides two vectors.

      Parameters
        vector_b: list[float]
          The vector to be divided from the vector object calling the function.

      Example
        >>> from pysycl import vector_object
        >>> vector_a = vector_object.vector_object(3000, 0, 0)
        >>> vector_a.set_data([1.0, 2.0, 3.0])
        >>> vector_b = [4.0, 5.0, 6.0]
        >>> vector_a.divide_vector(vector_b)
        >>> vector_a.get_data()
        [0.25, 0.4, 0.5]
      )delim")
    .def("divide_constant", &pysycl::Vector_Object::divide_constant, R"delim(
      Description
        This function divides a vector by a constant.

      Parameters
        C: double
          The constant to be divided from the vector object calling the function.

      Example
        >>> from pysycl import vector_object
        >>> vector_a = vector_object.vector_object(3000, 0, 0)
        >>> vector_a.set_data([1.0, 2.0, 3.0])
        >>> C = 4.0
        >>> vector_a.divide_constant(C)
        >>> vector_a.get_data()
        [0.25, 0.5, 0.75]
      )delim");
}

#endif // VECTOR_OBJECT_PYBIND_MODULE_CPP
