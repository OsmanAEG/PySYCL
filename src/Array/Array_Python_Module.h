#ifndef ARRAY_PYTHON_MODULE_H
#define ARRAY_PYTHON_MODULE_H

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
/// \brief Python module for an array object in PySYCL.
///////////////////////////////////////////////////////////////////////
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Array.h"
#include "../Device/Device_Instance.h"

namespace py = pybind11;

using Scalar_T = float;
using Device_T = pysycl::Device_Instance;

///////////////////////////////////////////////////////////////////////
// Device class and functions
///////////////////////////////////////////////////////////////////////
void array_module(py::module& m){
  m.def("array", [](int dims, Device_T& device){
    return pysycl::array_selector<Scalar_T>(dims, device);
  });

  m.def("array", [](std::tuple<int, int> dims, Device_T& device){
    return pysycl::array_selector<Scalar_T>(dims, device);
  });

  m.def("array", [](py::args args){
    throw std::runtime_error("ERROR IN ARRAY: Unsupported number of dimensions.");
  });
}

#endif //ARRAY_PYTHON_MODULE_H
