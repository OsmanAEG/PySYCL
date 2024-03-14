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

///////////////////////////////////////////////////////////////////////
/// pybind11
///////////////////////////////////////////////////////////////////////
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

///////////////////////////////////////////////////////////////////////
/// stl
///////////////////////////////////////////////////////////////////////
#include <tuple>

///////////////////////////////////////////////////////////////////////
/// local
///////////////////////////////////////////////////////////////////////
#include "../Device/Device_Instance.h"
#include "../Device/Device_Manager.h"
#include "../Data_Types/Data_Types.h"
#include "Array.h"

namespace py = pybind11;

using Device_T = pysycl::Device_Instance;
using Data_T = pysycl::Data_Types;

///////////////////////////////////////////////////////////////////////
// Device class and functions
///////////////////////////////////////////////////////////////////////
void array_module(py::module &m) {
  // Array 1D Factories
  m.def("array", [](int dims, Device_T& device, Data_T& dtype) {
    return pysycl::array_selector(dims, device, dtype);
  }, py::arg("dims"),
     py::arg("device"),
     py::arg("dtype"));

  // Array 2D Factories
  m.def("array", [](std::tuple<int, int> dims, Device_T& device, Data_T& dtype) {
    return pysycl::array_selector(dims, device, dtype);
  }, py::arg("dims"),
     py::arg("device"),
     py::arg("dtype"));
}

#endif // ARRAY_PYTHON_MODULE_H
