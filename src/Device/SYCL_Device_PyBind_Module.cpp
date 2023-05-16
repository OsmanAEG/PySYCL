#ifndef SYCL_DEVICE_PYBIND_MODULE_CPP
#define SYCL_DEVICE_PYBIND_MODULE_CPP

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
/// \brief Python module for SYCL device selection in PySYCL.
///////////////////////////////////////////////////////////////////////

#include "SYCL_Device.h"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(device, m) {

  namespace py = pybind11;

  m.doc() = "SYCL device selection sub-module for PySYCL.";

  py::class_<pysycl::SYCL_Device>(m, "device_select", "SYCL device selection class")
    .def(py::init<int, int>(), R"delim(
      Constructor that selects a SYCL device.

      Args:
        platform_index (int): Index of the sycl platform to select.
        device_index (int): Index of the sycl device to select."
      )delim",
      py::arg("platform_index") = 0,
      py::arg("device_index") = 0)
    .def("device_name", &pysycl::SYCL_Device::device_name, R"delim(
      Output device name.
      )delim")
    .def("device_vendor", &pysycl::SYCL_Device::device_vendor, R"delim(
      Output device vendor.
      )delim");
}

#endif // #ifndef SYCL_DEVICE_PYBIND_MODULE_CPP
