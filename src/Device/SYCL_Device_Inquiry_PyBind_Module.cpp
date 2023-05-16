#ifndef SYCL_DEVICE_INQUIRY_PYBIND_MODULE_CPP
#define SYCL_DEVICE_INQUIRY_PYBIND_MODULE_CPP

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
/// \brief Python module for SYCL device inquiry in PySYCL.
///////////////////////////////////////////////////////////////////////

#include "SYCL_Device_Inquiry.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(device_inquiry, m) {
  namespace py = pybind11;

  m.doc() = "SYCL device inquiry sub-module for PySYCL.";

  m.def("platform_list", &pysycl::platform_list, R"delim(
    Returns a list of available SYCL platforms.
  )delim");

  m.def("device_list", &pysycl::device_list, R"delim(
    Returns a list of available SYCL devices.
  )delim");
}

#endif // #ifndef SYCL_DEVICE_INQUIRY_PYBIND_MODULE_CPP