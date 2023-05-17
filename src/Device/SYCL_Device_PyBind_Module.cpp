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

PYBIND11_MODULE(device_queue, m) {

  namespace py = pybind11;

  m.doc() = R"delim(
    SYCL device selection sub-module for PySYCL
      This sub-module is a class object for selecting and investigating a specific device.
  )delim";

  py::class_<pysycl::SYCL_Device>(m, "device_select")
    .def(py::init<int, int>(), R"delim(
      Description
        This is a class for SYCL device selection in PySYCL.

      Constructor Parameters
        platform_index: int
          Optional: Index for the sycl platform to select.
        device_index: int
          Optional: Index for the sycl device to select."

      Example
        >>> from pysycl import device_queue
        >>> gpu_queue = device_queue.device_select(0, 0)
      )delim",
      py::arg("platform_index") = 0,
      py::arg("device_index") = 0)
    .def("device_name", &pysycl::SYCL_Device::device_name, R"delim(
      Description
        This function outputs the selected device name.

      Returns
        str
          The name of the selected device.

      Example
        >>> from pysycl import device_queue
        >>> gpu_queue = device_queue.device_select(0, 0)
        >>> gpu_queue.device_name()
        'Intel(R) Gen9 HD Graphics NEO'
      )delim")
    .def("device_vendor", &pysycl::SYCL_Device::device_vendor, R"delim(
      Description
        This function outputs the selected device vendor.

      Returns
        str
          The vendor of the selected device.

      Example
        >>> from pysycl import device_queue
        >>> gpu_queue = device_queue.device_select(0, 0)
        >>> gpu_queue.device_vendor()
        'Intel(R) Corporation'
      )delim");
}

#endif // #ifndef SYCL_DEVICE_PYBIND_MODULE_CPP
