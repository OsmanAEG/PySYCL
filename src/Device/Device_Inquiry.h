#ifndef DEVICE_INQUIRY_H
#define DEVICE_INQUIRY_H

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
/// \brief Device inquiry in PySYCL.
///////////////////////////////////////////////////////////////////////

/// Device Management in PySYCL

///////////////////////////////////////////////////////////////////////
// sycl
///////////////////////////////////////////////////////////////////////
#include <CL/sycl.hpp>

///////////////////////////////////////////////////////////////////////
// stl
///////////////////////////////////////////////////////////////////////
#include <iostream>

///////////////////////////////////////////////////////////////////////
/// \addtogroup Device
/// @{

namespace pysycl {
///////////////////////////////////////////////////////////////////////
/// \brief Function that returns a list of available devices.
std::vector<std::string> get_device_list() {
  std::vector<std::string> device_list;
  const auto &platforms = sycl::platform::get_platforms();

  for (int i = 0; i < platforms.size(); ++i) {
    const auto &devices = platforms[i].get_devices();

    for (int j = 0; j < devices.size(); ++j) {
      auto si = std::to_string(i);
      auto sj = std::to_string(j);
      device_list.push_back(devices[j].get_info<sycl::info::device::name>() +
                            " [" + si + ", " + sj + "]");
    }
  }

  return device_list;
}

} // namespace pysycl

#endif // #ifndef DEVICE_INQUIRY_H
