#ifndef SYCL_DEVICE_INQUIRY_H
#define SYCL_DEVICE_INQUIRY_H

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
/// \brief Collecting SYCL device availability and information.
///////////////////////////////////////////////////////////////////////

#include <iostream>
#include <CL/sycl.hpp>

///////////////////////////////////////////////////////////////////////
/// \addtogroup Device
/// @{

namespace pysycl{

///////////////////////////////////////////////////////////////////////
/// \brief Function that returns a list of available SYCL platforms.
std::vector<std::string> platform_list(){
  std::vector<std::string> platform_names;
  auto platforms = sycl::platform::get_platforms();

  int i = 0;

  for(auto &p : platforms){
    platform_names.push_back(p.get_info<sycl::info::platform::name>() +
                             " ( platform index = " + std::to_string(i) + ")");

    i++;
  }
  return platform_names;
}

///////////////////////////////////////////////////////////////////////
/// \brief Function that returns a list of available SYCL devices.
std::vector<std::string> device_list(int platform_index){
  std::vector<std::string> device_names;
  auto platforms = sycl::platform::get_platforms();
  auto selected_platform = platforms[platform_index];
  auto devices = selected_platform.get_devices();

  int i = 0;

  for(auto &d : devices){
    device_names.push_back(d.get_info<sycl::info::device::name>() +
                           " ( device index = " + std::to_string(i) + ")");

    i++;
  }
  return device_names;
}

///////////////////////////////////////////////////////////////////////
/// \brief Function that returns a sycl device queue.
sycl::queue get_queue(int platform_index = 0, int device_index = 0){
  auto platforms = sycl::platform::get_platforms();
  auto selected_platform = platforms[platform_index];
  auto devices = selected_platform.get_devices();
  auto selected_device = devices[device_index];
  sycl::queue q(selected_device);
  return q;
}

} // namespace pysycl

#endif // #ifndef SYCL_DEVICE_INQUIRY_H