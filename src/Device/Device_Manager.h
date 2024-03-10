#ifndef DEVICE_MANAGER_H
#define DEVICE_MANAGER_H

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
/// \brief Device instance for device selection in PySYCL.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
/// sycl
///////////////////////////////////////////////////////////////////////
#include <CL/sycl.hpp>

///////////////////////////////////////////////////////////////////////
/// local
///////////////////////////////////////////////////////////////////////
#include "Device_Instance.h"

///////////////////////////////////////////////////////////////////////
/// \addtogroup Device
/// @{

namespace pysycl {

///////////////////////////////////////////////////////////////////////
/// \brief Class representing a device instance.
class Device_Manager {
public:
  /////////////////////////////////////////////////////////////////////
  /// \brief Get singleton instance of device manager.
  /// \return The singleton device manager.
  static Device_Manager &get_device_manager() {
    static Device_Manager device_manager;
    return device_manager;
  }

  /////////////////////////////////////////////////////////////////////
  /// \brief Copy constructor, deleted.
  Device_Manager(const Device_Manager &) = delete;

  /////////////////////////////////////////////////////////////////////
  /// \brief Copy assignment, deleted.
  /// \return reference to the assigned object.
  Device_Manager &operator=(const Device_Manager &) = delete;

  Device_Instance &get_device(int platform_idx = 0, int device_idx = 0) {
    const std::string hash_device_idx =
        std::to_string(platform_idx) + "-" + std::to_string(device_idx);
    auto device_map_it = device_map.find(hash_device_idx);
    if (device_map_it == device_map.end()) {
      auto device_it =
          device_map
              .insert_or_assign(hash_device_idx,
                                Device_Instance(platform_idx, device_idx))
              .first;
      return device_it->second;
    } else {
      return device_map_it->second;
    }
  }

private:
  /////////////////////////////////////////////////////////////////////
  /// \brief Default constructor, use compiler generated version.
  Device_Manager() = default;

  std::unordered_map<std::string, Device_Instance> device_map;
};

inline Device_Instance &get_device(int platform_idx = 0, int device_idx = 0) {
  auto &device_manager = Device_Manager::get_device_manager();
  return device_manager.get_device(platform_idx, device_idx);
}

} // namespace pysycl

#endif // #ifndef DEVICE_MANAGER_H
