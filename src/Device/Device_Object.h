#ifndef DEVICE_OBJECT_H
#define DEVICE_OBJECT_H

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
/// \brief Device object for device selection in PySYCL.
///////////////////////////////////////////////////////////////////////

#include <iostream>
#include <CL/sycl.hpp>

///////////////////////////////////////////////////////////////////////
/// \addtogroup Device
/// @{

namespace pysycl{

///////////////////////////////////////////////////////////////////////
/// \brief Class representing a device object.
class Device_Object {
public:
  /////////////////////////////////////////////////////////////////////
  /// \brief Default constructor, use compiler generated version.
  Device_Object() = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Copy constructor, use compiler generated version.
  Device_Object(const Device_Object&) = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Move constructor, use compiler generated version.
  Device_Object(Device_Object&&) = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Copy assignment, use compiler generated version.
  /// \return reference to the assigned object.
  Device_Object& operator=(const Device_Object&) = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Move assignment, use compiler generated version.
  /// \return reference to the assigned object.
  Device_Object& operator=(Device_Object&&) = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Output device name.
  std::string device_name();

  /////////////////////////////////////////////////////////////////////
  /// \brief Output device vendor.
  std::string device_vendor();

  /////////////////////////////////////////////////////////////////////
  /// \brief Return Queue.
  sycl::queue queue() const;

  /////////////////////////////////////////////////////////////////////
  /// \brief Wait for all queues to finish.
  void wait();

  /////////////////////////////////////////////////////////////////////
  /// \brief Constructor that selects a SYCL device.
  /// \param[in] platform_index Index of the sycl platform to select.
  /// \param[in] device_index Index of the sycl device to select.
  Device_Object(int platform_index = 0, int device_index = 0) {
    if (platform_index < 0) {
      throw std::runtime_error("platform index must be non-negative.");
    }

    if (device_index < 0) {
      throw std::runtime_error("device index must be non-negative.");
    }

    auto platforms = sycl::platform::get_platforms();

    if (platform_index >= platforms.size()) {
      throw std::runtime_error("platform index out of range.");
    }

    auto devices = platforms[platform_index].get_devices();

    if (device_index >= devices.size()) {
      throw std::runtime_error("device index out of range.");
    }

    Q = sycl::queue(devices[device_index]);
  }

  private:
    /////////////////////////////////////////////////////////////////////
    /// \brief The selected sycl device.
    sycl::queue Q;
};

/////////////////////////////////////////////////////////////////////////
std::string Device_Object::device_name() {
  return Q.get_device().get_info<sycl::info::device::name>();
}

/////////////////////////////////////////////////////////////////////////
std::string Device_Object::device_vendor() {
  return Q.get_device().get_info<sycl::info::device::vendor>();
}

/////////////////////////////////////////////////////////////////////////
sycl::queue Device_Object::queue() const {
  return Q;
}

/////////////////////////////////////////////////////////////////////////
void Device_Object::wait() {
  Q.wait();
}

} // namespace pysycl

#endif // #ifndef SYCL_DEVICE_H