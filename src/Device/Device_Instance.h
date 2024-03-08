#ifndef DEVICE_INSTANCE_H
#define DEVICE_INSTANCE_H

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
#include <CL/sycl.hpp>

///////////////////////////////////////////////////////////////////////
/// \addtogroup Device
/// @{

namespace pysycl {

///////////////////////////////////////////////////////////////////////
/// \brief Class representing a device instance.
class Device_Instance {
public:
    /////////////////////////////////////////////////////////////////////
    /// \brief Copy constructor, use compiler generated version.
    Device_Instance(const Device_Instance&) = default;

    /////////////////////////////////////////////////////////////////////
    /// \brief Move constructor, use compiler generated version.
    Device_Instance(Device_Instance&&) = default;

    /////////////////////////////////////////////////////////////////////
    /// \brief Copy assignment, use compiler generated version.
    /// \return reference to the assigned object.
    Device_Instance& operator=(const Device_Instance&) = default;

    /////////////////////////////////////////////////////////////////////
    /// \brief Move assignment, use compiler generated version.
    /// \return reference to the assigned object.
    Device_Instance& operator=(Device_Instance&&) = default;

    /////////////////////////////////////////////////////////////////////
    /// \brief Output device name.
    auto name();

    /////////////////////////////////////////////////////////////////////
    /// \brief Output device vendor.
    auto vendor();

    /////////////////////////////////////////////////////////////////////
    /// \brief Returns the device queue.
    auto& get_queue();

    /////////////////////////////////////////////////////////////////////
    /// \brief Returns the platform index.
    auto get_platform_index() const;

    /////////////////////////////////////////////////////////////////////
    /// \brief Returns the device index.
    auto get_device_index() const;

    /////////////////////////////////////////////////////////////////////
    /// \brief Returns the maximum work group size of the device.
    auto get_max_workgroup_size() const;

    /////////////////////////////////////////////////////////////////////
    /// \brief Constructor that selects a SYCL device.
    /// \param[in] platform_index Index of the sycl platform to select.
    /// \param[in] device_index Index of the sycl device to select.
    Device_Instance(int platform_idx_in = 0, int device_idx_in = 0)
      : platform_idx(platform_idx_in)
      , device_idx(device_idx_in) {
        if (platform_idx < 0) {
            throw std::runtime_error(
              "ERROR: Platform index must be non-negative.");
        }

        if (device_idx < 0) {
            throw std::runtime_error(
              "ERROR: Device index must be non-negative.");
        }

        auto platforms = sycl::platform::get_platforms();

        if (platform_idx >= platforms.size()) {
            throw std::runtime_error("ERROR: Platform index out of range.");
        }

        auto devices = platforms[platform_idx].get_devices();

        if (device_idx >= devices.size()) {
            throw std::runtime_error("ERROR: Device index out of range.");
        }

        Q = sycl::queue(devices[device_idx]);
    }

private:
    /////////////////////////////////////////////////////////////////////
    /// \brief The selected device queue.
    sycl::queue Q;

    /////////////////////////////////////////////////////////////////////
    /// \brief The platform index.
    int platform_idx;

    /////////////////////////////////////////////////////////////////////
    /// \brief The device index based on the platform index.
    int device_idx;
};

/////////////////////////////////////////////////////////////////////////
auto Device_Instance::name() {
    static const std::string name
      = Q.get_device().get_info<sycl::info::device::name>();
    return name;
}

/////////////////////////////////////////////////////////////////////////
auto Device_Instance::vendor() {
    static const std::string vendor
      = Q.get_device().get_info<sycl::info::device::vendor>();
    return vendor;
}

/////////////////////////////////////////////////////////////////////////
auto& Device_Instance::get_queue() { return Q; }

/////////////////////////////////////////////////////////////////////////
auto Device_Instance::get_platform_index() const { return platform_idx; }

/////////////////////////////////////////////////////////////////////////
auto Device_Instance::get_device_index() const { return device_idx; }

/////////////////////////////////////////////////////////////////////////
auto Device_Instance::get_max_workgroup_size() const {
    return Q.get_device().get_info<sycl::info::device::max_work_group_size>();
}

} // namespace pysycl

#endif // #ifndef DEVICE_INSTANCE_H
