#ifndef ARRAY2D_SHARED_H
#define ARRAY2D_SHARED_H

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
/// \brief Array2D in PySYCL. This is the shared version of the
///        Array2D class. It is used when the user wants to implicitly
///        control memory movement between the host and device.
///////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <vector>

#include "../Device/Device_Instance.h"

///////////////////////////////////////////////////////////////////////
/// \addtogroup Array2D
/// @{
namespace pysycl {
///////////////////////////////////////////////////////////////////////
/// \brief Array2D class for PySYCL
class Array2D {
public:
  ///////////////////////////////////////////////////////////////////////
  /// \brief Defining the device type.
  using Device_T = pysycl::Device_Instance;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Default constructor, delete compiler generated version.
  Array2D() = delete;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy constructor, use compiler generated version.
  Array2D(const Array2D&) = default;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Move constructor, use compiler generated version.
  Array2D(Array2D&&) = default;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy assignment operator, use compiler generated version.
  /// \return Reference to the assigned object.
  Array2D& operator=(const Array2D&) = default;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Move assignment operator, use compiler generated version.
  /// \return Reference to the assigned object.
  Array2D& operator=(Array2D&&) = default;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Constructor for the Array2D class.
  /// \param[in] cols Number of columns in the array.
  /// \param[in] rows Number of rows in the array.
  Array2D(int cols_in, int rows_in, Device_T device_in = Device_T(0, 0)) :
    cols(cols_in),
    rows(rows_in),
    device(device_in){
      if(cols <= 0 || rows <= 0){
        throw std::runtime_error("ERROR IN ARRAY2D: number of cols and rows must be > 0.");
      }

      Q = device.get_queue();
      data = sycl::malloc_shared<double>(cols*rows, Q);
    }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Class desctructor to free allocated memory
  ~Array2D(){sycl::free(data, Q);}

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator() direct element access
  double& operator()(int i, int j) {
    if(i < 0 || i >= rows || j < 0 || j >= cols) throw std::out_of_range("Array2D access out of range");
    return data[i * cols + j];
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator() read-only element access
  const double& operator()(int i, int j) const {
    if(i < 0 || i >= rows || j < 0 || j >= cols) throw std::out_of_range("Array2D access out of range");
    return data[i * cols + j];
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator+ for matrix addition (creating a new array)
  Array2D operator+(const Array2D& rhs) const {
    const auto cols = this->num_cols();
    const auto rows = this->num_rows();

    if(cols != rhs.num_cols() || rows != rhs.num_rows()){
      throw std::runtime_error("ERROR: Incompatible Array2D dimensions.");
    }

    const auto platform_idx = this->device.get_platform_index();
    const auto device_idx = this->device.get_device_index();

    if(platform_idx != rhs.device.get_platform_index() || device_idx != rhs.device.get_device_index()){
      throw std::runtime_error("ERROR: Incompatible PySYCL device.");
    }

    const size_t B = this->device.get_max_workgroup_size();

    Array2D lhs = Array2D(cols, rows, this->device);

    auto event = Q.submit([&](sycl::handler& h){

    })
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator+ for matrix addition (adjusting the current array)
  Array2D operator+=(const Array2D& rhs) const {
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the number of columns in the Array2D.
  /// \return Number of columns in the Array2D.
  int num_cols() const;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the number of rows in the Array2D.
  /// \return Number of rows in the Array2D.
  int num_rows() const;

private:
  ///////////////////////////////////////////////////////////////////////
  /// \brief Number of columns in the array.
  int cols;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Number of rows in the array.
  int rows;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Pointer to data stored in the array.
  double* data;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Device SYCL queue.
  sycl::queue Q;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Device that will store and handle Array2D memory and operations
  Device_T device;

}; // class Array2D

/// @} // end "Array2D" doxygen group

} // namespace pysycl

/////////////////////////////////////////////////////////////////////////
int pysycl::Array2D::num_cols() const {
  return cols;
}

/////////////////////////////////////////////////////////////////////////
int pysycl::Array2D::num_rows() const {
  return rows;
}

#endif // ARRAY2D_SHARED_H