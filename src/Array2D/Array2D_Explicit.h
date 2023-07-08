#ifndef ARRAY2D_EXPLICIT_H
#define ARRAY2D_EXPLICIT_H

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
/// \brief Array2D in PySYCL. This is the explicit version of the
///        Array2D class. It is used when the user wants to explicitly
///        control memory movement between the host and device.
///////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <vector>

#include "../Device/Device_Object.h"

///////////////////////////////////////////////////////////////////////
/// \addtogroup Array2D
/// @{
namespace pysycl {
///////////////////////////////////////////////////////////////////////
/// \brief Array2D class for PySYCL (explicit version)
class Array2D_Explicit {
public:
  /////////////////////////////////////////////////////////////////////
  /// \brief Default constructor, delete compiler generated version.
  Array2D_Explicit() = delete;

  /////////////////////////////////////////////////////////////////////
  /// \brief Copy constructor, use compiler generated version.
  Array2D_Explicit(const Array2D_Explicit&) = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Move constructor, use compiler generated version.
  Array2D_Explicit(Array2D_Explicit&&) = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Copy assignment operator, use compiler generated version.
  /// \return Reference to the assigned object.
  Array2D_Explicit& operator=(const Array2D_Explicit&) = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Move assignment operator, use compiler generated version.
  /// \return Reference to the assigned object.
  Array2D_Explicit& operator=(Array2D_Explicit&&) = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Constructor for the Array2D class.
  /// \param[in] rows Number of rows in the array.
  /// \param[in] cols Number of columns in the array.
  /// \param[in] device SYCL device to use for allocation and compilation.
  Array2D_Explicit(int rows_in,
                   int cols_in,
                   pysycl::Device_Object device_in) :
    rows(rows_in),
    cols(cols_in),
    device(device_in),
    Q(device.queue()) {
      if(rows <= 0 || cols <= 0){
        throw std::runtime_error("Array2D_Explicit: rows and cols must be > 0");
      }

      host_data.resize(rows*cols);
      device_data = sycl::malloc_device<float>(rows*cols, Q);
  }

  /////////////////////////////////////////////////////////////////////
  /// \brief Get the number of rows in the array.
  /// \return Number of rows in the array.
  int number_of_rows() const;

  /////////////////////////////////////////////////////////////////////
  /// \brief Get the number of columns in the array.
  /// \return Number of columns in the array.
  int number_of_cols() const;

  /////////////////////////////////////////////////////////////////////
  /// \brief Get the SYCL device used for allocation and compilation.
  /// \return SYCL device used for allocation and compilation.
  pysycl::Device_Object get_device() const;

  /////////////////////////////////////////////////////////////////////
  /// \brief Copy data from host to device.
  void copy_host_to_device();

  /////////////////////////////////////////////////////////////////////
  /// \brief Copy data from device to host.
  void copy_device_to_host();

  /////////////////////////////////////////////////////////////////////
  /// \brief Set the value of an element in the array on the host.
  /// \param[in] row Row index of the element to set.
  /// \param[in] col Column index of the element to set.
  /// \param[in] value Value to set the element to.
  void set_host_value(int row, int col, float value);

  /////////////////////////////////////////////////////////////////////
  /// \brief Get the value of an element in the array on the host.
  /// \param[in] row Row index of the element to get.
  /// \param[in] col Column index of the element to get.
  /// \return Value of the element.
  float get_host_value(int row, int col) const;

  /////////////////////////////////////////////////////////////////////
  /// \brief Return host data as a 2D vector.
  /// \return Host data as a 2D vector.
  std::vector<std::vector<float>> get_host_data() const;

  /////////////////////////////////////////////////////////////////////
  /// \brief Get data pointer on the device.
  /// \return Data pointer on the device.
  float* get_data_ptr() const{return device_data;}

private:
  /////////////////////////////////////////////////////////////////////
  /// \brief Number of rows in the array.
  int rows;

  /////////////////////////////////////////////////////////////////////
  /// \brief Number of columns in the array.
  int cols;

  /////////////////////////////////////////////////////////////////////
  /// \brief SYCL device to use for allocation and compilation.
  pysycl::Device_Object device;

  /////////////////////////////////////////////////////////////////////
  /// \brief Device queue to use for allocation and compilation.
  ///        Queue is extracted from device object.
  sycl::queue Q;

  /////////////////////////////////////////////////////////////////////
  /// \brief Vector containing the data on the host.
  std::vector<float> host_data;

  /////////////////////////////////////////////////////////////////////
  /// \brief Pointer to the data on the device.
  float* device_data;

}; // class Array2D_Explicit

/// @} // end "Array2D" doxygen group

} // namespace pysycl

///////////////////////////////////////////////////////////////////////
// Get the number of rows in the array.
int pysycl::Array2D_Explicit::number_of_rows() const {
  return rows;
}

///////////////////////////////////////////////////////////////////////
// Get the number of columns in the array.
int pysycl::Array2D_Explicit::number_of_cols() const {
  return cols;
}

///////////////////////////////////////////////////////////////////////
// Get the SYCL device used for allocation and compilation.
pysycl::Device_Object pysycl::Array2D_Explicit::get_device() const {
  return device;
}

///////////////////////////////////////////////////////////////////////
// Copy data from host to device.
void pysycl::Array2D_Explicit::copy_host_to_device() {
  Q.memcpy(device_data, host_data.data(), rows*cols*sizeof(float));
}

///////////////////////////////////////////////////////////////////////
// Copy data from device to host.
void pysycl::Array2D_Explicit::copy_device_to_host() {
  Q.memcpy(host_data.data(), device_data, rows*cols*sizeof(float));
}

///////////////////////////////////////////////////////////////////////
// Set the value of an element in the array on the host.
void pysycl::Array2D_Explicit::set_host_value(int row,
                                              int col,
                                              float value) {
  if(row < 0 || row >= rows || col < 0 || col >= cols){
    throw std::runtime_error("Array2D_Explicit: row or col out of bounds");
  }
  host_data[row*cols + col] = value;
}

///////////////////////////////////////////////////////////////////////
// Get the value of an element in the array on the host.
float pysycl::Array2D_Explicit::get_host_value(int row, int col) const {
  if(row < 0 || row >= rows || col < 0 || col >= cols){
    throw std::runtime_error("Array2D_Explicit: row or col out of bounds");
  }
  return host_data[row*cols + col];
}

///////////////////////////////////////////////////////////////////////
// Return host data as a 2D vector.
std::vector<std::vector<float>> pysycl::Array2D_Explicit::get_host_data() const {
  std::vector<std::vector<float>> host_data_2d(rows, std::vector<float>(cols));

  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      host_data_2d[i][j] = host_data[i*cols + j];
    }
  }

  return host_data_2d;
}

#endif // ARRAY2D_EXPLICIT_H