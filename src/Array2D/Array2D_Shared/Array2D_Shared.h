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

#include "../../Device/Device_Object.h"

///////////////////////////////////////////////////////////////////////
/// \addtogroup Array2D
/// @{
namespace pysycl {
///////////////////////////////////////////////////////////////////////
/// \brief Array2D class for PySYCL (shared version)
class Array2D_Shared {
public:
  ///////////////////////////////////////////////////////////////////////
  /// \brief Default constructor, delete compiler generated version.
  Array2D_Shared() = delete;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy constructor, use compiler generated version.
  Array2D_Shared(const Array2D_Shared&) = default;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Move constructor, use compiler generated version.
  Array2D_Shared(Array2D_Shared&&) = default;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy assignment operator, use compiler generated version.
  /// \return Reference to the assigned object.
  Array2D_Shared& operator=(const Array2D_Shared&) = default;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Move assignment operator, use compiler generated version.
  /// \return Reference to the assigned object.
  Array2D_Shared& operator=(Array2D_Shared&&) = default;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Constructor for the Array2D class.
  /// \param[in] rows Number of rows in the array.
  /// \param[in] cols Number of columns in the array.
  Array2D_Shared(int rows_in,
                 int cols_in,
                 pysycl::Device_Object device_in) :
    rows(rows_in),
    cols(cols_in),
    device(device_in),
    Q(device.queue()) {
      if(rows <= 0 || cols <= 0){
        throw std::runtime_error("Array2D_Explicit: rows and cols must be > 0");
      }

      data = sycl::malloc_shared<float>(rows*cols, Q);
    }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the number of rows in the array.
  /// \return Number of rows in the array.
  int number_of_rows() const;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the number of columns in the array.
  /// \return Number of columns in the array.
  int number_of_cols() const;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the device object for the array.
  /// \return Device object for the array.
  pysycl::Device_Object get_device() const;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Set the value of an element in the array.
  /// \param[in] row Row of the element to set.
  /// \param[in] col Column of the element to set.
  /// \param[in] value Value to set the element to.
  void set_value(int row, int col, float value);

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the value of an element in the array.
  /// \param[in] row Row of the element to get.
  /// \param[in] col Column of the element to get.
  /// \return Value of the element.
  float get_value(int row, int col) const;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Return host data as a 2D vector.
  /// \return Host data as a 2D vector.
  std::vector<std::vector<float>> get_data() const;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get data pointer
  /// \return Data pointer
  float* get_data_ptr() const{return data;}

private:
  ///////////////////////////////////////////////////////////////////////
  /// \brief Number of rows in the array.
  int rows;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Number of columns in the array.
  int cols;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Device object for the array.
  pysycl::Device_Object device;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Queue for the device.
  sycl::queue Q;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Shared data for the array.
  float* data;

}; // class Array2D_Explicit

/// @} // end "Array2D" doxygen group

} // namespace pysycl

///////////////////////////////////////////////////////////////////////
// Get the number of rows in the array.
int pysycl::Array2D_Shared::number_of_rows() const {
  return rows;
}

///////////////////////////////////////////////////////////////////////
// Get the number of columns in the array.
int pysycl::Array2D_Shared::number_of_cols() const {
  return cols;
}

///////////////////////////////////////////////////////////////////////
// Get the device object for the array.
pysycl::Device_Object pysycl::Array2D_Shared::get_device() const {
  return device;
}

///////////////////////////////////////////////////////////////////////
// Set the value of an element in the array.
void pysycl::Array2D_Shared::set_value(int row, int col, float value) {
  if(row < 0 || row >= rows || col < 0 || col >= cols){
    throw std::runtime_error("Array2D_Shared: row or col out of bounds");
  }

  data[row*cols + col] = value;
}

///////////////////////////////////////////////////////////////////////
// Get the value of an element in the array.
float pysycl::Array2D_Shared::get_value(int row, int col) const {
  if(row < 0 || row >= rows || col < 0 || col >= cols){
    throw std::runtime_error("Array2D_Shared: row or col out of bounds");
  }

  return data[row*cols + col];
}

///////////////////////////////////////////////////////////////////////
// Return host data as a 2D vector.
std::vector<std::vector<float>> pysycl::Array2D_Shared::get_data() const {
  std::vector<std::vector<float>> host_data(rows, std::vector<float>(cols));

  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      host_data[i][j] = data[i*cols + j];
    }
  }

  return host_data;
}

#endif // ARRAY2D_SHARED_H