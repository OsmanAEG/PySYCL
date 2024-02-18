#ifndef ARRAY2D_H
#define ARRAY2D_H

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
#include <cmath>

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
  /// \brief Default constructor.
  Array2D() = delete;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy constructor.
  Array2D(const Array2D& og) :
      rows(og.rows),
      cols(og.cols),
      data_host(og.data_host),
      device(og.device),
      Q(og.Q){
        data_device = sycl::malloc_device<double>(rows*cols, Q);
        Q.memcpy(data_device, og.data_device, rows*cols*sizeof(double)).wait();
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Move constructor.
  Array2D(Array2D&& og) noexcept :
      rows(og.rows),
      cols(og.cols),
      data_host(std::move(og.data_host)),
      data_device(og.data_device),
      Q(std::move(og.Q)),
      device(std::move(og.device)){
        og.data_device = nullptr;
        og.rows = 0;
        og.cols = 0;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy assignment operator.
  Array2D& operator=(const Array2D& og){
    if(this != &og){
      data_host = og.data_host;
      sycl::free(data_device, Q);
      rows = og.rows;
      cols = og.cols;
      device = og.device;
      Q = og.Q;
      data_device = sycl::malloc_device<double>(rows*cols, Q);
      Q.memcpy(data_device, og.data_device, rows*cols*sizeof(double)).wait();
    }

    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Move assignment operator.
  Array2D& operator=(Array2D&& og) noexcept{
    if(this != &og){
      data_host = std::move(og.data_host);
      sycl::free(data_device, Q);
      data_device = og.data_device;
      og.data_device = nullptr;
      rows = og.rows;
      cols = og.cols;
      Q = std::move(og.Q);
      device = std::move(og.device);
      rows = 0;
      cols = 0;
    }

    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Destructor.
  ~Array2D(){
    if(data_device){
      sycl::free(data_device, Q);
      data_device = nullptr;
    }
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Constructor for the Array2D class.
  /// \param[in] rows Number of rows in the array.
  /// \param[in] cols Number of columns in the array.
  Array2D(int rows_in, int cols_in, Device_T device_in = Device_T(0, 0)) :
      rows(rows_in),
      cols(cols_in),
      data_host(rows*cols),
      device(device_in),
    Q(device_in.get_queue()){
    if(rows <= 0 || cols <= 0){
      throw std::runtime_error("ERROR IN ARRAY2D: number of cols and rows must be > 0.");
    }

    data_device = sycl::malloc_device<double>(rows*cols, Q);
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator() direct element access
  double& operator()(int i, int j) {
    if(i < 0 || i >= rows || j < 0 || j >= cols) throw std::out_of_range("Array2D access out of range");
    return data_host[i*cols + j];
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator() read-only element access
  const double& operator()(int i, int j) const {
    if(i < 0 || i >= rows || j < 0 || j >= cols) throw std::out_of_range("Array2D access out of range");
    return data_host[i*cols + j];
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator+ for matrix addition (creating a new array).
  /// \param[in] B The Array2D that is being added.
  /// \return Array2D representing the sum of the addition.
  Array2D operator+(Array2D& B){return binary_matrix_operations(B, binary_operations::ADD, false);}

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator+= for matrix addition (edits self).
  /// \param[in] B The Array2D that is being added to self.
  Array2D& operator+=(Array2D& B){
    binary_matrix_operations(B, binary_operations::ADD, true);
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator- for matrix subtraction (creating a new array).
  /// \param[in] B The Array2D that is being subtracted.
  /// \return Array2D representing the difference of the subtraction.
  Array2D operator-(Array2D& B){return binary_matrix_operations(B, binary_operations::SUBTRACT, false);}

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator-= for matrix subtraction (edits self).
  /// \param[in] B The Array2D that is being subtracted.
  Array2D& operator-=(Array2D& B){
    binary_matrix_operations(B, binary_operations::SUBTRACT, true);
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator* for element-wise multiplication (creating a new array).
  /// \param[in] B The Array2D that is being multiplied.
  /// \return Array2D representing the product of the multiplication.
  Array2D operator*(Array2D& B){return binary_matrix_operations(B, binary_operations::MULTIPLY, false);}

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator*= for element-wise multiplication (edits self).
  /// \param[in] B The Array2D that is being multiplied.
  Array2D& operator*=(Array2D& B){
    binary_matrix_operations(B, binary_operations::MULTIPLY, true);
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator/ for element-wise division (creating a new array).
  /// \param[in] B The Array2D that is being divided.
  /// \return Array2D representing the result of the division.
  Array2D operator/(Array2D& B){return binary_matrix_operations(B, binary_operations::DIVIDE, false);}

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator/= for element-wise division (edits self).
  /// \param[in] B The Array2D that is being divided.
  Array2D& operator/=(Array2D& B){
    binary_matrix_operations(B, binary_operations::DIVIDE, true);
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the number of rows in the Array2D.
  /// \return Number of rows in the Array2D.
  int num_rows() const;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the number of columns in the Array2D.
  /// \return Number of columns in the Array2D.
  int num_cols() const;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the data pointer from Array2D.
  /// \return Pointer to Array2D host data.
  std::vector<double> get_host_data_vector();

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the data pointer from Array2D.
  /// \return Pointer to Array2D device data.
  double* get_device_data_ptr();

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy memory from the CPU to the GPU.
  void mem_to_gpu();

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy memory from the GPU to the CPU
  void mem_to_cpu();

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the device from Array2D.
  /// \return The Array2D device instance.
  pysycl::Device_Instance& get_device();

  ///////////////////////////////////////////////////////////////////////
  /// \brief Fill the device with a specific value
  void fill(const double C);

private:
  ///////////////////////////////////////////////////////////////////////
  /// \brief Number of rows in the array.
  int rows;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Number of columns in the array.
  int cols;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Vector for data stored in host memory.
  std::vector<double> data_host;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Pointer to data stored in device memory.
  double* data_device;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Device that will store and handle Array2D memory and operations
  Device_T device;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Device SYCL queue.
  sycl::queue Q;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Defining enumerations of binary operations
  enum class binary_operations{ADD, SUBTRACT, MULTIPLY, DIVIDE};

  ///////////////////////////////////////////////////////////////////////
  /// \brief Function to perform binary matrix operations
  Array2D binary_matrix_operations(Array2D& B, binary_operations op, bool edit_self);

}; // class Array2D

/// @} // end "Array2D" doxygen group

} // namespace pysycl

/////////////////////////////////////////////////////////////////////////
int pysycl::Array2D::num_rows() const {
  return rows;
}

/////////////////////////////////////////////////////////////////////////
int pysycl::Array2D::num_cols() const {
  return cols;
}

/////////////////////////////////////////////////////////////////////////
std::vector<double> pysycl::Array2D::get_host_data_vector(){
  return data_host;
}

/////////////////////////////////////////////////////////////////////////
double* pysycl::Array2D::get_device_data_ptr(){
  return data_device;
}

/////////////////////////////////////////////////////////////////////////
/// \brief Copy memory from the CPU to the GPU.
void pysycl::Array2D::mem_to_gpu(){
  Q.memcpy(data_device, &data_host[0], rows*cols*sizeof(double)).wait();
}

/////////////////////////////////////////////////////////////////////////
/// \brief Copy memory from the GPU to the CPU
void pysycl::Array2D::mem_to_cpu(){
  Q.memcpy(&data_host[0], data_device, rows*cols*sizeof(double)).wait();
}

/////////////////////////////////////////////////////////////////////////
pysycl::Device_Instance& pysycl::Array2D::get_device(){
  return device;
}

/////////////////////////////////////////////////////////////////////////
void pysycl::Array2D::fill(const double C){
  const size_t B = sqrt(device.get_max_workgroup_size());

  Q.submit([&](sycl::handler& h){
    const size_t global_size_rows = ((rows + B - 1)/B)*B;
    const size_t global_size_cols = ((cols + B - 1)/B)*B;
    const auto M = rows;
    const auto N = cols;
    const auto A = data_device;

    sycl::range<2> global{global_size_rows, global_size_cols};
    sycl::range<2> local{B, B};

    h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it){
      const auto i = it.get_global_id(0);
      const auto j = it.get_global_id(1);

      if(i >= M || j >= N) return;

      A[i*N + j] = C;
    });
  }).wait();
}

/////////////////////////////////////////////////////////////////////////
pysycl::Array2D pysycl::Array2D::binary_matrix_operations(Array2D& B,
                                                          binary_operations op,
                                                          bool edit_self){
  const auto rows = this->num_rows();
  const auto cols = this->num_cols();

  if(rows != B.num_rows() || cols != B.num_cols()){
    throw std::runtime_error("ERROR: Incompatible Array2D dimensions.");
  }

  const auto platform_idx = this->device.get_platform_index();
  const auto device_idx = this->device.get_device_index();

  if(platform_idx != B.device.get_platform_index() || device_idx != B.device.get_device_index()){
    throw std::runtime_error("ERROR: Incompatible PySYCL device.");
  }

  const size_t wg_size = sqrt(this->device.get_max_workgroup_size());

  Array2D C = edit_self ? *this : Array2D(rows, cols, this->device);

  Q.submit([&](sycl::handler& h){
    const size_t global_size_rows = ((rows + wg_size - 1)/wg_size)*wg_size;
    const size_t global_size_cols = ((cols + wg_size - 1)/wg_size)*wg_size;

    sycl::range<2> global{global_size_rows, global_size_cols};
    sycl::range<2> local{wg_size, wg_size};

    auto data_1   = this->get_device_data_ptr();
    auto data_2   = B.get_device_data_ptr();
    auto data_new = C.get_device_data_ptr();

    h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it){
      const auto i = it.get_global_id(0);
      const auto j = it.get_global_id(1);

      if(i >= rows || j >= cols) return;

      switch(op){
        case binary_operations::ADD:
          data_new[i*cols + j] = data_1[i*cols + j] + data_2[i*cols + j];
          break;

        case binary_operations::SUBTRACT:
          data_new[i*cols + j] = data_1[i*cols + j] - data_2[i*cols + j];
          break;

        case binary_operations::MULTIPLY:
          data_new[i*cols + j] = data_1[i*cols + j] * data_2[i*cols + j];
          break;

        case binary_operations::DIVIDE:
          data_new[i*cols + j] = data_1[i*cols + j] / data_2[i*cols + j];
          break;
      }
    });
  }).wait();

  return C;
}

#endif // ARRAY2D_H