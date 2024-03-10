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
/// \brief Array2D in PySYCL.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
/// sycl
///////////////////////////////////////////////////////////////////////
#include <CL/sycl.hpp>

///////////////////////////////////////////////////////////////////////
/// pybind11
///////////////////////////////////////////////////////////////////////
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

///////////////////////////////////////////////////////////////////////
/// stl
///////////////////////////////////////////////////////////////////////
#include <cmath>
#include <vector>

///////////////////////////////////////////////////////////////////////
/// local
///////////////////////////////////////////////////////////////////////
#include "../Device/Device_Instance.h"
#include "../Device/Device_Manager.h"

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
/// \addtogroup Array2D
/// @{
namespace pysycl {
///////////////////////////////////////////////////////////////////////
/// \brief Array2D class for PySYCL
template <typename Scalar_type> class Array2D {
public:
  ///////////////////////////////////////////////////////////////////////
  /// \brief Defining the device scalar type.
  using Scalar_T = Scalar_type;

  using Array_T = Array2D<Scalar_T>;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Defining the device type.
  using Device_T = pysycl::Device_Instance;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Basic constructor that takes in the size and device of
  ///        the array.
  /// \param[in] rows_in Number of rows in the array.
  /// \param[in] cols_in Number of columns in the array.
  /// \param[in] device_in Number of elements in the array (Optional).
  Array2D(int rows_in, int cols_in, Device_T &device_in = get_device())
      : rows(rows_in), cols(cols_in), data_host(rows * cols), device(device_in),
        Q(device_in.get_queue()) {
    if (rows <= 0 || cols <= 0)
      throw std::runtime_error(
          "ERROR IN ARRAY2D: number of cols and rows must be > 0.");

    data_device = sycl::malloc_device<Scalar_T>(rows * cols, Q);
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Constructor that takes in a numpy array.
  /// \param[in] np_array_in Number of elements in the array.
  /// \param[in] device_in Number of elements in the array (Optional).
  Array2D(py::array_t<Scalar_T> np_array_in, Device_T &device_in = get_device())
      : device(device_in), Q(device_in.get_queue()) {
    if (np_array_in.ndim() != 2)
      throw std::runtime_error("The input numpy array must be 2D.");

    auto unchecked = np_array_in.template unchecked<2>();
    rows = unchecked.shape(0);
    cols = unchecked.shape(1);
    data_host.resize(rows * cols);

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        data_host[i * cols + j] = unchecked(i, j);
      }
    }

    data_device = sycl::malloc_device<Scalar_T>(rows * cols, Q);
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy constructor.
  Array2D(const Array2D &og)
      : rows(og.rows), cols(og.cols), data_host(og.data_host),
        device(og.device), Q(og.Q) {
    data_device = sycl::malloc_device<Scalar_T>(rows * cols, Q);
    Q.memcpy(data_device, og.data_device, rows * cols * sizeof(Scalar_T))
        .wait();
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Move constructor.
  Array2D(Array2D &&og) noexcept
      : rows(std::exchange(og.rows, 0)), cols(std::exchange(og.cols, 0)),
        data_host(std::move(og.data_host)),
        data_device(std::exchange(og.data_device, nullptr)), Q(og.Q),
        device(og.device) {}

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy assignment operator.
  Array2D &operator=(const Array2D &og) {
    data_host = og.data_host;
    sycl::free(data_device, Q);
    rows = og.rows;
    cols = og.cols;
    device = og.device;
    Q = og.Q;
    data_device = sycl::malloc_device<Scalar_T>(rows * cols, Q);
    Q.memcpy(data_device, og.data_device, rows * cols * sizeof(Scalar_T))
        .wait();

    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Move assignment operator.
  Array2D &operator=(Array2D &&og) noexcept {
    data_host = std::move(og.data_host);
    sycl::free(data_device, Q);
    data_device = std::exchange(og.data_device, nullptr);
    rows = std::exchange(og.rows, 0);
    cols = std::exchange(og.cols, 0);
    Q = og.Q;
    device = og.device;

    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Destructor.
  ~Array2D() {
    if (data_device) {
      sycl::free(data_device, Q);
    }

    data_device = nullptr;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator() direct element access
  Scalar_T &operator()(int i, int j) {
    if (i < 0 || i >= rows || j < 0 || j >= cols)
      throw std::out_of_range("Array2D access out of range");
    return data_host[i * cols + j];
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator() read-only element access
  const Scalar_T &operator()(int i, int j) const {
    if (i < 0 || i >= rows || j < 0 || j >= cols)
      throw std::out_of_range("Array2D access out of range");
    return data_host[i * cols + j];
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator+ for matrix addition (creating a new array).
  /// \param[in] B The Array2D that is being added.
  /// \return Array2D representing the sum of the addition.
  Array2D operator+(const Array2D &B) const {
    auto res = Array2D(rows, cols, this->device);
    binary_matrix_operations<BinaryOperation::ADD>(B, res);
    return res;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator+= for matrix addition (edits self).
  /// \param[in] B The Array2D that is being added to self.
  Array2D &operator+=(Array2D &B) {
    binary_matrix_operations<BinaryOperation::ADD>(B, *this);
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator- for matrix subtraction (creating a new
  /// array).
  /// \param[in] B The Array2D that is being subtracted.
  /// \return Array2D representing the difference of the subtraction.
  Array2D operator-(Array2D &B) {
    auto res = Array2D(rows, cols, this->device);
    binary_matrix_operations<BinaryOperation::SUBTRACT>(B, res);
    return res;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator-= for matrix subtraction (edits self).
  /// \param[in] B The Array2D that is being subtracted.
  Array2D &operator-=(Array2D &B) {
    binary_matrix_operations<BinaryOperation::SUBTRACT>(B, *this);
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator* for element-wise multiplication (creating a
  /// new array).
  /// \param[in] B The Array2D that is being multiplied.
  /// \return Array2D representing the product of the multiplication.
  Array2D operator*(Array2D &B) {
    auto res = Array2D(rows, cols, this->device);
    binary_matrix_operations<BinaryOperation::MULTIPLY>(B, res);
    return res;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator*= for element-wise multiplication (edits
  /// self).
  /// \param[in] B The Array2D that is being multiplied.
  Array2D &operator*=(Array2D &B) {
    binary_matrix_operations<BinaryOperation::MULTIPLY>(B, *this);
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator/ for element-wise division (creating a new
  /// array).
  /// \param[in] B The Array2D that is being divided.
  /// \return Array2D representing the result of the division.
  Array2D operator/(Array2D &B) {
    auto res = Array2D(rows, cols, this->device);
    binary_matrix_operations<BinaryOperation::DIVIDE>(B, res);
    return res;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator/= for element-wise division (edits self).
  /// \param[in] B The Array2D that is being divided.
  Array2D &operator/=(Array2D &B) {
    binary_matrix_operations<BinaryOperation::DIVIDE>(B, *this);
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the number of rows in the Array2D.
  /// \return Number of rows in the Array2D.
  int num_rows() const { return rows; }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the number of columns in the Array2D.
  /// \return Number of columns in the Array2D.
  int num_cols() const { return cols; }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the data pointer from Array2D.
  /// \return Pointer to Array2D host data.
  std::vector<Scalar_T> get_host_data_vector() { return data_host; }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the data pointer from Array2D.
  /// \return Pointer to Array2D device data.
  Scalar_T *get_device_data_ptr() const { return data_device; }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get platform index for the sycl device
  int get_platform_index() { return device.get_platform_index(); }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get platform index for the sycl device
  int get_device_index() { return device.get_device_index(); }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy memory from the CPU to the GPU.
  void mem_to_gpu() {
    Q.memcpy(data_device, &data_host[0], rows * cols * sizeof(Scalar_T)).wait();
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy memory from the GPU to the CPU
  void mem_to_cpu() {
    Q.memcpy(&data_host[0], data_device, rows * cols * sizeof(Scalar_T)).wait();
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Fill the device with a specific value
  void fill(const Scalar_T C) {
    const size_t B = sqrt(device.get_max_workgroup_size());

    Q.submit([&](sycl::handler &h) {
       const size_t global_size_rows = ((rows + B - 1) / B) * B;
       const size_t global_size_cols = ((cols + B - 1) / B) * B;
       const auto M = rows;
       const auto N = cols;
       const auto A = data_device;

       sycl::range<2> global{global_size_rows, global_size_cols};
       sycl::range<2> local{B, B};

       h.parallel_for(sycl::nd_range<2>(global, local),
                      [=](sycl::nd_item<2> it) {
                        const auto i = it.get_global_id(0);
                        const auto j = it.get_global_id(1);

                        if (i >= M || j >= N)
                          return;

                        A[i * N + j] = C;
                      });
     }).wait();
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Function that finds the maximum value in the array
  /// \return Maximum value in the array
  auto max() {
    return reductions(sycl::maximum<Scalar_T>(),
                      std::numeric_limits<Scalar_T>::lowest());
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Function that finds the minimum value in the array
  /// \return Minimum value in the array
  auto min() {
    return reductions(sycl::minimum<Scalar_T>(),
                      std::numeric_limits<Scalar_T>::max());
  };

  ///////////////////////////////////////////////////////////////////////
  /// \brief Function that finds the sum of all values in the array
  /// \return Sum of all values in the array
  auto sum() { return reductions(sycl::plus<Scalar_T>()); };

private:
  ///////////////////////////////////////////////////////////////////////
  /// \brief Number of rows in the array.
  int rows;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Number of columns in the array.
  int cols;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Vector for data stored in host memory.
  std::vector<Scalar_T> data_host;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Pointer to data stored in device memory.
  Scalar_T *data_device;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Device that will store and handle Array2D memory and operations
  Device_T &device;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Device SYCL queue.
  sycl::queue &Q;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Defining enumerations for binary operations
  enum class BinaryOperation { ADD, SUBTRACT, MULTIPLY, DIVIDE };

  ///////////////////////////////////////////////////////////////////////
  /// \brief Function to perform binary matrix operations
  template <BinaryOperation op>
  void binary_matrix_operations(const Array2D &B, Array2D &C) const {
    const auto rows = this->num_rows();
    const auto cols = this->num_cols();

    if (rows != B.num_rows() || cols != B.num_cols()) {
      throw std::runtime_error("ERROR: Incompatible Array2D dimensions.");
    }

    const auto platform_idx = this->device.get_platform_index();
    const auto device_idx = this->device.get_device_index();

    if (platform_idx != B.device.get_platform_index() ||
        device_idx != B.device.get_device_index()) {
      throw std::runtime_error("ERROR: Incompatible PySYCL device.");
    }

    const size_t wg_size = sqrt(this->device.get_max_workgroup_size());

    Q.submit([&](sycl::handler &h) {
       const size_t global_size_rows =
           ((rows + wg_size - 1) / wg_size) * wg_size;
       const size_t global_size_cols =
           ((cols + wg_size - 1) / wg_size) * wg_size;

       sycl::range<2> global{global_size_rows, global_size_cols};
       sycl::range<2> local{wg_size, wg_size};

       auto data_1 = this->get_device_data_ptr();
       auto data_2 = B.get_device_data_ptr();
       auto data_new = C.get_device_data_ptr();

       h.parallel_for(sycl::nd_range<2>(global, local),
                      [=](sycl::nd_item<2> it) {
                        const auto i = it.get_global_id(0);
                        const auto j = it.get_global_id(1);

                        if (i >= rows || j >= cols)
                          return;

                        if constexpr (op == BinaryOperation::ADD) {
                          data_new[i * cols + j] =
                              data_1[i * cols + j] + data_2[i * cols + j];
                        } else if constexpr (op == BinaryOperation::SUBTRACT) {
                          data_new[i * cols + j] =
                              data_1[i * cols + j] - data_2[i * cols + j];
                        } else if constexpr (op == BinaryOperation::MULTIPLY) {
                          data_new[i * cols + j] =
                              data_1[i * cols + j] * data_2[i * cols + j];
                        } else if constexpr (op == BinaryOperation::DIVIDE) {
                          data_new[i * cols + j] =
                              data_1[i * cols + j] / data_2[i * cols + j];
                        }
                      });
     }).wait();
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Function to perform reduction operations
  template <typename Operation_T>
  auto reductions(Operation_T op, Scalar_T val = 0.0) {
    sycl::buffer<Scalar_T> buf{&val, 1};

    const size_t wg_size = device.get_max_workgroup_size();

    Q.submit([&](sycl::handler &h) {
       const auto reduction_func =
           sycl::reduction(buf, h, std::forward<Operation_T>(op));

       const size_t global_size =
           ((rows * cols + wg_size - 1) / wg_size) * wg_size;
       sycl::range<1> global{global_size};
       sycl::range<1> local{wg_size};

       auto data_device_ptr = data_device;
       const size_t N = rows * cols;

       h.parallel_for(sycl::nd_range<1>(global, local), reduction_func,
                      [=](sycl::nd_item<1> it, auto &el) {
                        const auto idx = it.get_global_id();
                        if (idx >= N)
                          return;

                        el.combine(data_device_ptr[idx]);
                      });
     }).wait();

    sycl::host_accessor val_host{buf, sycl::read_only};
    val = val_host[0];

    return val;
  }
}; // class Array2D

/// @} // end "Array2D" doxygen group

} // namespace pysycl

#endif // ARRAY2D_H