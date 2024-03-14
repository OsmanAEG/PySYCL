#ifndef ARRAY_H
#define ARRAY_H

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
/// \brief Array in PySYCL.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
/// stl
///////////////////////////////////////////////////////////////////////
#include <tuple>
#include <variant>

///////////////////////////////////////////////////////////////////////
/// local
///////////////////////////////////////////////////////////////////////
#include "../Array1D/Array1D.h"
#include "../Array2D/Array2D.h"
#include "../Data_Types/Data_Types.h"
#include "../Device/Device_Instance.h"
#include "../Device/Device_Manager.h"

///////////////////////////////////////////////////////////////////////
/// \addtogroup Array
/// @{
namespace pysycl {

using Array1D_Variants = std::variant<Array1D<double>,
                                      Array1D<float>,
                                      Array1D<int>>;

using Array2D_Variants = std::variant<Array2D<double>,
                                      Array2D<float>,
                                      Array2D<int>>;

///////////////////////////////////////////////////////////////////////
/// \brief Function specialization for Array1D.
Array1D_Variants
array_selector(int dims, Device_Instance &device, Data_Types& dtype) {
  if(dtype == Data_Types::DOUBLE) {
    return Array1D<double>(dims, device);
  } else if(dtype == Data_Types::FLOAT) {
    return Array1D<float>(dims, device);
  } else if (dtype == Data_Types::INT) {
    return Array1D<int>(dims, device);
  } else {
    throw std::runtime_error("ERROR IN ARRAY: Unsupported datatype.");
  }
}

///////////////////////////////////////////////////////////////////////
/// \brief Function specialization for Array2D.
Array2D_Variants
array_selector(std::tuple<int, int> dims, Device_Instance &device, Data_Types& dtype) {
  if(dtype == Data_Types::DOUBLE) {
    return Array2D<double>(std::get<0>(dims), std::get<1>(dims), device);
  } else if(dtype == Data_Types::FLOAT) {
    return Array2D<float>(std::get<0>(dims), std::get<1>(dims), device);
  } else if (dtype == Data_Types::INT) {
    return Array2D<int>(std::get<0>(dims), std::get<1>(dims), device);
  } else {
    throw std::runtime_error("ERROR IN ARRAY: Unsupported datatype.");
  }
}

/// @} // end "Array" doxygen group

} // namespace pysycl

#endif // ARRAY_H
