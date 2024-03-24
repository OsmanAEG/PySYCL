#ifndef VECTOR_FACTORIES_H
#define VECTOR_FACTORIES_H

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
/// \brief PySYCL Vector Factories.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
/// stl
///////////////////////////////////////////////////////////////////////
#include <tuple>
#include <variant>

///////////////////////////////////////////////////////////////////////
/// local
///////////////////////////////////////////////////////////////////////
#include "../Data_Types/Data_Types.h"
#include "../Device/Device_Instance.h"
#include "../Device/Device_Manager.h"
#include "Vector_Type.h"

namespace py = pybind11;

using Device_T = pysycl::Device_Instance;
using Data_T = pysycl::Data_Types;

///////////////////////////////////////////////////////////////////////
/// \addtogroup Vector
/// @{
namespace pysycl {

using Vector_Variants = std::variant<Vector<double>,
                                     Vector<float>,
                                     Vector<int>>;

///////////////////////////////////////////////////////////////////////
/// \brief Function factory for Vector Types.
Vector_Variants
vector_factories(int dims, Device_Instance &device, Data_Types& dtype) {
  if(dtype == Data_Types::DOUBLE) {
    return Vector<double>(dims, device);
  } else if(dtype == Data_Types::FLOAT) {
    return Vector<float>(dims, device);
  } else if (dtype == Data_Types::INT) {
    return Vector<int>(dims, device);
  } else {
    throw std::runtime_error("ERROR IN VECTOR: Unsupported datatype.");
  }
}

/// @} // end "Vector" doxygen group

} // namespace pysycl

#endif // VECTOR_FACTORIES_H
