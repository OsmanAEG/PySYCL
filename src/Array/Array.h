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
#include <variant>

///////////////////////////////////////////////////////////////////////
/// local
///////////////////////////////////////////////////////////////////////
#include "../Array1D/Array1D.h"
#include "../Array2D/Array2D.h"
#include "../Device/Device_Instance.h"
#include "../Device/Device_Manager.h"

///////////////////////////////////////////////////////////////////////
/// \addtogroup Array
/// @{
namespace pysycl {

///////////////////////////////////////////////////////////////////////
/// \brief Function to handle the selection of a PySYCL Array type
template<typename Scalar_type>
std::variant<Array1D<Scalar_type>, Array2D<Scalar_type>>
array_selector(const std::vector<int>& dims, Device_Instance& device = get_device()){
  if(dims.size() == 1){
    return Array1D<Scalar_type>(dims[0], device);
  }else if(dims.size() == 2){
    return Array2D<Scalar_type>(dims[0], dims[1], device);
  }else{
    throw std::runtime_error("ERROR IN ARRAY: Number of dimensions cannot exceed 2.");
  }
}

/// @} // end "Array" doxygen group

} // namespace pysycl

#endif // ARRAY_H

