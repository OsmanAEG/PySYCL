#include <CL/sycl.hpp>

namespace pysycl{
// parallel matrix multiplication
template<typename Queue_type, typename Scalar_type>
void parallel_matrix_multiplication(Queue_type Q, Scalar_type* A, Scalar_type* B,
                                    Scalar_type* C, size_t M, size_t N, size_t K){
  Q.submit([&](sycl::handler &h){
    h.parallel_for(sycl::range{M, K}, [=](sycl::id<2> idx){
      int i = idx[0];
      int j = idx[1];

      Scalar_type c_ij = 0.0;

      for(int p = 0; p < N; ++p){
        c_ij += A[i*N + p]*B[p*K + j];
      }
      C[i*K + j] = c_ij;
    });
  }).wait();
}

// nd-range parallel matrix multiplication
template<typename Queue_type, typename Scalar_type>
void nd_range_parallel_matrix_multiplication(Queue_type Q, Scalar_type* A, Scalar_type* B,
                                             Scalar_type* C, size_t M, size_t N, size_t K,
                                             size_t b){
  Q.submit([&](sycl::handler &h){
    // global nd range problem size
    sycl::range global{M, K};

    // local workgroup size
    sycl::range local{b, b};

    h.parallel_for(sycl::nd_range{global, local}, [=](sycl::nd_item<2> it){
      int i = it.get_global_id(0);
      int j = it.get_global_id(1);

      Scalar_type c_ij = 0.0;

      for(int p = 0; p < N; ++p){
        c_ij += A[i*N + p]*B[p*K + j];
      }
      C[i*K + j] = c_ij;
    });
  }).wait();
}

void matmul_handler(const size_t M, const size_t N, const size_t K){
  // establishing gpu for device queue
  sycl::queue Q{sycl::gpu_selector_v};

  // tolerance
  const double tol = 1.0E-6;

  // local work group size
  constexpr size_t b = 4;

  // matrices on host memory
  std::vector<double> A_host(M*N);
  std::vector<double> B_host(N*K);
  std::vector<double> C_host(M*K);

  // creating a random distribution
  std::default_random_engine generate(53);
  std::uniform_real_distribution<double> distribution(0.0, 2.0);

  auto random_number_generator = [&](){
    return distribution(generate);
  };

  // filling the input and output matrices on host
  std::generate(A_host.begin(), A_host.end(), random_number_generator);
  std::generate(B_host.begin(), B_host.end(), random_number_generator);
  std::fill(C_host.begin(), C_host.end(), 0.0);

  // allocating device memory
  double *A_device = sycl::malloc_shared<double>(M*N, Q);
  double *B_device = sycl::malloc_shared<double>(N*K, Q);
  double *C_device = sycl::malloc_shared<double>(M*K, Q);

  // copying host to device memory
  Q.memcpy(A_device, &A_host[0], M*N*sizeof(double));
  Q.memcpy(B_device, &B_host[0], N*K*sizeof(double));
  Q.memcpy(C_device, &C_host[0], M*K*sizeof(double));

  //parallel_matrix_multiplication(Q, A_device, B_device, C_device, M, N, K);
  nd_range_parallel_matrix_multiplication(Q, A_device, B_device, C_device, M, N, K, b);

  // copying device to host memory
  Q.memcpy(&C_host[0], C_device, M*K*sizeof(double));
}

}