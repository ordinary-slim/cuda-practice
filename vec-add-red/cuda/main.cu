#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

template <typename scalar>
__global__ void vecRedAdd_1atomicPerThread(const scalar* vec, scalar* sum, size_t N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ float blockSum;
  if (threadIdx.x == 0) {
    blockSum = 0.0f;
  }
  __syncthreads();

  if (idx < N) {
    atomicAdd(&blockSum, vec[idx]);
  }

  __syncthreads();

  // 1 atomic add per block
  if (threadIdx.x == 0) {
    atomicAdd(sum, blockSum);
  }
}

template <typename scalar>
void hostVecRedAdd(const scalar* vec, scalar* sum, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    *sum += vec[i];
  }
}

template <typename scalar>
scalar* initialize_device_vector(size_t N, const scalar* h_vec) {
  /* Allocate device vec and copy vector from host to device*/
  scalar* d_vec = nullptr;
  size_t numbytes = N*sizeof(scalar);
  cudaMalloc(&d_vec, numbytes);
  if (h_vec != nullptr) {
    cudaMemcpy(d_vec, h_vec, numbytes, cudaMemcpyHostToDevice);
  }
  return d_vec;
}

int main() {
  size_t N = 1<<10;
  float* hvec = new float[N];

  /* Random floats between 0-1*/
  float randmax = (float)RAND_MAX;
  for (size_t i = 0; i < N; ++i) {
    hvec[i] = rand() / randmax;
  }
  float* dvec = initialize_device_vector(N, hvec);

  size_t block_size = 256;
  size_t grid_size = (N + (block_size - 1)) / block_size;

  float* hsum = new float[1];
  *hsum = 0.0f;
  float* dsum = initialize_device_vector(1, hsum);

  vecRedAdd_1atomicPerThread<<<grid_size, block_size>>>(dvec, dsum, N);
  cudaMemcpy(hsum, dsum, (sizeof(float))*1, cudaMemcpyDeviceToHost);
  printf("%-70s %f\n", "Device reduction using vecRedAdd_1atomicPerThread kernel equals", *hsum);

  float hsum_test = 0.0f;
  auto t0 = std::chrono::high_resolution_clock::now();
  hostVecRedAdd(hvec, &hsum_test, N);
  auto t1 = std::chrono::high_resolution_clock::now();
  double host_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  printf("%-70s %5.5f %-10s %5.5f\n", "Host reduction ", hsum_test, "Time (ms)", host_ms);


  // Confirm that CPU and GPU got the same answer
  float diff = fabs(*hsum - hsum_test);
  float eps = 1e-3f;
  if (diff < eps)
  {
      printf("CPU and GPU answers match\n");
  }
  else
  {
      printf("Error - CPU and GPU answers do not match\n");
  }

  delete[] hvec;
  delete[] hsum;

  cudaFree(dvec);
  cudaFree(dsum);
}

