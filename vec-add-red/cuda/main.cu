#include <cuda_runtime.h>
#include <cstdio>

template <typename scalar>
__global__ void vecRedAdd(const scalar* vec, scalar* sum, size_t N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ float blockSum;
  blockSum = 0.0f;

  if (idx < N) {
    atomicAdd(&blockSum, vec[idx]);
  }

  __syncthreads();

  // 1 atomic add per block
  if (idx == 0) {
    atomicAdd(sum, atomicAdd(&blockSum, 0.0f));
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

  vecRedAdd<<<grid_size, block_size>>>(dvec, dsum, N);
  cudaMemcpy(hsum, dsum, (sizeof(float))*1, cudaMemcpyDeviceToHost);
  printf("Device reduction equals %f\n", *hsum);

  float hsum_test = 0.0f;
  hostVecRedAdd(hvec, &hsum_test, N);
  printf("Host reduction equals %f\n", hsum_test);


  // Confirm that CPU and GPU got the same answer
  float diff = fabs(*hsum - hsum_test);
  float eps = 1e-5f;
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

