#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

static constexpr size_t N = 1<<24;
static constexpr size_t block_size = 256;
static constexpr size_t grid_size = (N + (block_size - 1)) / block_size;

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
__global__ void vecRedAdd_treeBased(const scalar* vec, scalar* sum, size_t N) {
  // assumes block_size is power of 2
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ float partial_sums[block_size];
  partial_sums[threadIdx.x] = (idx < N) ? vec[idx] : 0;
  __syncthreads();

  size_t stride = blockDim.x / 2;
  while (stride > 0) {
    if (threadIdx.x < stride) {
      partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
    }
    __syncthreads();
    stride >>= 1;
  }

  // 1 atomic add per block
  if (threadIdx.x == 0) atomicAdd(sum, partial_sums[0]);
}

template <typename scalar>
void hostVecRedAdd(const scalar* vec, scalar* sum, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    *sum += vec[i];
  }
}

template <typename scalar>
void wrapKernel(
    void(*func)(const scalar*, scalar*, size_t),
    const scalar* dvec, size_t N, const scalar reference) {

  float* hsum = new float[1];
  *hsum = 0.0f;
  float* dsum = initialize_device_vector(1, hsum);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  (*func)<<<grid_size, block_size>>>(dvec, dsum, N);
  cudaEventRecord(stop);
  cudaMemcpy(hsum, dsum, (sizeof(float))*1, cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);
  float ms_device = 0.0f;
  cudaEventElapsedTime(&ms_device, start, stop);
  printf("%-10s %5.5f %-10s %5.5f\n", "Result:", *hsum, "Time [ms]:", ms_device);

  // Confirm that CPU and GPU got the same answer
  float reldiff = fabs(*hsum - reference) / reference;
  float reldiff_tol = 1e-4;
  if (reldiff < reldiff_tol)
  {
      printf("CPU and GPU answers match within relative tolerance of %e\n\n", reldiff_tol);
  }
  else
  {
      printf("Error - CPU and GPU answers do not match\n");
  }
  delete[] hsum;
  cudaFree(dsum);
}

int main() {
  printf("Running reduction of %zu floats\n", N);
  float* hvec = new float[N];

  /* Random floats between 0-1*/
  float randmax = (float)RAND_MAX;
  for (size_t i = 0; i < N; ++i) {
    hvec[i] = rand() / randmax;
  }
  float* dvec = initialize_device_vector(N, hvec);

  float hsum_test = 0.0f;
  auto t0 = std::chrono::high_resolution_clock::now();
  hostVecRedAdd(hvec, &hsum_test, N);
  auto t1 = std::chrono::high_resolution_clock::now();
  double host_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  printf("Host\n");
  printf("====\n");
  printf("%-10s %5.5f %-10s %5.5f\n\n", "Result:", hsum_test, "Time [ms]:", host_ms);

  printf("Kernel vecRedAdd_1atomicPerThread\n");
  printf("=================================\n");
  wrapKernel(vecRedAdd_1atomicPerThread, dvec, N, hsum_test);

  printf("Kernel vecRedAdd_treeBased\n");
  printf("============================\n");
  wrapKernel(vecRedAdd_treeBased, dvec, N, hsum_test);

  delete[] hvec;

  cudaFree(dvec);
}

