#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

static constexpr size_t N = 1<<24;
static constexpr size_t warp_size = 32;
static constexpr size_t block_size = 1024;
static constexpr size_t grid_size = (N + (block_size - 1)) / block_size;
constexpr dim3 block_dim(warp_size, block_size/warp_size);
constexpr dim3 grid_dim(grid_size);

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
  /* Threads of each block perform an atomicAdd on a shared float, then first thread
   * of block sums its resul */
  // blcok dim is (32, block_size / 32)
  size_t warp_idx = threadIdx.y;
  size_t thread_idx = threadIdx.x;
  size_t block_thread_idx = warp_idx * warp_size + thread_idx;
  size_t global_thread_idx = (blockDim.x * blockDim.y) * blockIdx.x + block_thread_idx;

  __shared__ float blockSum;

  if (block_thread_idx) blockSum = 0.0f;

  __syncthreads();

  if (global_thread_idx < N) {
    atomicAdd(&blockSum, vec[global_thread_idx]);
  }

  __syncthreads();

  // 1 atomic add per block
  if (block_thread_idx == 0) {
    atomicAdd(sum, blockSum);
  }
}

template <typename scalar>
__global__ void vecRedAdd_treeBased(const scalar* vec, scalar* sum, size_t N) {
  // assumes block_size is power of 2
  size_t warp_idx = threadIdx.y; // in block
  size_t thread_idx_in_block = warp_idx * warp_size + threadIdx.x;
  size_t global_thread_idx = block_size * blockIdx.x + thread_idx_in_block;

  __shared__ scalar partial_sums[block_size];
  partial_sums[thread_idx_in_block] = (global_thread_idx < N) ? vec[global_thread_idx] : 0;
  __syncthreads();

  size_t stride = block_size / 2;
  while (stride > 0) {
    if (thread_idx_in_block < stride) {
      partial_sums[thread_idx_in_block] += partial_sums[thread_idx_in_block + stride];
    }
    __syncthreads();
    stride >>= 1;
  }

  // 1 atomic add per block
  if (thread_idx_in_block == 0) atomicAdd(sum, partial_sums[0]);
}

template <typename scalar>
__global__ void vecRedAdd_intraWarpRegOps(const scalar* vec, scalar* sum, size_t N) {
  /* 1d grid, 2D block
   * blockDim.x = # threads per warp
   * blockDim.y = # warps per block */
  size_t thread_idx_in_block = threadIdx.y * blockDim.x + threadIdx.x;
  size_t global_thread_idx = (blockDim.x * blockDim.y) * blockIdx.x + thread_idx_in_block;

  __shared__ scalar warp_sums[32];

  // TODO 1: warp_sums[0] = reduction of 0-31
  //         warp_sums[1] = reduction of 32-63
  //         ...
  // TODO 2: The first index of each warp adds its warp result
  // if (threadIdx.x % 32 == 0) atomicAdd(sum, partial_sums[0]);
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
  (*func)<<<grid_dim, block_dim>>>(dvec, dsum, N);
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
