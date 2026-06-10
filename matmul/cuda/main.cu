#include <cuda_runtime.h>
#include <cstdio>

constexpr size_t warp_size = 32;

template <typename scalar>
scalar* initialize_device_vector(size_t N, const scalar* h_vec) {
  scalar* d_vec = nullptr;
  size_t numbytes = N*sizeof(scalar);
  cudaMalloc(&d_vec, numbytes);
  if (h_vec != nullptr) {
    cudaMemcpy(d_vec, h_vec, numbytes, cudaMemcpyHostToDevice);
  }
  return d_vec;
}

void hostMatMul(size_t N,
  const int* A, const int* B, int* C) {
  /* Assuming row major */
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      size_t idx = i*N + j;
      C[idx] = 0;
      for (size_t k = 0; k < N; ++k) {
        C[idx] += A[N*i + k] * B[N*k + j];
      }
    }
  }
}

__global__ void deviceNaiveMatMul(size_t N,
  const int* A, const int* B, int* C) {
  /*
   * Assuming row major
   * threadIdx.x is threadIdx in warp
   * threadIdx.y is warpIdx in block
   */
  // global indices
  size_t warp_idx = blockIdx.x * blockDim.y + threadIdx.y;
  size_t thread_idx = blockIdx.x * blockDim.x * blockDim.y
    + threadIdx.y * blockDim.x + threadIdx.x;

  size_t row = thread_idx / N;
  size_t col = thread_idx - (row*N);

  if (thread_idx < N*N) {
    C[thread_idx] = 0;
    for (size_t k = 0; k < N; ++k) {
      C[thread_idx] += A[row*N + k] * B[k*N + col];
    }
  }
}

__global__ void deviceSharedMemMatMul(size_t N,
  const int* A, const int* B, int* C) {
  /*
   * Assuming row major
   * threadIdx.x is threadIdx in warp
   * threadIdx.y is warpIdx in block
   */
  // global indices
  size_t warp_idx = blockIdx.x * blockDim.y + threadIdx.y;
  size_t thread_idx = blockIdx.x * blockDim.x * blockDim.y
    + threadIdx.y * blockDim.x + threadIdx.x;

  size_t row = thread_idx / N;
  size_t col = thread_idx - (row*N);

  __shared__ int bA[warp_size][warp_size];
  __shared__ int bB[warp_size][warp_size];

  size_t numBlocks = (N  + (warp_size - 1))/ warp_size;

  int Cij = 0;
  for (size_t k = 0; k < numBlocks; ++k) {
    // Load mem
    bA[threadIdx.x][threadIdx.y] = A[row * N + (k * warp_size) + threadIdx.y];
    bB[threadIdx.x][threadIdx.y] = B[((k * warp_size) + threadIdx.x) * N + col];
    __syncthreads();
    // block mul
    for (size_t w = 0; w < warp_size; ++w) {
      Cij += bA[threadIdx.x][w] * bB[w][threadIdx.y];
    }
  }
  C[row*N + col] = Cij;
}


template <typename scalar>
bool vectorsEqual(scalar* A, scalar* B, int length)
{
    for(int i=0; i<length; i++)
    {
        if(A[i] != B[i])
        {
            printf("Index %d mismatch: %d != %d", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

int main() {
  size_t N = 1<<10;

  // matrices of ints
  int *h_A, *h_B, *h_C, *h_C_ref;
  h_A = new int[N*N];
  h_B = new int[N*N];
  h_C = new int[N*N];
  h_C_ref = new int[N*N];
  float randmax = (float)RAND_MAX;
  for (size_t i = 0; i < N*N; ++i) {
    h_A[i] = rand() / randmax;
    h_B[i] = rand() / randmax;
  }

  int* d_A = initialize_device_vector(N*N, h_A);
  int* d_B = initialize_device_vector(N*N, h_B);
  int* d_C = initialize_device_vector(N*N, (int*)nullptr);

  // host ref
  hostMatMul(N, h_A, h_B, h_C_ref);

  // device
  size_t block_size = 1024;
  dim3 blockDim(warp_size, block_size / warp_size);
  size_t grid_size = (N*N + (block_size - 1)) / block_size;
  dim3 gridDim(grid_size);

  // deviceNaiveMatMul<<<gridDim, blockDim>>>(N, d_A, d_B, d_C);
  deviceSharedMemMatMul<<<gridDim, blockDim>>>(N, d_A, d_B, d_C);
  cudaMemcpy(h_C, d_C, N*N*sizeof(int), cudaMemcpyDeviceToHost);

  // Confirm that CPU and GPU got the same answer
  if(vectorsEqual(h_C, h_C_ref, N*N))
  {
      printf("CPU and GPU answers match\n");
  }
  else
  {
      printf("Error - CPU and GPU answers do not match\n");
  }



  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  delete[] h_C_ref;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
