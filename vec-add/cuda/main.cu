#include <cuda_runtime.h>
#include <cstdio>

template <typename scalar>
__global__ void vecAdd(const scalar* A, const scalar* B, scalar* C, size_t N) {
  /* Assuming 1D grid and 1D block*/
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}

template <typename scalar>
void hostVecRedAdd(const scalar* A, const scalar* B, scalar* C, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    C[i] = A[i] + B[i];
  }
}

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

template <typename scalar>
bool vectorApproximatelyEqual(scalar* A, scalar* B, int length, scalar epsilon=0.00001)
{
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] -B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

int main() {
  size_t N = 1<<10;
  float* h_A = new float[N];
  float* h_B = new float[N];
  float* h_C = new float[N];
  float* h_C_test = new float[N];

  float randmax = (float)RAND_MAX;
  for (size_t i = 0; i < N; ++i) {
    h_A[i] = rand() / randmax;
    h_B[i] = rand() / randmax;
  }

  float* d_A = initialize_device_vector(N, h_A);
  float* d_B = initialize_device_vector(N, h_B);
  float* d_C = initialize_device_vector(N, (float*)nullptr);

  size_t block_size = 256;
  size_t grid_size = (N + (block_size - 1)) / block_size;

  vecAdd<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
  cudaMemcpy(h_C, d_C, sizeof(float)*N, cudaMemcpyDeviceToHost);

  hostVecRedAdd(h_A, h_B, h_C_test, N);


  // Confirm that CPU and GPU got the same answer
  if(vectorApproximatelyEqual(h_C, h_C_test, N))
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
  delete[] h_C_test;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

