#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_kernel()
{
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main()
{
    printf("Launching kernel...\n");

    hello_kernel<<<1, 8>>>();

    // Check launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    // Check execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Kernel execution failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    printf("Kernel completed successfully.\n");

    return 0;
}
