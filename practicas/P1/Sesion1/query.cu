#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int ndev;
    cudaGetDeviceCount(&ndev);
    for (int d = 0; d < ndev; ++d) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, d);
        printf("Device %d: %s\n", d, p.name);
        printf("    Compute Capability: %d.%d\n", p.major, p.minor);
        printf("    Number of SMs: %d\n", p.multiProcessorCount);
        printf("    Warp size: %d\n", p.warpSize);
        printf("    Global Mem: %.2f GB", p.totalGlobalMem / (1024.0 *1024.0 *1024.0));
        printf("    Shared Mem/Block: %zu KB\n", p.sharedMemPerBlock / 1024);
        printf("    Regs/Block: %d\n", p.regsPerBlock);
        printf("    Max Threads/Block: %d\n", p.maxThreadsPerBlock);
        printf("    Max Threads/Multi-Processor: %d\n", p.maxThreadsPerMultiProcessor);
        printf("    Concurrent Kernels: %d\n", p.concurrentKernels);
        printf("    L2 Cache: %d KB\n", p.l2CacheSize / 1024);
        printf("    Max Grid: [%d, %d, %d]\n", p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
    }
    return 0;
}