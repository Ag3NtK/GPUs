#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int ndev;
    cudaGetDeviceCount(&ndev);
    
    for (int d = 0; d < ndev; ++d) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, d);
        
        printf("Device %d: %s\n", d, p.name);
        printf("  Compute Capability: %d.%d\n", p.major, p.minor);
        printf("  SMs: %d\n", p.multiProcessorCount);
        printf("  Warp Size: %d\n", p.warpSize);
        printf("  Global Mem: %.2f GB\n", (float)p.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
        printf("  Shared Mem/Block: %zu KB\n", p.sharedMemPerBlock / 1024);
        printf("  Regs/Block: %d\n", p.regsPerBlock);
        printf("  Max Threads/Block: %d\n", p.maxThreadsPerBlock);
        
        // --- EXTRAS REQUERIDOS ---
        printf("  L2 Cache: %d KB\n", p.l2CacheSize / 1024);
        printf("  Max Grid: [%d, %d, %d]\n", p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);

        // --- SOLUCIÓN PARA CLOCKS Y POTENCIA ---
        // Usamos variables más genéricas para evitar errores de versión
        printf("  Memory Bus Width: %d bits\n", p.memoryBusWidth);
        
        // Para la potencia, como no está en cudaDeviceProp, usamos el comando directo.
        // Esto es lo que te piden en la Parte A e inspección dinámica.
        printf("\n  -- Snapshot de Consumo (nvidia-smi) --\n  ");
        fflush(stdout); 
        system("nvidia-smi --query-gpu=power.draw,power.limit --format=csv,noheader");
        printf("--------------------------------------------------\n");
    }
    return 0;
}