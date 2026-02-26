#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>

double getMicroSeconds()
{
  return omp_get_wtime() * 1000000.0;
}

void init_seed()
{
  srand((unsigned int) time(NULL));
}

static inline void cudaCheck(cudaError_t e, const char *msg)
{
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA ERROR (%s): %s\n", msg, cudaGetErrorString(e));
        exit(1);
    }
}

float **getmemory2D(int nx, int ny)
{
    float **buffer = (float**)malloc((size_t)nx * sizeof(float*));
    if (!buffer) return NULL;

    buffer[0] = (float*)malloc((size_t)nx * (size_t)ny * sizeof(float));
    if (!buffer[0]) {
        free(buffer);
        return NULL;
    }

    for (int i = 1; i < nx; i++)
        buffer[i] = buffer[i - 1] + ny;

    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            buffer[i][j] = 0.0f;

    return buffer;
}

float *getmemory1D(int n)
{
    float *buffer = (float*)malloc((size_t)n * sizeof(float));
    if (!buffer) return NULL;

    for (int i = 0; i < n; i++)
        buffer[i] = 0.0f;

    return buffer;
}

void init2Drand(float **buffer, int n)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            buffer[i][j] = 500.0f * ((float)rand() / (float)RAND_MAX) - 500.0f;
}

/* CPU reference */
void transpose1D_cpu(const float *in, float *out, int n)
{
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            out[(size_t)j * n + i] = in[(size_t)i * n + j];
}

int check(const float *GPU, const float *CPU, int n)
{
    for (int i = 0; i < n; i++)
        if (GPU[i] != CPU[i]) return 1;
    return 0;
}

#define TILE_DIM 32
#define BLOCK_ROWS 8

// MODIFICACIÓN: Renombrado a v3 por el Ejercicio 4 
__global__ void transpose_v3(float *in, float *out, int n) 
{
    // Ejercicio 4 - Añadido el +1 para evitar bank conflicts 
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // Coordenadas globales de lectura
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Cargar tile desde memoria global a shared
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n && (y + j) < n) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * n + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x; 
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Escribir tile transpuesto a memoria global
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n && (y + j) < n) {
            out[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}


/* ================= MAIN ================= */

int main(int argc, char **argv)
{
    int n;

    if (argc == 2) n = atoi(argv[1]);
    else {
        n = 8192;
        printf("./exec n (by default n=%i)\n", n);
    }

    init_seed();

    float **array2D       = getmemory2D(n, n);
    float **array2D_trans = getmemory2D(n, n);
    float *array1D_trans_GPU = getmemory1D(n * n);

    if (!array2D || !array2D_trans || !array1D_trans_GPU) return 1;

    float *array1D       = array2D[0];
    float *array1D_trans = array2D_trans[0];

    init2Drand(array2D, n);

    /* CPU reference */
    double bytes = 2.0 * (double)n * (double)n * (double)sizeof(float);

    double t0 = getMicroSeconds();
    transpose1D_cpu(array1D, array1D_trans, n);
    double t1 = getMicroSeconds();
    double secCPU = (t1 - t0) / 1e6;

    printf("Transpose CPU: %f MB/s\n\n",
           (bytes / secCPU) / 1024.0 / 1024.0);

    float *d_in  = NULL;
    float *d_out = NULL;

    cudaMalloc((void**)&d_in, n*n*sizeof(float));    
    cudaMalloc((void**)&d_out, n*n*sizeof(float));    

    // MODIFICACIÓN: Ejercicio 5 - Medición separada del tiempo H2D
    double tH2D_0 = getMicroSeconds();
    cudaMemcpy(d_in, array1D, n*n*sizeof(float), cudaMemcpyHostToDevice);
    double tH2D_1 = getMicroSeconds();
    double secH2D = (tH2D_1 - tH2D_0) / 1e6;

    // Bloque de 32x8 hilos
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS); 
    // Grid basado en TILE_DIM (32x32)
    dim3 dimGrid((n + TILE_DIM - 1) / TILE_DIM, (n + TILE_DIM - 1) / TILE_DIM);

    // Ejercicio 5 - Medición separada del tiempo Kernel 
    double tKernel0 = getMicroSeconds();
    transpose_v3 <<<dimGrid, dimBlock>>> (d_in, d_out, n); 
    cudaDeviceSynchronize();
    double tKernel1 = getMicroSeconds();
    double secGPU = (tKernel1 - tKernel0) / 1e6;

    // Ejercicio 5 - Medición separada del tiempo D2H 
    double tD2H_0 = getMicroSeconds();
    cudaMemcpy(array1D_trans_GPU, d_out, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    double tD2H_1 = getMicroSeconds();
    double secD2H = (tD2H_1 - tD2H_0) / 1e6;

    // Impresión de resultados separados 
    printf("Resultados de Tiempos CUDA:\n");
    printf("  -> Tiempo Transferencia H2D: %f segundos\n", secH2D);
    printf("  -> Tiempo Ejecucion Kernel:  %f segundos\n", secGPU);
    printf("  -> Tiempo Transferencia D2H: %f segundos\n", secD2H);
    
    double secTotalCUDA = secH2D + secGPU + secD2H;
    printf("\n  -> Tiempo Total (H2D + Kernel + D2H): %f segundos\n", secTotalCUDA);
    
    printf("\nTranspose GPU (Solo Kernel): %f MB/s\n", (bytes / secGPU) / 1024.0 / 1024.0);
    

    if (check(array1D_trans_GPU, array1D_trans, n * n))
        printf("Transpose CPU-GPU differs!!\n");
    else
        printf("Check OK\n");

    cudaFree(d_in);         cudaFree(d_out);
    free(array2D[0]);       free(array2D);
    free(array2D_trans[0]); free(array2D_trans);
    free(array1D_trans_GPU);

    return 0;
}