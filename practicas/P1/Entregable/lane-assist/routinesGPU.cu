#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "routinesGPU.h"

#define DEG2RAD 0.017453f

// Dimensiones de bloque y configuración para Shared Memory en imágenes
#define BLOCK_SIZE 16
#define FILTER_RADIUS 2
#define TILE_SIZE (BLOCK_SIZE + 2 * FILTER_RADIUS)

// =========================================================================
// KERNEL 1: Reducción de Ruido con Shared Memory (Imágenes)
// =========================================================================
__global__ void noise_reduction_shmem_kernel(uint8_t *imBW, float *NR, int height, int width) {
    __shared__ uint8_t tile[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    
    int tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y;
    int tile_elements = TILE_SIZE * TILE_SIZE;

    // Carga colaborativa a Shared Memory (incluye el halo/padding)
    for (int i = tid; i < tile_elements; i += num_threads) {
        int tile_r = i / TILE_SIZE;
        int tile_c = i % TILE_SIZE;
        int global_r = by + tile_r - FILTER_RADIUS;
        int global_c = bx + tile_c - FILTER_RADIUS;

        if (global_r >= 0 && global_r < height && global_c >= 0 && global_c < width) {
            tile[tile_r][tile_c] = imBW[global_r * width + global_c];
        } else {
            tile[tile_r][tile_c] = 0;
        }
    }

    __syncthreads();

    int col = bx + tx;
    int row = by + ty;

    if (col < width && row < height) {
        int lt = ty + FILTER_RADIUS;
        int lc = tx + FILTER_RADIUS;

        float sum = 
            (2.0f * tile[lt-2][lc-2] + 4.0f * tile[lt-2][lc-1] + 5.0f * tile[lt-2][lc] + 4.0f * tile[lt-2][lc+1] + 2.0f * tile[lt-2][lc+2] +
             4.0f * tile[lt-1][lc-2] + 9.0f * tile[lt-1][lc-1] + 12.0f * tile[lt-1][lc] + 9.0f * tile[lt-1][lc+1] + 4.0f * tile[lt-1][lc+2] +
             5.0f * tile[lt][lc-2]   + 12.0f * tile[lt][lc-1]   + 15.0f * tile[lt][lc]   + 12.0f * tile[lt][lc+1]   + 5.0f * tile[lt][lc+2]   +
             4.0f * tile[lt+1][lc-2] + 9.0f * tile[lt+1][lc-1] + 12.0f * tile[lt+1][lc] + 9.0f * tile[lt+1][lc+1] + 4.0f * tile[lt+1][lc+2] +
             2.0f * tile[lt+2][lc-2] + 4.0f * tile[lt+2][lc-1] + 5.0f * tile[lt+2][lc] + 4.0f * tile[lt+2][lc+1] + 2.0f * tile[lt+2][lc+2]) / 159.0f;

        if (col >= 2 && col < width - 2 && row >= 2 && row < height - 2) {
            NR[row * width + col] = sum;
        }
    }
}

// =========================================================================
// KERNEL 2: Gradiente con Shared Memory (Imágenes)
// =========================================================================
__global__ void gradient_shmem_kernel(float *NR, float *G, float *phi, int height, int width) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    
    int tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y;
    int tile_elements = TILE_SIZE * TILE_SIZE;

    // Carga colaborativa a Shared Memory
    for (int i = tid; i < tile_elements; i += num_threads) {
        int tile_r = i / TILE_SIZE;
        int tile_c = i % TILE_SIZE;
        int global_r = by + tile_r - FILTER_RADIUS;
        int global_c = bx + tile_c - FILTER_RADIUS;

        if (global_r >= 0 && global_r < height && global_c >= 0 && global_c < width) {
            tile[tile_r][tile_c] = NR[global_r * width + global_c];
        } else {
            tile[tile_r][tile_c] = 0.0f;
        }
    }

    __syncthreads();

    int col = bx + tx;
    int row = by + ty;

    if (col >= 2 && col < width - 2 && row >= 2 && row < height - 2) {
        int lt = ty + FILTER_RADIUS;
        int lc = tx + FILTER_RADIUS;

        float Gx_val = (1.0f * tile[lt-2][lc-2] + 2.0f * tile[lt-2][lc-1] + (-2.0f) * tile[lt-2][lc+1] + (-1.0f) * tile[lt-2][lc+2]
                 + 4.0f * tile[lt-1][lc-2] + 8.0f * tile[lt-1][lc-1] + (-8.0f) * tile[lt-1][lc+1] + (-4.0f) * tile[lt-1][lc+2]
                 + 6.0f * tile[lt][lc-2]   + 12.0f * tile[lt][lc-1]   + (-12.0f) * tile[lt][lc+1] + (-6.0f) * tile[lt][lc+2]
                 + 4.0f * tile[lt+1][lc-2] + 8.0f * tile[lt+1][lc-1] + (-8.0f) * tile[lt+1][lc+1] + (-4.0f) * tile[lt+1][lc+2]
                 + 1.0f * tile[lt+2][lc-2] + 2.0f * tile[lt+2][lc-1] + (-2.0f) * tile[lt+2][lc+1] + (-1.0f) * tile[lt+2][lc+2]);

        float Gy_val = ((-1.0f) * tile[lt-2][lc-2] + (-4.0f) * tile[lt-2][lc-1] + (-6.0f) * tile[lt-2][lc] + (-4.0f) * tile[lt-2][lc+1] + (-1.0f) * tile[lt-2][lc+2]
                 + (-2.0f) * tile[lt-1][lc-2] + (-8.0f) * tile[lt-1][lc-1] + (-12.0f) * tile[lt-1][lc] + (-8.0f) * tile[lt-1][lc+1] + (-2.0f) * tile[lt-1][lc+2]
                 + 2.0f * tile[lt+1][lc-2]  + 8.0f * tile[lt+1][lc-1]  + 12.0f * tile[lt+1][lc]  + 8.0f * tile[lt+1][lc+1]  + 2.0f * tile[lt+1][lc+2]
                 + 1.0f * tile[lt+2][lc-2]  + 4.0f * tile[lt+2][lc-1]  + 6.0f * tile[lt+2][lc]  + 4.0f * tile[lt+2][lc+1]  + 1.0f * tile[lt+2][lc+2]);

        int idx = row * width + col;
        G[idx] = sqrtf(Gx_val * Gx_val + Gy_val * Gy_val);
        
        float p = atan2f(Gy_val, Gx_val);
        float PI = 3.141593f;
        if (p < 0.0f) p += PI; 

        if (p <= PI / 8.0f || p > 15.0f * (PI / 8.0f)) phi[idx] = 0;
        else if (p <= 3.0f * (PI / 8.0f)) phi[idx] = 45;
        else if (p <= 5.0f * (PI / 8.0f)) phi[idx] = 90;
        else if (p <= 7.0f * (PI / 8.0f)) phi[idx] = 135;
        else phi[idx] = 0;
    }
}

// =========================================================================
// KERNELS 3 & 4: Supresión e Histéresis (Lectura directa 1-a-1)
// =========================================================================
__global__ void edge_kernel_2d(float *G, float *phi, uint8_t *pedge, int height, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= 3 && col < width - 3 && row >= 3 && row < height - 3) {
        int idx = row * width + col;
        pedge[idx] = 0;
        float angle = phi[idx];
        float val = G[idx];

        if (angle == 0) {
            if (val > G[idx + 1] && val > G[idx - 1]) pedge[idx] = 1;
        } else if (angle == 45) {
            if (val > G[(row+1)*width + col + 1] && val > G[(row-1)*width + col - 1]) pedge[idx] = 1;
        } else if (angle == 90) {
            if (val > G[(row+1)*width + col] && val > G[(row-1)*width + col]) pedge[idx] = 1;
        } else if (angle == 135) {
            if (val > G[(row+1)*width + col - 1] && val > G[(row-1)*width + col + 1]) pedge[idx] = 1;
        }
    }
}

__global__ void hysteresis_kernel_2d(float *G, uint8_t *pedge, uint8_t *imEdge, float level, int height, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= 3 && col < width - 3 && row >= 3 && row < height - 3) {
        int idx = row * width + col;
        float lowthres = level / 2.0f;
        float hithres = 2.0f * level;
        
        imEdge[idx] = 0;
        if (G[idx] > hithres && pedge[idx]) {
            imEdge[idx] = 255;
        } else if (pedge[idx] && G[idx] >= lowthres && G[idx] < hithres) {
            for (int ii = -1; ii <= 1; ii++)
                for (int jj = -1; jj <= 1; jj++)
                    if (G[(row + ii) * width + col + jj] > hithres)
                        imEdge[idx] = 255;
        }
    }
}

// =========================================================================
// KERNEL 5: Transformada de Hough con Shared Memory para tablas LUT
// =========================================================================
__global__ void hough_kernel_2d_shared_tables(uint8_t *imEdge, uint32_t *accum, int accu_width, int accu_height, float *d_sin_table, float *d_cos_table, int height, int width) {
    
    // Memoria compartida para las tablas trigonométricas
    __shared__ float s_sin_table[180];
    __shared__ float s_cos_table[180];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;

    int tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y;

    // Carga colaborativa de las tablas (180 elementos) desde la memoria global
    for (int i = tid; i < 180; i += num_threads) {
        s_sin_table[i] = d_sin_table[i];
        s_cos_table[i] = d_cos_table[i];
    }

    __syncthreads(); // Esperar a que las tablas estén listas para todos los hilos

    if (col < width && row < height) {
        int idx = row * width + col;
        if (imEdge[idx] > 250) { 
            float hough_h = ((sqrtf(2.0f) * (float)(height > width ? height : width)) / 2.0f);
            float center_x = width / 2.0f;
            float center_y = height / 2.0f;
            
            for (int theta = 0; theta < 180; theta++) {
                // Leemos directamente desde memoria compartida local
                float rho = ((float)col - center_x) * s_cos_table[theta] + ((float)row - center_y) * s_sin_table[theta];
                int rho_idx = (int)roundf(rho + hough_h);
                
                if (rho_idx >= 0 && rho_idx < accu_height) {
                    int acc_idx = rho_idx * accu_width + theta;
                    atomicAdd(&accum[acc_idx], 1); 
                }
            }
        }
    }
}

// =========================================================================
// HOST FUNCTION
// =========================================================================
void lane_assist_GPU(uint8_t *im, int height, int width, int *x1, int *y1, int *x2, int *y2, int *nlines)
{
    cudaEvent_t start_h2d, stop_h2d, start_kernels, stop_kernels, start_d2h, stop_d2h;
    cudaEventCreate(&start_h2d); cudaEventCreate(&stop_h2d);
    cudaEventCreate(&start_kernels); cudaEventCreate(&stop_kernels);
    cudaEventCreate(&start_d2h); cudaEventCreate(&stop_d2h);
    float time_h2d = 0, time_kernels = 0, time_d2h = 0;

    // Reservar memoria 
    uint8_t *d_imBW, *d_imEdge, *d_pedge;
    float *d_NR, *d_G, *d_phi;
    uint32_t *d_accum;
    float *d_sin_table, *d_cos_table;

    cudaMalloc(&d_imBW, height * width * sizeof(uint8_t));
    cudaMalloc(&d_NR, height * width * sizeof(float));
    cudaMalloc(&d_G, height * width * sizeof(float));
    cudaMalloc(&d_phi, height * width * sizeof(float));
    cudaMalloc(&d_pedge, height * width * sizeof(uint8_t));
    cudaMalloc(&d_imEdge, height * width * sizeof(uint8_t));
    
    cudaMemset(d_imEdge, 0, height * width * sizeof(uint8_t));

    int accu_width = 180;
    float hough_h = ((sqrtf(2.0f) * (float)(height > width ? height : width)) / 2.0f);
    int accu_height = (int)(hough_h * 2.0f);
    cudaMalloc(&d_accum, accu_width * accu_height * sizeof(uint32_t));

    // Reserva global normal para las tablas
    cudaMalloc(&d_sin_table, 180 * sizeof(float));
    cudaMalloc(&d_cos_table, 180 * sizeof(float));

    // Generar tablas en el Host
    float h_sin[180], h_cos[180];
    for (int i = 0; i < 180; i++) {
        h_sin[i] = sinf(i * DEG2RAD);
        h_cos[i] = cosf(i * DEG2RAD);
    }

    // Transferencias H2D
    cudaEventRecord(start_h2d);
    cudaMemcpy(d_imBW, im, height * width * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin_table, h_sin, 180 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos_table, h_cos, 180 * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop_h2d);
    cudaEventSynchronize(stop_h2d);
    cudaEventElapsedTime(&time_h2d, start_h2d, stop_h2d);

    // lanzamiento de Kernels
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    cudaEventRecord(start_kernels);
    
    noise_reduction_shmem_kernel<<<grid, block>>>(d_imBW, d_NR, height, width);
    gradient_shmem_kernel<<<grid, block>>>(d_NR, d_G, d_phi, height, width);
    edge_kernel_2d<<<grid, block>>>(d_G, d_phi, d_pedge, height, width);
    hysteresis_kernel_2d<<<grid, block>>>(d_G, d_pedge, d_imEdge, 1000.0f, height, width);
    
    cudaMemset(d_accum, 0, accu_width * accu_height * sizeof(uint32_t));
    // Pasamos d_sin_table y d_cos_table como parámetros
    hough_kernel_2d_shared_tables<<<grid, block>>>(d_imEdge, d_accum, accu_width, accu_height, d_sin_table, d_cos_table, height, width);
    
    cudaEventRecord(stop_kernels);
    cudaEventSynchronize(stop_kernels);
    cudaEventElapsedTime(&time_kernels, start_kernels, stop_kernels);

    // D2H
    cudaEventRecord(start_d2h);
    uint32_t *h_accum = (uint32_t *)malloc(accu_width * accu_height * sizeof(uint32_t));
    cudaMemcpy(h_accum, d_accum, accu_width * accu_height * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_d2h);
    cudaEventSynchronize(stop_d2h);
    cudaEventElapsedTime(&time_d2h, start_d2h, stop_d2h);

    // 5. Extracción de líneas en Host
    int threshold = (width > height) ? width / 6 : height / 6;
    *nlines = 0;
    for (int rho = 0; rho < accu_height; rho++) {
        for (int theta = 0; theta < accu_width; theta++) {
            if (h_accum[rho * accu_width + theta] >= threshold) {
                uint32_t max_val = h_accum[rho * accu_width + theta];
                for (int ii = -4; ii <= 4; ii++) {
                    for (int jj = -4; jj <= 4; jj++) {
                        if ((ii + rho >= 0 && ii + rho < accu_height) && (jj + theta >= 0 && jj + theta < accu_width)) {
                            if (h_accum[(rho + ii) * accu_width + (theta + jj)] > max_val) {
                                max_val = h_accum[(rho + ii) * accu_width + (theta + jj)];
                            }
                        }
                    }
                }
                if (max_val == h_accum[rho * accu_width + theta]) {
                    int x1l, y1l, x2l, y2l;
                    if (theta >= 45 && theta <= 135) {
                        if (theta > 90) {
                            x1l = width / 2;
                            y1l = ((float)(rho - (accu_height / 2)) - ((x1l - (width / 2)) * h_cos[theta])) / h_sin[theta] + (height / 2);
                            x2l = width;
                            y2l = ((float)(rho - (accu_height / 2)) - ((x2l - (width / 2)) * h_cos[theta])) / h_sin[theta] + (height / 2);
                        } else {
                            x1l = 0;
                            y1l = ((float)(rho - (accu_height / 2)) - ((x1l - (width / 2)) * h_cos[theta])) / h_sin[theta] + (height / 2);
                            x2l = width * 2 / 5;
                            y2l = ((float)(rho - (accu_height / 2)) - ((x2l - (width / 2)) * h_cos[theta])) / h_sin[theta] + (height / 2);
                        }
                    } else {
                        y1l = 0;
                        x1l = ((float)(rho - (accu_height / 2)) - ((y1l - (height / 2)) * h_sin[theta])) / h_cos[theta] + (width / 2);
                        y2l = height;
                        x2l = ((float)(rho - (accu_height / 2)) - ((y2l - (height / 2)) * h_sin[theta])) / h_cos[theta] + (width / 2);
                    }
                    x1[*nlines] = x1l; y1[*nlines] = y1l;
                    x2[*nlines] = x2l; y2[*nlines] = y2l;
                    (*nlines)++;
                    if (*nlines >= 10) break; 
                }
            }
        }
        if (*nlines >= 10) break;
    }

    printf("=== GPU Timing Results (100%% Shared Mem) ===\n");
    printf("Transferencias H2D:     %.3f ms\n", time_h2d);
    printf("Kernels:                %.3f ms\n", time_kernels);
    printf("Transferencias D2H:     %.3f ms\n", time_d2h);
    printf("Total GPU:              %.3f ms\n", time_h2d + time_kernels + time_d2h);
    printf("=============================================\n");

    free(h_accum);
    cudaFree(d_imBW); cudaFree(d_NR); cudaFree(d_G); cudaFree(d_phi);
    cudaFree(d_pedge); cudaFree(d_imEdge); cudaFree(d_accum); 
    cudaFree(d_sin_table); cudaFree(d_cos_table);

    cudaEventDestroy(start_h2d); cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_kernels); cudaEventDestroy(stop_kernels);
    cudaEventDestroy(start_d2h); cudaEventDestroy(stop_d2h);
}
