
/*
//------------------------------------------------------V0--------------------------------------------------


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "routinesGPU.h"

#define DEG2RAD 0.017453f


__global__ void rgb2bw_kernel(uint8_t *im, uint8_t *imBW, int height, int width) {
    int col = blockIdx.x;
    if (col >= width) return;
    for (int row = 0; row < height; row++) {
        int idx = row * width + col;
        float R = (float)im[3 * idx];
        float G = (float)im[3 * idx + 1];
        float B = (float)im[3 * idx + 2];
        imBW[idx] = (uint8_t)(0.2989f * R + 0.5870f * G + 0.1140f * B);
    }
}

__global__ void noise_reduction_kernel(uint8_t *imBW, float *NR, int height, int width) {
    int col = blockIdx.x;
    if (col < 2 || col >= width - 2) return;
    for (int row = 2; row < height - 2; row++) {
        int idx = row * width + col;
        NR[idx] = (2.0f * imBW[(row-2)*width + (col-2)] + 4.0f * imBW[(row-2)*width + (col-1)] + 5.0f * imBW[(row-2)*width + col] + 4.0f * imBW[(row-2)*width + (col+1)] + 2.0f * imBW[(row-2)*width + (col+2)]
                 + 4.0f * imBW[(row-1)*width + (col-2)] + 9.0f * imBW[(row-1)*width + (col-1)] + 12.0f * imBW[(row-1)*width + col] + 9.0f * imBW[(row-1)*width + (col+1)] + 4.0f * imBW[(row-1)*width + (col+2)]
                 + 5.0f * imBW[row*width + (col-2)] + 12.0f * imBW[row*width + (col-1)] + 15.0f * imBW[row*width + col] + 12.0f * imBW[row*width + (col+1)] + 5.0f * imBW[row*width + (col+2)]
                 + 4.0f * imBW[(row+1)*width + (col-2)] + 9.0f * imBW[(row+1)*width + (col-1)] + 12.0f * imBW[(row+1)*width + col] + 9.0f * imBW[(row+1)*width + (col+1)] + 4.0f * imBW[(row+1)*width + (col+2)]
                 + 2.0f * imBW[(row+2)*width + (col-2)] + 4.0f * imBW[(row+2)*width + (col-1)] + 5.0f * imBW[(row+2)*width + col] + 4.0f * imBW[(row+2)*width + (col+1)] + 2.0f * imBW[(row+2)*width + (col+2)]) / 159.0f;
    }
}

__global__ void gradient_kernel(float *NR, float *G, float *phi, float *Gx, float *Gy, int height, int width) {
    int col = blockIdx.x;
    if (col < 2 || col >= width - 2) return;
    for (int row = 2; row < height - 2; row++) {
        int idx = row * width + col;
        Gx[idx] = (1.0f * NR[(row-2)*width + (col-2)] + 2.0f * NR[(row-2)*width + (col-1)] + (-2.0f) * NR[(row-2)*width + (col+1)] + (-1.0f) * NR[(row-2)*width + (col+2)]
                 + 4.0f * NR[(row-1)*width + (col-2)] + 8.0f * NR[(row-1)*width + (col-1)] + (-8.0f) * NR[(row-1)*width + (col+1)] + (-4.0f) * NR[(row-1)*width + (col+2)]
                 + 6.0f * NR[row*width + (col-2)] + 12.0f * NR[row*width + (col-1)] + (-12.0f) * NR[row*width + (col+1)] + (-6.0f) * NR[row*width + (col+2)]
                 + 4.0f * NR[(row+1)*width + (col-2)] + 8.0f * NR[(row+1)*width + (col-1)] + (-8.0f) * NR[(row+1)*width + (col+1)] + (-4.0f) * NR[(row+1)*width + (col+2)]
                 + 1.0f * NR[(row+2)*width + (col-2)] + 2.0f * NR[(row+2)*width + (col-1)] + (-2.0f) * NR[(row+2)*width + (col+1)] + (-1.0f) * NR[(row+2)*width + (col+2)]);

        Gy[idx] = ((-1.0f) * NR[(row-2)*width + (col-2)] + (-4.0f) * NR[(row-2)*width + (col-1)] + (-6.0f) * NR[(row-2)*width + col] + (-4.0f) * NR[(row-2)*width + (col+1)] + (-1.0f) * NR[(row-2)*width + (col+2)]
                 + (-2.0f) * NR[(row-1)*width + (col-2)] + (-8.0f) * NR[(row-1)*width + (col-1)] + (-12.0f) * NR[(row-1)*width + col] + (-8.0f) * NR[(row-1)*width + (col+1)] + (-2.0f) * NR[(row-1)*width + (col+2)]
                 + 2.0f * NR[(row+1)*width + (col-2)] + 8.0f * NR[(row+1)*width + (col-1)] + 12.0f * NR[(row+1)*width + col] + 8.0f * NR[(row+1)*width + (col+1)] + 2.0f * NR[(row+1)*width + (col+2)]
                 + 1.0f * NR[(row+2)*width + (col-2)] + 4.0f * NR[(row+2)*width + (col-1)] + 6.0f * NR[(row+2)*width + col] + 4.0f * NR[(row+2)*width + (col+1)] + 1.0f * NR[(row+2)*width + (col+2)]);

        G[idx] = sqrtf(Gx[idx] * Gx[idx] + Gy[idx] * Gy[idx]);
        phi[idx] = atan2f(fabsf(Gy[idx]), fabsf(Gx[idx]));

        float PI = 3.141593f;
        if (fabsf(phi[idx]) <= PI / 8)
            phi[idx] = 0;
        else if (fabsf(phi[idx]) <= 3 * (PI / 8))
            phi[idx] = 45;
        else if (fabsf(phi[idx]) <= 5 * (PI / 8))
            phi[idx] = 90;
        else if (fabsf(phi[idx]) <= 7 * (PI / 8))
            phi[idx] = 135;
        else
            phi[idx] = 0;
    }
}

__global__ void edge_kernel(float *G, float *phi, uint8_t *pedge, int height, int width) {
    int col = blockIdx.x;
    if (col < 3 || col >= width - 3) return;
    for (int row = 3; row < height - 3; row++) {
        int idx = row * width + col;
        pedge[idx] = 0;
        if (phi[idx] == 0) {
            if (G[idx] > G[idx + 1] && G[idx] > G[idx - 1])
                pedge[idx] = 1;
        } else if (phi[idx] == 45) {
            if (G[idx] > G[(row+1)*width + col + 1] && G[idx] > G[(row-1)*width + col - 1])
                pedge[idx] = 1;
        } else if (phi[idx] == 90) {
            if (G[idx] > G[(row+1)*width + col] && G[idx] > G[(row-1)*width + col])
                pedge[idx] = 1;
        } else if (phi[idx] == 135) {
            if (G[idx] > G[(row+1)*width + col - 1] && G[idx] > G[(row-1)*width + col + 1])
                pedge[idx] = 1;
        }
    }
}

__global__ void hysteresis_kernel(float *G, uint8_t *pedge, uint8_t *imEdge, float level, int height, int width) {
    int col = blockIdx.x;
    if (col < 3 || col >= width - 3) return;
    float lowthres = level / 2;
    float hithres = 2 * level;
    for (int row = 3; row < height - 3; row++) {
        int idx = row * width + col;
        imEdge[idx] = 0;
        if (G[idx] > hithres && pedge[idx])
            imEdge[idx] = 255;
        else if (pedge[idx] && G[idx] >= lowthres && G[idx] < hithres) {
            // check 3x3 neighbors
            for (int ii = -1; ii <= 1; ii++)
                for (int jj = -1; jj <= 1; jj++)
                    if (G[(row + ii) * width + col + jj] > hithres)
                        imEdge[idx] = 255;
        }
    }
}

__global__ void hough_kernel(uint8_t *imEdge, uint32_t *accum, int accu_width, int accu_height, float *sin_table, float *cos_table, int height, int width) {
    int col = blockIdx.x;
    if (col >= width) return;
    float hough_h = ((sqrtf(2.0f) * (float)(height > width ? height : width)) / 2.0f);
    float center_x = width / 2.0f;
    float center_y = height / 2.0f;
    for (int row = 0; row < height; row++) {
        int idx = row * width + col;
        if (imEdge[idx] > 250) {
            for (int theta = 0; theta < 180; theta++) {
                float rho = ((float)col - center_x) * cos_table[theta] + ((float)row - center_y) * sin_table[theta];
                int rho_idx = (int)roundf(rho + hough_h);
                int acc_idx = rho_idx * 180 + theta;
                atomicAdd(&accum[acc_idx], 1);
            }
        }
    }
}

void lane_assist_GPU(uint8_t *im, int height, int width,
	int *x1, int *y1, int *x2, int *y2, int *nlines)
{
    // TIMING
    cudaEvent_t start_h2d, stop_h2d, start_kernels, stop_kernels, start_d2h, stop_d2h;
    cudaEventCreate(&start_h2d);
    cudaEventCreate(&stop_h2d);
    cudaEventCreate(&start_kernels);
    cudaEventCreate(&stop_kernels);
    cudaEventCreate(&start_d2h);
    cudaEventCreate(&stop_d2h);
    float time_h2d = 0.0f, time_kernels = 0.0f, time_d2h = 0.0f;

    // Allocate device memory
    uint8_t *d_imBW, *d_imEdge, *d_pedge;
    float *d_NR, *d_G, *d_phi, *d_Gx, *d_Gy;
    uint32_t *d_accum;
    float *d_sin_table, *d_cos_table;

    cudaMalloc(&d_imBW, height * width * sizeof(uint8_t));
    cudaMalloc(&d_NR, height * width * sizeof(float));
    cudaMalloc(&d_G, height * width * sizeof(float));
    cudaMalloc(&d_phi, height * width * sizeof(float));
    cudaMalloc(&d_Gx, height * width * sizeof(float));
    cudaMalloc(&d_Gy, height * width * sizeof(float));
    cudaMalloc(&d_pedge, height * width * sizeof(uint8_t));
    cudaMalloc(&d_imEdge, height * width * sizeof(uint8_t));

    int accu_width = 180;
    float hough_h = ((sqrtf(2.0f) * (float)(height > width ? height : width)) / 2.0f);
    int accu_height = (int)(hough_h * 2.0f);
    cudaMalloc(&d_accum, accu_width * accu_height * sizeof(uint32_t));

    cudaMalloc(&d_sin_table, 180 * sizeof(float));
    cudaMalloc(&d_cos_table, 180 * sizeof(float));

    // Transfer H2D: input image and sin/cos tables
    cudaEventRecord(start_h2d, 0);
    
    cudaMemcpy(d_imBW, im, height * width * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Init sin cos tables
    float h_sin[180], h_cos[180];
    for (int i = 0; i < 180; i++) {
        h_sin[i] = sinf(i * DEG2RAD);
        h_cos[i] = cosf(i * DEG2RAD);
    }
    cudaMemcpy(d_sin_table, h_sin, 180 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos_table, h_cos, 180 * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(stop_h2d, 0);
    cudaEventSynchronize(stop_h2d);
    cudaEventElapsedTime(&time_h2d, start_h2d, stop_h2d);

    // Kernels
    dim3 grid(width);
    dim3 block(1);

    cudaEventRecord(start_kernels, 0);

    noise_reduction_kernel<<<grid, block>>>(d_imBW, d_NR, height, width);
    cudaDeviceSynchronize();

    gradient_kernel<<<grid, block>>>(d_NR, d_G, d_phi, d_Gx, d_Gy, height, width);
    cudaDeviceSynchronize();

    edge_kernel<<<grid, block>>>(d_G, d_phi, d_pedge, height, width);
    cudaDeviceSynchronize();

    hysteresis_kernel<<<grid, block>>>(d_G, d_pedge, d_imEdge, 1000.0f, height, width);
    cudaDeviceSynchronize();

    cudaMemset(d_accum, 0, accu_width * accu_height * sizeof(uint32_t));

    hough_kernel<<<grid, block>>>(d_imEdge, d_accum, accu_width, accu_height, d_sin_table, d_cos_table, height, width);
    cudaDeviceSynchronize();

    cudaEventRecord(stop_kernels, 0);
    cudaEventSynchronize(stop_kernels);
    cudaEventElapsedTime(&time_kernels, start_kernels, stop_kernels);

    // Transfer D2H: accumulator
    cudaEventRecord(start_d2h, 0);

    uint32_t *h_accum = (uint32_t *)malloc(accu_width * accu_height * sizeof(uint32_t));
    cudaMemcpy(h_accum, d_accum, accu_width * accu_height * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_d2h, 0);
    cudaEventSynchronize(stop_d2h);
    cudaEventElapsedTime(&time_d2h, start_d2h, stop_d2h);

    int threshold = (width > height) ? width / 6 : height / 6;
    *nlines = 0;
    // Implement getlines
    for (int rho = 0; rho < accu_height; rho++) {
        for (int theta = 0; theta < accu_width; theta++) {
            if (h_accum[rho * accu_width + theta] >= threshold) {
                uint32_t max = h_accum[rho * accu_width + theta];
                for (int ii = -4; ii <= 4; ii++) {
                    for (int jj = -4; jj <= 4; jj++) {
                        if ((ii + rho >= 0 && ii + rho < accu_height) && (jj + theta >= 0 && jj + theta < accu_width)) {
                            if (h_accum[(rho + ii) * accu_width + (theta + jj)] > max) {
                                max = h_accum[(rho + ii) * accu_width + (theta + jj)];
                            }
                        }
                    }
                }
                if (max == h_accum[rho * accu_width + theta]) {
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
                    x1[*nlines] = x1l;
                    y1[*nlines] = y1l;
                    x2[*nlines] = x2l;
                    y2[*nlines] = y2l;
                    (*nlines)++;
                }
            }
        }
    }

    free(h_accum);

    // Print timing information
    float total_time = time_h2d + time_kernels + time_d2h;
    printf("=== GPU Timing Results ===\n");
    printf("Transferencias H2D/D2H: %.3f ms\n", time_h2d);
    printf("Kernels:                %.3f ms\n", time_kernels);
    printf("Transferencias D2H:     %.3f ms\n", time_d2h);
    printf("Total GPU:              %.3f ms\n", total_time);
    printf("==========================\n");

    // Free device memory
    cudaFree(d_imBW);
    cudaFree(d_NR);
    cudaFree(d_G);
    cudaFree(d_phi);
    cudaFree(d_Gx);
    cudaFree(d_Gy);
    cudaFree(d_pedge);
    cudaFree(d_imEdge);
    cudaFree(d_accum);
    cudaFree(d_sin_table);
    cudaFree(d_cos_table);

    // Destroy timing events
    cudaEventDestroy(start_h2d);
    cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_kernels);
    cudaEventDestroy(stop_kernels);
    cudaEventDestroy(start_d2h);
    cudaEventDestroy(stop_d2h);
}
*/







































//--------------------------------v1---------------------------------


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "routinesGPU.h"

#define DEG2RAD 0.017453f

// --- KERNELS BIDIMENSIONALES (v1) ---
// Kernel para conversión a escala de grises (2D)
__global__ void rgb2bw_kernel_2d(uint8_t *im, uint8_t *imBW, int height, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int idx = row * width + col;
        float R = (float)im[3 * idx];
        float G = (float)im[3 * idx + 1];
        float B = (float)im[3 * idx + 2];
        imBW[idx] = (uint8_t)(0.2989f * R + 0.5870f * G + 0.1140f * B);
    }
}

// Kernel para reducción de ruido - Filtro Gaussiano 5x5 (2D)
__global__ void noise_reduction_kernel_2d(uint8_t *imBW, float *NR, int height, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Margen de 2 píxeles para el filtro 5x5
    if (col >= 2 && col < width - 2 && row >= 2 && row < height - 2) {
        int idx = row * width + col;
        NR[idx] = (2.0f * imBW[(row-2)*width + (col-2)] + 4.0f * imBW[(row-2)*width + (col-1)] + 5.0f * imBW[(row-2)*width + col] + 4.0f * imBW[(row-2)*width + (col+1)] + 2.0f * imBW[(row-2)*width + (col+2)]
                 + 4.0f * imBW[(row-1)*width + (col-2)] + 9.0f * imBW[(row-1)*width + (col-1)] + 12.0f * imBW[(row-1)*width + col] + 9.0f * imBW[(row-1)*width + (col+1)] + 4.0f * imBW[(row-1)*width + (col+2)]
                 + 5.0f * imBW[row*width + (col-2)] + 12.0f * imBW[row*width + (col-1)] + 15.0f * imBW[row*width + col] + 12.0f * imBW[row*width + (col+1)] + 5.0f * imBW[row*width + (col+2)]
                 + 4.0f * imBW[(row+1)*width + (col-2)] + 9.0f * imBW[(row+1)*width + (col-1)] + 12.0f * imBW[(row+1)*width + col] + 9.0f * imBW[(row+1)*width + (col+1)] + 4.0f * imBW[(row+1)*width + (col+2)]
                 + 2.0f * imBW[(row+2)*width + (col-2)] + 4.0f * imBW[(row+2)*width + (col-1)] + 5.0f * imBW[(row+2)*width + col] + 4.0f * imBW[(row+2)*width + (col+1)] + 2.0f * imBW[(row+2)*width + (col+2)]) / 159.0f;
    }
}

// Kernel para cálculo de gradiente y dirección (2D)
__global__ void gradient_kernel_2d(float *NR, float *G, float *phi, float *Gx, float *Gy, int height, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= 2 && col < width - 2 && row >= 2 && row < height - 2) {
        int idx = row * width + col;
        Gx[idx] = (1.0f * NR[(row-2)*width + (col-2)] + 2.0f * NR[(row-2)*width + (col-1)] + (-2.0f) * NR[(row-2)*width + (col+1)] + (-1.0f) * NR[(row-2)*width + (col+2)]
                 + 4.0f * NR[(row-1)*width + (col-2)] + 8.0f * NR[(row-1)*width + (col-1)] + (-8.0f) * NR[(row-1)*width + (col+1)] + (-4.0f) * NR[(row-1)*width + (col+2)]
                 + 6.0f * NR[row*width + (col-2)] + 12.0f * NR[row*width + (col-1)] + (-12.0f) * NR[row*width + (col+1)] + (-6.0f) * NR[row*width + (col+2)]
                 + 4.0f * NR[(row+1)*width + (col-2)] + 8.0f * NR[(row+1)*width + (col-1)] + (-8.0f) * NR[(row+1)*width + (col+1)] + (-4.0f) * NR[(row+1)*width + (col+2)]
                 + 1.0f * NR[(row+2)*width + (col-2)] + 2.0f * NR[(row+2)*width + (col-1)] + (-2.0f) * NR[(row+2)*width + (col+1)] + (-1.0f) * NR[(row+2)*width + (col+2)]);

        Gy[idx] = ((-1.0f) * NR[(row-2)*width + (col-2)] + (-4.0f) * NR[(row-2)*width + (col-1)] + (-6.0f) * NR[(row-2)*width + col] + (-4.0f) * NR[(row-2)*width + (col+1)] + (-1.0f) * NR[(row-2)*width + (col+2)]
                 + (-2.0f) * NR[(row-1)*width + (col-2)] + (-8.0f) * NR[(row-1)*width + (col-1)] + (-12.0f) * NR[(row-1)*width + col] + (-8.0f) * NR[(row-1)*width + (col+1)] + (-2.0f) * NR[(row-1)*width + (col+2)]
                 + 2.0f * NR[(row+1)*width + (col-2)] + 8.0f * NR[(row+1)*width + (col-1)] + 12.0f * NR[(row+1)*width + col] + 8.0f * NR[(row+1)*width + (col+1)] + 2.0f * NR[(row+1)*width + (col+2)]
                 + 1.0f * NR[(row+2)*width + (col-2)] + 4.0f * NR[(row+2)*width + (col-1)] + 6.0f * NR[(row+2)*width + col] + 4.0f * NR[(row+2)*width + (col+1)] + 1.0f * NR[(row+2)*width + (col+2)]);

        G[idx] = sqrtf(Gx[idx] * Gx[idx] + Gy[idx] * Gy[idx]);
        float p = atan2f(fabsf(Gy[idx]), fabsf(Gx[idx]));

        float PI = 3.141593f;
        if (p <= PI / 8.0f) phi[idx] = 0;
        else if (p <= 3.0f * (PI / 8.0f)) phi[idx] = 45;
        else if (p <= 5.0f * (PI / 8.0f)) phi[idx] = 90;
        else if (p <= 7.0f * (PI / 8.0f)) phi[idx] = 135;
        else phi[idx] = 0;
    }
}

// Kernel para supresión de no-máximos (2D)
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

// Kernel para histéresis (2D)
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
            // Comprobar vecinos 3x3
            for (int ii = -1; ii <= 1; ii++)
                for (int jj = -1; jj <= 1; jj++)
                    if (G[(row + ii) * width + col + jj] > hithres)
                        imEdge[idx] = 255;
        }
    }
}

// Kernel para Transformada de Hough con Votación Atómica (2D)
__global__ void hough_kernel_2d(uint8_t *imEdge, uint32_t *accum, int accu_width, int accu_height, float *sin_table, float *cos_table, int height, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int idx = row * width + col;
        if (imEdge[idx] > 250) { // El píxel es un borde
            float hough_h = ((sqrtf(2.0f) * (float)(height > width ? height : width)) / 2.0f);
            float center_x = width / 2.0f;
            float center_y = height / 2.0f;
            
            for (int theta = 0; theta < 180; theta++) {
                float rho = ((float)col - center_x) * cos_table[theta] + ((float)row - center_y) * sin_table[theta];
                int rho_idx = (int)roundf(rho + hough_h);
                int acc_idx = rho_idx * 180 + theta;
                atomicAdd(&accum[acc_idx], 1);
            }
        }
    }
}

void lane_assist_GPU(uint8_t *im, int height, int width,
	int *x1, int *y1, int *x2, int *y2, int *nlines)
{
    // EVENTOS PARA MEDIDAS DE TIEMPO
    cudaEvent_t start_h2d, stop_h2d, start_kernels, stop_kernels, start_d2h, stop_d2h;
    cudaEventCreate(&start_h2d); cudaEventCreate(&stop_h2d);
    cudaEventCreate(&start_kernels); cudaEventCreate(&stop_kernels);
    cudaEventCreate(&start_d2h); cudaEventCreate(&stop_d2h);
    
    float time_h2d = 0, time_kernels = 0, time_d2h = 0;

    // ALOCACIÓN DE MEMORIA EN DISPOSITIVO
    uint8_t *d_imBW, *d_imEdge, *d_pedge;
    float *d_NR, *d_G, *d_phi, *d_Gx, *d_Gy;
    uint32_t *d_accum;
    float *d_sin_table, *d_cos_table;

    cudaMalloc(&d_imBW, height * width * sizeof(uint8_t));
    cudaMalloc(&d_NR, height * width * sizeof(float));
    cudaMalloc(&d_G, height * width * sizeof(float));
    cudaMalloc(&d_phi, height * width * sizeof(float));
    cudaMalloc(&d_Gx, height * width * sizeof(float));
    cudaMalloc(&d_Gy, height * width * sizeof(float));
    cudaMalloc(&d_pedge, height * width * sizeof(uint8_t));
    cudaMalloc(&d_imEdge, height * width * sizeof(uint8_t));

    int accu_width = 180;
    float hough_h = ((sqrtf(2.0f) * (float)(height > width ? height : width)) / 2.0f);
    int accu_height = (int)(hough_h * 2.0f);
    cudaMalloc(&d_accum, accu_width * accu_height * sizeof(uint32_t));
    cudaMalloc(&d_sin_table, 180 * sizeof(float));
    cudaMalloc(&d_cos_table, 180 * sizeof(float));

    // TRANSFERENCIAS H2D
    cudaEventRecord(start_h2d);
    cudaMemcpy(d_imBW, im, height * width * sizeof(uint8_t), cudaMemcpyHostToDevice);
    
    float h_sin[180], h_cos[180];
    for (int i = 0; i < 180; i++) {
        h_sin[i] = sinf(i * DEG2RAD);
        h_cos[i] = cosf(i * DEG2RAD);
    }
    cudaMemcpy(d_sin_table, h_sin, 180 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos_table, h_cos, 180 * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop_h2d);
    cudaEventSynchronize(stop_h2d);
    cudaEventElapsedTime(&time_h2d, start_h2d, stop_h2d);

    // CONFIGURACIÓN DE EJECUCIÓN 2D (Bloques de 16x16)
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    cudaEventRecord(start_kernels);
    
    // Ejecución de pipeline de detección
    noise_reduction_kernel_2d<<<grid, block>>>(d_imBW, d_NR, height, width);
    gradient_kernel_2d<<<grid, block>>>(d_NR, d_G, d_phi, d_Gx, d_Gy, height, width);
    edge_kernel_2d<<<grid, block>>>(d_G, d_phi, d_pedge, height, width);
    hysteresis_kernel_2d<<<grid, block>>>(d_G, d_pedge, d_imEdge, 1000.0f, height, width);
    
    // Hough
    cudaMemset(d_accum, 0, accu_width * accu_height * sizeof(uint32_t));
    hough_kernel_2d<<<grid, block>>>(d_imEdge, d_accum, accu_width, accu_height, d_sin_table, d_cos_table, height, width);
    
    cudaEventRecord(stop_kernels);
    cudaEventSynchronize(stop_kernels);
    cudaEventElapsedTime(&time_kernels, start_kernels, stop_kernels);

    // TRANSFERENCIAS D2H
    cudaEventRecord(start_d2h);
    uint32_t *h_accum = (uint32_t *)malloc(accu_width * accu_height * sizeof(uint32_t));
    cudaMemcpy(h_accum, d_accum, accu_width * accu_height * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_d2h);
    cudaEventSynchronize(stop_d2h);
    cudaEventElapsedTime(&time_d2h, start_d2h, stop_d2h);

    // EXTRACCIÓN DE LÍNEAS (Lógica de Host)
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

    // Informe de tiempos
    printf("=== GPU Timing Results (v1 2D) ===\n");
    printf("Transferencias H2D:     %.3f ms\n", time_h2d);
    printf("Kernels:                %.3f ms\n", time_kernels);
    printf("Transferencias D2H:     %.3f ms\n", time_d2h);
    printf("Total GPU:              %.3f ms\n", time_h2d + time_kernels + time_d2h);
    printf("==================================\n");

    // Limpieza
    free(h_accum);
    cudaFree(d_imBW); cudaFree(d_NR); cudaFree(d_G); cudaFree(d_phi);
    cudaFree(d_Gx); cudaFree(d_Gy); cudaFree(d_pedge); cudaFree(d_imEdge);
    cudaFree(d_accum); cudaFree(d_sin_table); cudaFree(d_cos_table);
    
    cudaEventDestroy(start_h2d); cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_kernels); cudaEventDestroy(stop_kernels);
    cudaEventDestroy(start_d2h); cudaEventDestroy(stop_d2h);
}










