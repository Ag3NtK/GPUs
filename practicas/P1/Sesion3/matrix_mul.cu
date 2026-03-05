#include <stdio.h>
#include "matrix_mul.h"

// Thread block size
#define BLOCK_SIZE 16 

// Forward declaration of the device multiplication function (naive 2D grid kernel)
// parameters: A, B, hA, wA, wB, C
__global__ void Muld(float*, float*, int, int, int, float*);

// Host multiplication function
// Compute C = A * B
// hA is the height of A
// wA is the width of A
// wB is the width of B




// naive kernel: each thread computes one element C[row,col]
__global__ void Muld(float* A, float* B, int hA, int wA, int wB, float* C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < hA && col < wB) {
        float acc = 0.0f;
        for (int k = 0; k < wA; ++k) {
            acc += A[row * wA + k] * B[k * wB + col];
        }
        C[row * wB + col] = acc;
    }
}

// test harness main
int main(int argc, char** argv) {
    int N = 1024;
    if (argc > 1) N = atoi(argv[1]);
    int hA = N, wA = N, wB = N;
    size_t sizeA = hA * wA * sizeof(float);
    size_t sizeB = wA * wB * sizeof(float);
    size_t sizeC = hA * wB * sizeof(float);

    float *hA_m = (float*)malloc(sizeA);
    float *hB_m = (float*)malloc(sizeB);
    float *hC_m = (float*)malloc(sizeC);
    float *hC_ref = (float*)malloc(sizeC);
    srand(0);
    for (int i = 0; i < hA * wA; ++i) hA_m[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < wA * wB; ++i) hB_m[i] = rand() / (float)RAND_MAX;
    // CPU reference
    for (int i = 0; i < hA; ++i)
        for (int j = 0; j < wB; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < wA; ++k)
                acc += hA_m[i*wA + k] * hB_m[k*wB + j];
            hC_ref[i*wB + j] = acc;
        }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeA);
    cudaMalloc(&dB, sizeB);
    cudaMalloc(&dC, sizeC);
    cudaMemcpy(dA, hA_m, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB_m, sizeB, cudaMemcpyHostToDevice);

    dim3 blocks[] = { dim3(8,8), dim3(16,16), dim3(32,8) };
    printf("Matrix %dx%d x %dx%d\n", hA, wA, wA, wB);
    printf("bx x by    time(ms)    GFLOP/s\n");
    for (auto b : blocks) {
        dim3 grid((wB + b.x - 1) / b.x, (hA + b.y - 1) / b.y);
        // warmup
        Muld<<<grid,b>>>(dA,dB,hA,wA,wB,dC);
        cudaDeviceSynchronize();
        cudaEvent_t st, en;
        cudaEventCreate(&st);
        cudaEventCreate(&en);
        cudaEventRecord(st);
        Muld<<<grid,b>>>(dA,dB,hA,wA,wB,dC);
        cudaEventRecord(en);
        cudaEventSynchronize(en);
        float ms;
        cudaEventElapsedTime(&ms, st, en);
        cudaMemcpy(hC_m, dC, sizeC, cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < hA*wB; ++i) if (fabs(hC_m[i]-hC_ref[i]) > 1e-3f) { ok=false; break; }
        if (!ok) printf("Mismatch for block %dx%d\n", b.x, b.y);
        double ops = 2.0 * hA * wA * wB;
        double gflops = ops / (ms*1e-3) / 1e9;
        printf("%4d x %4d    %8.3f    %8.2f\n", b.x, b.y, ms, gflops);
        cudaEventDestroy(st);
        cudaEventDestroy(en);
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA_m);
    free(hB_m);
    free(hC_m);
    free(hC_ref);
    return 0;
}

#if 0  // tiled version not used in this exercise, kept for reference
__global__ void Muld(float* A, float* B, int hA, int wA, int wB, float* C) {

	// compute global row and column for this thread
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	// boundary check (A is hA x wA, B is wA x wB)
	if (row < hA && col < wB) {
		float acc = 0.0f;
		for (int k = 0; k < wA; k++) {
			acc += A[row * wA + k] * B[k * wB + col];
		}
		C[row * wB + col] = acc;
	}
}

#endif


#if 0
// Device multiplication function called by Mul()
// Compute C = A * B
// wA is the width of A
// wB is the width of B

__global__ void Muld(float* A, float* B, int wA, int wB, float* C)	V0
{
	for (int row = 0; row < wA; row++) {
		for (int col = 0; col < wB; col++) {
			float acc = 0.0f;
			for (int k = 0; k < wA; k++) {
				acc += A[row * wA + k] * B[k * wB + col];
			}
			C[row * wB + col] = acc;
		}
	}
}


__global__ void Muld(float* A, float* B, int wA, int wB, float* C)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = ...;

	// Index of the last sub-matrix of A processed by the block
	int aEnd = ...;

	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * wB;

	// The element of the block sub-matrix that is computed
	// by the thread
	float Csub = 0;

	// Loop over all the sub-matrices of A and B required to
	// compute the block sub-matrix
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		// Shared memory for the sub-matrix of A
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

		// Shared memory for the sub-matrix of B
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load the matrices from global memory to shared memory;
		// each thread loads one element of each matrix
		As[ty][tx] = A[...];
		Bs[ty][tx] = B[...];
		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
		for (int k = 0; k < BLOCK_SIZE; ++k)
			....

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}
	
	// Write the block sub-matrix to global memory;
	// each thread writes one element
	...
}
#endif
