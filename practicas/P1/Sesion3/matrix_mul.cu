#include <stdio.h>
#include "matrix_mul.h"

// Thread block size
#define BLOCK_SIZE 16 

// Forward declaration of the device multiplication function
__global__ void Muld(float*, float*, int, int, float*);

// Host multiplication function
// Compute C = A * B
// hA is the height of A
// wA is the width of A
// wB is the width of B


extern "C" void Mul___(float* A, float* B, int hA, int wA, int wB, float* C)
{
	int size;

	// Load A and B to the device
	float* Ad;
	size = hA * wA * sizeof(float);
	cudaMalloc((void**)&Ad, size);
	cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
	float* Bd;
	size = wA * wB * sizeof(float);
	cudaMalloc((void**)&Bd, size);
	cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);

	// Allocate C on the device
	float* Cd;
	size = hA * wB * sizeof(float);
	cudaMalloc((void**)&Cd, size);

	// Compute the execution configuration assuming
	// the matrix dimensions are multiples of BLOCK_SIZE
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(wB / dimBlock.x, hA / dimBlock.y);

	// Launch the device computation
	Muld<<<dimGrid, dimBlock>>>(Ad, Bd, wA, wB, Cd);

	// Read C from the device
	cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(Ad);
	cudaFree(Bd);
	cudaFree(Cd);
}


__global__ void Muld(float* A, float* B, int wA, int wB, float* C)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int col = blockIdx.x * BLOCK_SIZE + tx;
	int row = blockIdx.y * BLOCK_SIZE + ty;
	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
	float acc = 0.0f;
	int numTiles = (wA + BLOCK_SIZE- 1) / BLOCK_SIZE;
	for (int t = 0; t < numTiles; t++) {
		int Acol = t * BLOCK_SIZE + tx;
		int Brow = t * BLOCK_SIZE + ty;
		if (row < wA && Acol < wA)
			As[ty][tx] = A[row*wA + Acol];
		else
			As[ty][tx] = 0.0f;
		if (Brow < wA && col < wB)
			Bs[ty][tx] = B[Brow*wB + col];
		else
			Bs[ty][tx] = 0.0f;
		__syncthreads();
		for (int k = 0; k < BLOCK_SIZE; k++)
			acc += As[ty][k] * Bs[k][tx];
		__syncthreads();
	}
	if (row < wA && col < wB)
		C[row*wB + col] = acc;
}





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
