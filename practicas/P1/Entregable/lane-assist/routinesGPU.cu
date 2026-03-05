#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "routinesGPU.h"

#define DEG2RAD 0.017453f

void canny(uint8_t *im, uint8_t *image_out,
	float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float level,
	int height, int width)
{
	int i, j;
	int ii, jj;
	float PI = 3.141593;

	float lowthres, hithres;

	for(i=2; i<height-2; i++)
		for(j=2; j<width-2; j++)
		{
			// Noise reduction
			NR[i*width+j] =
				 (2.0*im[(i-2)*width+(j-2)] +  4.0*im[(i-2)*width+(j-1)] +  5.0*im[(i-2)*width+(j)] +  4.0*im[(i-2)*width+(j+1)] + 2.0*im[(i-2)*width+(j+2)]
				+ 4.0*im[(i-1)*width+(j-2)] +  9.0*im[(i-1)*width+(j-1)] + 12.0*im[(i-1)*width+(j)] +  9.0*im[(i-1)*width+(j+1)] + 4.0*im[(i-1)*width+(j+2)]
				+ 5.0*im[(i  )*width+(j-2)] + 12.0*im[(i  )*width+(j-1)] + 15.0*im[(i  )*width+(j)] + 12.0*im[(i  )*width+(j+1)] + 5.0*im[(i  )*width+(j+2)]
				+ 4.0*im[(i+1)*width+(j-2)] +  9.0*im[(i+1)*width+(j-1)] + 12.0*im[(i+1)*width+(j)] +  9.0*im[(i+1)*width+(j+1)] + 4.0*im[(i+1)*width+(j+2)]
				+ 2.0*im[(i+2)*width+(j-2)] +  4.0*im[(i+2)*width+(j-1)] +  5.0*im[(i+2)*width+(j)] +  4.0*im[(i+2)*width+(j+1)] + 2.0*im[(i+2)*width+(j+2)])
				/159.0;
		}


	for(i=2; i<height-2; i++)
		for(j=2; j<width-2; j++)
		{
			// Intensity gradient of the image
			Gx[i*width+j] = 
				 (1.0*NR[(i-2)*width+(j-2)] +  2.0*NR[(i-2)*width+(j-1)] +  (-2.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
				+ 4.0*NR[(i-1)*width+(j-2)] +  8.0*NR[(i-1)*width+(j-1)] +  (-8.0)*NR[(i-1)*width+(j+1)] + (-4.0)*NR[(i-1)*width+(j+2)]
				+ 6.0*NR[(i  )*width+(j-2)] + 12.0*NR[(i  )*width+(j-1)] + (-12.0)*NR[(i  )*width+(j+1)] + (-6.0)*NR[(i  )*width+(j+2)]
				+ 4.0*NR[(i+1)*width+(j-2)] +  8.0*NR[(i+1)*width+(j-1)] +  (-8.0)*NR[(i+1)*width+(j+1)] + (-4.0)*NR[(i+1)*width+(j+2)]
				+ 1.0*NR[(i+2)*width+(j-2)] +  2.0*NR[(i+2)*width+(j-1)] +  (-2.0)*NR[(i+2)*width+(j+1)] + (-1.0)*NR[(i+2)*width+(j+2)]);


			Gy[i*width+j] = 
				 ((-1.0)*NR[(i-2)*width+(j-2)] + (-4.0)*NR[(i-2)*width+(j-1)] +  (-6.0)*NR[(i-2)*width+(j)] + (-4.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
				+ (-2.0)*NR[(i-1)*width+(j-2)] + (-8.0)*NR[(i-1)*width+(j-1)] + (-12.0)*NR[(i-1)*width+(j)] + (-8.0)*NR[(i-1)*width+(j+1)] + (-2.0)*NR[(i-1)*width+(j+2)]
				+    2.0*NR[(i+1)*width+(j-2)] +    8.0*NR[(i+1)*width+(j-1)] +    12.0*NR[(i+1)*width+(j)] +    8.0*NR[(i+1)*width+(j+1)] +    2.0*NR[(i+1)*width+(j+2)]
				+    1.0*NR[(i+2)*width+(j-2)] +    4.0*NR[(i+2)*width+(j-1)] +     6.0*NR[(i+2)*width+(j)] +    4.0*NR[(i+2)*width+(j+1)] +    1.0*NR[(i+2)*width+(j+2)]);

			G[i*width+j]   = sqrtf((Gx[i*width+j]*Gx[i*width+j])+(Gy[i*width+j]*Gy[i*width+j]));	//G = √Gx²+Gy²
			phi[i*width+j] = atan2f(fabs(Gy[i*width+j]),fabs(Gx[i*width+j]));

			if(fabs(phi[i*width+j])<=PI/8 )
				phi[i*width+j] = 0;
			else if (fabs(phi[i*width+j])<= 3*(PI/8))
				phi[i*width+j] = 45;
			else if (fabs(phi[i*width+j]) <= 5*(PI/8))
				phi[i*width+j] = 90;
			else if (fabs(phi[i*width+j]) <= 7*(PI/8))
				phi[i*width+j] = 135;
			else phi[i*width+j] = 0;
	}

	// Edge
	for(i=3; i<height-3; i++)
		for(j=3; j<width-3; j++)
		{
			pedge[i*width+j] = 0;
			if(phi[i*width+j] == 0){
				if(G[i*width+j]>G[i*width+j+1] && G[i*width+j]>G[i*width+j-1]) //edge is in N-S
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 45) {
				if(G[i*width+j]>G[(i+1)*width+j+1] && G[i*width+j]>G[(i-1)*width+j-1]) // edge is in NW-SE
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 90) {
				if(G[i*width+j]>G[(i+1)*width+j] && G[i*width+j]>G[(i-1)*width+j]) //edge is in E-W
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 135) {
				if(G[i*width+j]>G[(i+1)*width+j-1] && G[i*width+j]>G[(i-1)*width+j+1]) // edge is in NE-SW
					pedge[i*width+j] = 1;
			}
		}

	// Hysteresis Thresholding
	lowthres = level/2;
	hithres  = 2*(level);

	for(i=3; i<height-3; i++)
		for(j=3; j<width-3; j++)
		{
			image_out[i*width+j] = 0;
			if(G[i*width+j]>hithres && pedge[i*width+j])
				image_out[i*width+j] = 255;
			else if(pedge[i*width+j] && G[i*width+j]>=lowthres && G[i*width+j]<hithres)
				// check neighbours 3x3
				for (ii=-1;ii<=1; ii++)
					for (jj=-1;jj<=1; jj++)
						if (G[(i+ii)*width+j+jj]>hithres)
							image_out[i*width+j] = 255;
		}
}


// KERNEL 1: Noise reduction (5x5 Gaussian-like convolution)
__global__ void kernel_noise_reduction(uint8_t *im, float *NR, int height, int width)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= 2 && i < height - 2 && j >= 2 && j < width - 2)
	{
		NR[i * width + j] =
			(2.0f * im[(i - 2) * width + (j - 2)] + 4.0f * im[(i - 2) * width + (j - 1)] + 5.0f * im[(i - 2) * width + (j)] + 4.0f * im[(i - 2) * width + (j + 1)] + 2.0f * im[(i - 2) * width + (j + 2)]
				+ 4.0f * im[(i - 1) * width + (j - 2)] + 9.0f * im[(i - 1) * width + (j - 1)] + 12.0f * im[(i - 1) * width + (j)] + 9.0f * im[(i - 1) * width + (j + 1)] + 4.0f * im[(i - 1) * width + (j + 2)]
				+ 5.0f * im[(i)*width + (j - 2)] + 12.0f * im[(i)*width + (j - 1)] + 15.0f * im[(i)*width + (j)] + 12.0f * im[(i)*width + (j + 1)] + 5.0f * im[(i)*width + (j + 2)]
				+ 4.0f * im[(i + 1) * width + (j - 2)] + 9.0f * im[(i + 1) * width + (j - 1)] + 12.0f * im[(i + 1) * width + (j)] + 9.0f * im[(i + 1) * width + (j + 1)] + 4.0f * im[(i + 1) * width + (j + 2)]
				+ 2.0f * im[(i + 2) * width + (j - 2)] + 4.0f * im[(i + 2) * width + (j - 1)] + 5.0f * im[(i + 2) * width + (j)] + 4.0f * im[(i + 2) * width + (j + 1)] + 2.0f * im[(i + 2) * width + (j + 2)])
			/ 159.0f;
	}
}

// KERNEL 2: Gradient calculation (Sobel operators)
__global__ void kernel_gradient(float *NR, float *G, float *phi, int height, int width)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	float PI = 3.141593f;

	if (i >= 2 && i < height - 2 && j >= 2 && j < width - 2)
	{
		float Gx_value =
			(1.0f * NR[(i - 2) * width + (j - 2)] + 2.0f * NR[(i - 2) * width + (j - 1)] + (-2.0f) * NR[(i - 2) * width + (j + 1)] + (-1.0f) * NR[(i - 2) * width + (j + 2)]
				+ 4.0f * NR[(i - 1) * width + (j - 2)] + 8.0f * NR[(i - 1) * width + (j - 1)] + (-8.0f) * NR[(i - 1) * width + (j + 1)] + (-4.0f) * NR[(i - 1) * width + (j + 2)]
				+ 6.0f * NR[(i)*width + (j - 2)] + 12.0f * NR[(i)*width + (j - 1)] + (-12.0f) * NR[(i)*width + (j + 1)] + (-6.0f) * NR[(i)*width + (j + 2)]
				+ 4.0f * NR[(i + 1) * width + (j - 2)] + 8.0f * NR[(i + 1) * width + (j - 1)] + (-8.0f) * NR[(i + 1) * width + (j + 1)] + (-4.0f) * NR[(i + 1) * width + (j + 2)]
				+ 1.0f * NR[(i + 2) * width + (j - 2)] + 2.0f * NR[(i + 2) * width + (j - 1)] + (-2.0f) * NR[(i + 2) * width + (j + 1)] + (-1.0f) * NR[(i + 2) * width + (j + 2)]);

		float Gy_value =
			((-1.0f) * NR[(i - 2) * width + (j - 2)] + (-4.0f) * NR[(i - 2) * width + (j - 1)] + (-6.0f) * NR[(i - 2) * width + (j)] + (-4.0f) * NR[(i - 2) * width + (j + 1)] + (-1.0f) * NR[(i - 2) * width + (j + 2)]
				+ (-2.0f) * NR[(i - 1) * width + (j - 2)] + (-8.0f) * NR[(i - 1) * width + (j - 1)] + (-12.0f) * NR[(i - 1) * width + (j)] + (-8.0f) * NR[(i - 1) * width + (j + 1)] + (-2.0f) * NR[(i - 1) * width + (j + 2)]
				+ 2.0f * NR[(i + 1) * width + (j - 2)] + 8.0f * NR[(i + 1) * width + (j - 1)] + 12.0f * NR[(i + 1) * width + (j)] + 8.0f * NR[(i + 1) * width + (j + 1)] + 2.0f * NR[(i + 1) * width + (j + 2)]
				+ 1.0f * NR[(i + 2) * width + (j - 2)] + 4.0f * NR[(i + 2) * width + (j - 1)] + 6.0f * NR[(i + 2) * width + (j)] + 4.0f * NR[(i + 2) * width + (j + 1)] + 1.0f * NR[(i + 2) * width + (j + 2)]);

		G[i * width + j] = sqrtf((Gx_value * Gx_value) + (Gy_value * Gy_value));
		phi[i * width + j] = atan2f(fabs(Gy_value), fabs(Gx_value));

		if (fabs(phi[i * width + j]) <= PI / 8)
			phi[i * width + j] = 0;
		else if (fabs(phi[i * width + j]) <= 3 * (PI / 8))
			phi[i * width + j] = 45;
		else if (fabs(phi[i * width + j]) <= 5 * (PI / 8))
			phi[i * width + j] = 90;
		else if (fabs(phi[i * width + j]) <= 7 * (PI / 8))
			phi[i * width + j] = 135;
		else
			phi[i * width + j] = 0;
	}
}

// KERNEL 3: Edge detection (non-maximum suppression)
__global__ void kernel_edge_detection(float *G, float *phi, uint8_t *pedge, int height, int width)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= 3 && i < height - 3 && j >= 3 && j < width - 3)
	{
		pedge[i * width + j] = 0;
		if (phi[i * width + j] == 0)
		{
			if (G[i * width + j] > G[i * width + j + 1] && G[i * width + j] > G[i * width + j - 1])
				pedge[i * width + j] = 1;
		}
		else if (phi[i * width + j] == 45)
		{
			if (G[i * width + j] > G[(i + 1) * width + j + 1] && G[i * width + j] > G[(i - 1) * width + j - 1])
				pedge[i * width + j] = 1;
		}
		else if (phi[i * width + j] == 90)
		{
			if (G[i * width + j] > G[(i + 1) * width + j] && G[i * width + j] > G[(i - 1) * width + j])
				pedge[i * width + j] = 1;
		}
		else if (phi[i * width + j] == 135)
		{
			if (G[i * width + j] > G[(i + 1) * width + j - 1] && G[i * width + j] > G[(i - 1) * width + j + 1])
				pedge[i * width + j] = 1;
		}
	}
}

// KERNEL 4: Hysteresis thresholding
__global__ void kernel_hysteresis(float *G, uint8_t *pedge, uint8_t *image_out, float level, int height, int width)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	float lowthres = level / 2.0f;
	float hithres = 2.0f * level;

	if (i >= 3 && i < height - 3 && j >= 3 && j < width - 3)
	{
		image_out[i * width + j] = 0;
		if (G[i * width + j] > hithres && pedge[i * width + j])
			image_out[i * width + j] = 255;
		else if (pedge[i * width + j] && G[i * width + j] >= lowthres && G[i * width + j] < hithres)
		{
			for (int ii = -1; ii <= 1; ii++)
				for (int jj = -1; jj <= 1; jj++)
					if (G[(i + ii) * width + j + jj] > hithres)
						image_out[i * width + j] = 255;
		}
	}
}

// KERNEL 5: Hough transform (line detection voting)
__global__ void kernel_hough_transform(uint8_t *image_out, uint32_t *accumulators, float *sin_table, float *cos_table,
	int height, int width, int accu_height, int accu_width)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < height && j < width && image_out[i * width + j] > 250)
	{
		float hough_h = (sqrtf(2.0f) * (float)(height > width ? height : width)) / 2.0f;
		float center_x = width / 2.0f;
		float center_y = height / 2.0f;

		for (int theta = 0; theta < 180; theta++)
		{
			float rho = (((float)j - center_x) * cos_table[theta]) + (((float)i - center_y) * sin_table[theta]);
			int rho_int = (int)round(rho + hough_h);
			int idx = (rho_int * 180) + theta;

			if (idx >= 0 && idx < accu_width * accu_height)
			{
				atomicAdd(&accumulators[idx], 1);
			}
		}
	}
}


void getlines(int threshold, uint32_t *accumulators, int accu_width, int accu_height, int width, int height, 
	float *sin_table, float *cos_table,
	int *x1_lines, int *y1_lines, int *x2_lines, int *y2_lines, int *lines)
{
	int rho, theta, ii, jj;
	uint32_t max;

	for(rho=0;rho<accu_height;rho++)
	{
		for(theta=0;theta<accu_width;theta++)  
		{  

			if(accumulators[(rho*accu_width) + theta] >= threshold)  
			{  
				//Is this point a local maxima (9x9)  
				max = accumulators[(rho*accu_width) + theta]; 
				for(int ii=-4;ii<=4;ii++)  
				{  
					for(int jj=-4;jj<=4;jj++)  
					{  
						if( (ii+rho>=0 && ii+rho<accu_height) && (jj+theta>=0 && jj+theta<accu_width) )  
						{  
							if( accumulators[((rho+ii)*accu_width) + (theta+jj)] > max )  
							{
								max = accumulators[((rho+ii)*accu_width) + (theta+jj)];
							}  
						}  
					}  
				}  

				if(max == accumulators[(rho*accu_width) + theta]) //local maxima
				{
					int x1, y1, x2, y2;  
					x1 = y1 = x2 = y2 = 0;  

					if(theta >= 45 && theta <= 135)  
					{
						if (theta>90) {
							//y = (r - x cos(t)) / sin(t)  
							x1 = width/2;  
							y1 = ((float)(rho-(accu_height/2)) - ((x1 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);
							x2 = width;  
							y2 = ((float)(rho-(accu_height/2)) - ((x2 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);  
						} else {
							//y = (r - x cos(t)) / sin(t)  
							x1 = 0;  
							y1 = ((float)(rho-(accu_height/2)) - ((x1 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);
							x2 = width*2/5;  
							y2 = ((float)(rho-(accu_height/2)) - ((x2 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2); 
						}
					} else {
						//x = (r - y sin(t)) / cos(t);  
						y1 = 0;  
						x1 = ((float)(rho-(accu_height/2)) - ((y1 - (height/2) ) * sin_table[theta])) / cos_table[theta] + (width / 2);  
						y2 = height;  
						x2 = ((float)(rho-(accu_height/2)) - ((y2 - (height/2) ) * sin_table[theta])) / cos_table[theta] + (width / 2);  
					}
					x1_lines[*lines] = x1;
					y1_lines[*lines] = y1;
					x2_lines[*lines] = x2;
					y2_lines[*lines] = y2;
					(*lines)++;
				}
			}
		}
	}
}

void init_cos_sin_table(float *sin_table, float *cos_table, int n)
{
	int i;
	for (i=0; i<n; i++)
	{
		sin_table[i] = sinf(i*DEG2RAD);
		cos_table[i] = cosf(i*DEG2RAD);
	}
}

void houghtransform(uint8_t *im, int width, int height, uint32_t *accumulators, int accu_width, int accu_height, 
	float *sin_table, float *cos_table)
{
	int i, j, theta;

	float hough_h = ((sqrt(2.0) * (float)(height>width?height:width)) / 2.0);

	for(i=0; i<accu_width*accu_height; i++)
		accumulators[i]=0;	

	float center_x = width/2.0; 
	float center_y = height/2.0;
	for(i=0;i<height;i++)  
	{  
		for(j=0;j<width;j++)  
		{  
			if( im[ (i*width) + j] > 250 ) // Pixel is edge  
			{  
				for(theta=0;theta<180;theta++)  
				{  
					float rho = ( ((float)j - center_x) * cos_table[theta]) + (((float)i - center_y) * sin_table[theta]);
					accumulators[ (int)((round(rho + hough_h) * 180.0)) + theta]++;

				} 
			} 
		} 
	}
}

uint8_t *image_RGB2BW(uint8_t *image_in, int height, int width)
{
	int i, j;
	uint8_t *imageBW = (uint8_t *)malloc(sizeof(uint8_t) * width * height);
	float R, B, G;

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			R = (float)(image_in[3 * (i * width + j)]);
			G = (float)(image_in[3 * (i * width + j) + 1]);
			B = (float)(image_in[3 * (i * width + j) + 2]);

			imageBW[i * width + j] = (uint8_t)(0.2989 * R + 0.5870 * G + 0.1140 * B);
		}

	return imageBW;
}

void draw_lines(uint8_t *imgtmp, int width, int height, int *x1, int *y1, int *x2, int *y2, int nlines)
{
	int x, y, wl, l;
	int width_line=9;

	for(l=0; l<nlines; l++)
		for(wl=-(width_line>>1); wl<=(width_line>>1); wl++)
			for (x=x1[l]; x<x2[l]; x++)
			{
				y = (float)(y2[l]-y1[l])/(x2[l]-x1[l])*(x-x1[l])+y1[l]; //Line eq. known two points
				if (x+wl>0 && x+wl<width && y>0 && y<height)
				{
					imgtmp[3*((y)*width+x+wl)  ] = 255;
					imgtmp[3*((y)*width+x+wl)+1] = 0;
					imgtmp[3*((y)*width+x+wl)+2] = 0;
				}
			}
}

void lane_assist_GPU(uint8_t *im, int height, int width,
	int *x1, int *y1, int *x2, int *y2, int *nlines)
{
	// Allocate device memory for essential buffers only
	uint8_t *d_im;
	float *d_NR, *d_G, *d_phi;
	uint8_t *d_pedge, *d_image_out;
	float *d_sin_table, *d_cos_table;
	uint32_t *d_accumulators;

	cudaMalloc(&d_im, height * width * sizeof(uint8_t));
	cudaMalloc(&d_NR, height * width * sizeof(float));
	cudaMalloc(&d_G, height * width * sizeof(float));
	cudaMalloc(&d_phi, height * width * sizeof(float));
	cudaMalloc(&d_pedge, height * width * sizeof(uint8_t));
	cudaMalloc(&d_image_out, height * width * sizeof(uint8_t));
	cudaMalloc(&d_sin_table, 180 * sizeof(float));
	cudaMalloc(&d_cos_table, 180 * sizeof(float));

	// Allocate and initialize accumulator
	int accu_height = (int)(sqrtf(2.0f) * (float)(height > width ? height : width)) + 1;
	int accu_width = 180;
	cudaMalloc(&d_accumulators, accu_height * accu_width * sizeof(uint32_t));
	cudaMemset(d_accumulators, 0, accu_height * accu_width * sizeof(uint32_t));

	// Copy input image to device
	cudaMemcpy(d_im, im, height * width * sizeof(uint8_t), cudaMemcpyHostToDevice);

	// Initialize sin/cos tables on host
	float *h_sin_table = (float *)malloc(180 * sizeof(float));
	float *h_cos_table = (float *)malloc(180 * sizeof(float));
	for (int i = 0; i < 180; i++)
	{
		h_sin_table[i] = sinf(i * DEG2RAD);
		h_cos_table[i] = cosf(i * DEG2RAD);
	}
	cudaMemcpy(d_sin_table, h_sin_table, 180 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cos_table, h_cos_table, 180 * sizeof(float), cudaMemcpyHostToDevice);

	// Configure block and grid dimensions
	dim3 blockDim(16, 16);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

	// Execute kernels sequentially
	kernel_noise_reduction<<<gridDim, blockDim>>>(d_im, d_NR, height, width);
	cudaDeviceSynchronize();

	kernel_gradient<<<gridDim, blockDim>>>(d_NR, d_G, d_phi, height, width);
	cudaDeviceSynchronize();

	kernel_edge_detection<<<gridDim, blockDim>>>(d_G, d_phi, d_pedge, height, width);
	cudaDeviceSynchronize();

	kernel_hysteresis<<<gridDim, blockDim>>>(d_G, d_pedge, d_image_out, 1000.0f, height, width);
	cudaDeviceSynchronize();

	kernel_hough_transform<<<gridDim, blockDim>>>(d_image_out, d_accumulators, d_sin_table, d_cos_table,
		height, width, accu_height, accu_width);
	cudaDeviceSynchronize();

	// Copy results back to host
	cudaMemcpy(im, d_image_out, height * width * sizeof(uint8_t), cudaMemcpyDeviceToHost);

	// Copy accumulator back to host and extract lines
	uint32_t *h_accumulators = (uint32_t *)malloc(accu_height * accu_width * sizeof(uint32_t));
	cudaMemcpy(h_accumulators, d_accumulators, accu_height * accu_width * sizeof(uint32_t), cudaMemcpyDeviceToHost);

	// Calculate threshold
	int threshold = (width > height) ? width / 6 : height / 6;

	// Extract lines from accumulator
	*nlines = 0;
	getlines(threshold, h_accumulators, accu_width, accu_height, width, height,
		h_sin_table, h_cos_table,
		x1, y1, x2, y2, nlines);

	// Free device memory
	cudaFree(d_im);
	cudaFree(d_NR);
	cudaFree(d_G);
	cudaFree(d_phi);
	cudaFree(d_pedge);
	cudaFree(d_image_out);
	cudaFree(d_sin_table);
	cudaFree(d_cos_table);
	cudaFree(d_accumulators);

	free(h_sin_table);
	free(h_cos_table);
	free(h_accumulators);
}
