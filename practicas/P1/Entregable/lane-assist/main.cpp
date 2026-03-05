#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <chrono> // Reemplaza a las librerías POSIX de Linux

#include "routinesCPU.h"
#include "routinesGPU.h"
#include "png_io.h"

int main(int argc, char **argv)
{
    uint8_t *imtmp, *im;
    int width, height;

    float sin_table[180], cos_table[180];
    int nlines=0; 
    int x1[10], x2[10], y1[10], y2[10];

    /* Only accept a concrete number of arguments */
    if(argc != 3)
    {
        printf("./exec image.png [c/g]\n");
        exit(-1);
    }

    /* Read images */
    imtmp = read_png_fileRGB(argv[1], &width, &height);
    im    = image_RGB2BW(imtmp, height, width);

    init_cos_sin_table(sin_table, cos_table, 180);  

    // Create temporal buffers 
    uint8_t *imEdge = (uint8_t *)malloc(sizeof(uint8_t) * width * height);
    float *NR = (float *)malloc(sizeof(float) * width * height);
    float *G = (float *)malloc(sizeof(float) * width * height);
    float *phi = (float *)malloc(sizeof(float) * width * height);
    float *Gx = (float *)malloc(sizeof(float) * width * height);
    float *Gy = (float *)malloc(sizeof(float) * width * height);
    uint8_t *pedge = (uint8_t *)malloc(sizeof(uint8_t) * width * height);

    //Create the accumulators
    float hough_h = ((sqrt(2.0) * (float)(height>width?height:width)) / 2.0);
    int accu_height = hough_h * 2.0; // -rho -> +rho
    int accu_width  = 180;
    uint32_t *accum = (uint32_t*)malloc(accu_width*accu_height*sizeof(uint32_t));

    switch (argv[2][0]) {
        case 'c':
        {   // Las llaves son necesarias en C++ al declarar variables dentro de un case
            auto start = std::chrono::high_resolution_clock::now();
            
            lane_assist_CPU(im, height, width, 
                imEdge, NR, G, phi, Gx, Gy, pedge,
                sin_table, cos_table,
                accum, accu_height, accu_width,
                x1, y1, x2, y2, &nlines);
                
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            printf("CPU Execution time %f ms.\n", duration.count());
            break;
        }
        case 'g':
        {
            auto start = std::chrono::high_resolution_clock::now();
            
            lane_assist_GPU(im, height, width,
                x1, y1, x2, y2, &nlines);
                
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            printf("GPU Execution time %f ms.\n", duration.count());
            break;
        }
        default:
            printf("Not Implemented yet!!\n");
    }

    for (int l=0; l<nlines; l++)
        printf("(x1,y1)=(%d,%d) (x2,y2)=(%d,%d)\n", x1[l], y1[l], x2[l], y2[l]);

    draw_lines(imtmp, width, height, x1, y1, x2, y2, nlines);

    write_png_fileRGB("out.png", imtmp, width, height);

    // Liberación de la memoria alojada
    free(imEdge); free(NR); free(G); free(phi); 
    free(Gx); free(Gy); free(pedge); free(accum);

    return 0;
}