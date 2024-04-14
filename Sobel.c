#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define SOBEL_KERNEL_SIZE 3

// Sobel kernels for horizontal and vertical changes
int sobel_kernel_x[SOBEL_KERNEL_SIZE][SOBEL_KERNEL_SIZE] = {{-1, 0, 1},
                                                            {-2, 0, 2},
                                                            {-1, 0, 1}};

int sobel_kernel_y[SOBEL_KERNEL_SIZE][SOBEL_KERNEL_SIZE] = {{-1, -2, -1},
                                                            { 0,  0,  0},
                                                            { 1,  2,  1}};

// Function to apply Sobel filter to a given image
void sobel_filter(unsigned char* input_image, unsigned char* output_image, int width, int height) {
    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            // Compute gradient in X direction
            int gradient_x = 0;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    gradient_x += input_image[(y + i) * width + (x + j)] * sobel_kernel_x[i + 1][j + 1];
                }
            }
            
            // Compute gradient in Y direction
            int gradient_y = 0;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    gradient_y += input_image[(y + i) * width + (x + j)] * sobel_kernel_y[i + 1][j + 1];
                }
            }
            
            // Compute gradient magnitude
            output_image[y * width + x] = sqrt(gradient_x * gradient_x + gradient_y * gradient_y);
        }
    }
}

int main() {
    int width, height, channels;
    unsigned char* input_image = stbi_load("input_image.jpg", &width, &height, &channels, 0);
    
    if (!input_image) {
        printf("Error loading the image!\n");
        return 1;
    }
    
    unsigned char* output_image = (unsigned char*)malloc(width * height);
    
    
    sobel_filter(input_image, output_image, width, height);
    
   
    stbi_write_jpg("output_image.jpg", width, height, 1, output_image, 100);
    
    
    stbi_image_free(input_image);
    free(output_image);
    
    return 0;
}
