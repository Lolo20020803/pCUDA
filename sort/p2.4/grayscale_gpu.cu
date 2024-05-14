#include <stdio.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

__global__ void RGBtoGrayscale(unsigned char *image, int width, int height, unsigned char *grayscale) {
//rellenar
}

#define NUM_CHANNELS 3

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Uso: %s <nombre_imagen>\n", argv[0]);
    return -1;
  }

  // Load image using OpenCV
  cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
  int width = image.cols;
  int height = image.rows;

  printf("cols: %d rows: %d\n", image.cols, image.rows);

  // cv::imshow("Original", image);
  // cv::waitKey(0);

  // Allocate memory for grayscale image on host and device
  unsigned char *host_image = (unsigned char*)image.data;
  unsigned char *device_image, *device_grayscale;
  cudaMalloc(&device_image, width * height * NUM_CHANNELS * sizeof(unsigned char));
  cudaMalloc(&device_grayscale, width * height * NUM_CHANNELS * sizeof(unsigned char));

  // Copy image to device memory
  cudaMemcpy(device_image, host_image, width * height * NUM_CHANNELS * sizeof(unsigned char), cudaMemcpyHostToDevice);

  // Define thread block size and grid size
  int blockLen = 16; 
  dim3 dimGrid((width/blockLen)+1, (height/blockLen)+1);
  dim3 dimBlocks(blockLen, blockLen);
  printf("dimGrid: %d cols (width), %d rows (height)\n", dimGrid.x, dimGrid.y);

  // Launch kernel to convert to grayscale
  RGBtoGrayscale<<<dimGrid, dimBlocks>>>(device_image, width, height, device_grayscale);
  assert(cudaDeviceSynchronize() == 0);

  // Copy grayscale image back to host memory
  cudaMemcpy(host_image, device_grayscale, width * height * NUM_CHANNELS * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  // Show grayscale image
  cv::imshow("Grayscale Image", image);
  cv::waitKey(0);
  cv::imwrite("imagen_gris.png", image);

  // Free memory
  cudaFree(device_image);
  cudaFree(device_grayscale);

  return 0;
}

