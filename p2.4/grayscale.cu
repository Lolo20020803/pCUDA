#include <stdio.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#define anchoBloque 1024
__global__ void RGBtoGrayscale(unsigned char *image, int width, int height, unsigned char *grayscale) {


	//Ancho de la imagen / los threads y que vayan cambiando la imagen de manera vertical
	int col = blockIdx.x*width+threadIdx.x;
	int fila = blockIdx.y *height + threadIdx.y;
	float pixel = 0.299f * image[fila,col]+ 0.587f * image[fila,col+1] + 0.114f * image[fila,col+2];
	grayscale[fila,col] =(unsigned char) pixel;
	grayscale[fila,col+1] = (unsigned char) pixel;
	grayscale[fila,col+2] =(unsigned char)  pixel;

	
}


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
  cudaMalloc(&device_image, width * height * 3 * sizeof(unsigned char));
  cudaMalloc(&device_grayscale, width * height * 3 * sizeof(unsigned char));

  // Copy image to device memory
  cudaMemcpy(device_image, host_image, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

  // Define thread block size and grid size
  int blockLen = 16; 
  dim3 dimGrid((width/blockLen)+1, (height/blockLen)+1);
  dim3 dimBlocks(blockLen, blockLen);
  printf("dimGrid: %d cols (width), %d rows (height)\n", dimGrid.x, dimGrid.y);

  // Launch kernel to convert to grayscale
  for (int i = 0; i < 1000; i++) {
    RGBtoGrayscale<<<dimGrid, dimBlocks>>>(device_image, width, height, device_grayscale);
    assert(cudaDeviceSynchronize() == 0);
  }

  // Copy grayscale image back to host memory
  cv::Mat gray_image(height,width ,CV_8UC1);
  unsigned char *host_grayscale = (unsigned char*)gray_image.data;
  cudaMemcpy(host_image, device_grayscale, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  //cudaMemcpy(host_grayscale, device_grayscale, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  // Convert OpenCV Mat back to BGR format for display
  // cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

  // Show grayscale image
  cv::imshow("Grayscale Image",host_image);
  cv::waitKey(0);
  cv::imwrite("imagen_gris.png", host_image);
  // Free memory
  cudaFree(device_image);
  cudaFree(device_grayscale);

  return 0;
}

