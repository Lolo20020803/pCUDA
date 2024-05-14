#include <stdio.h>
#include <opencv2/opencv.hpp>

void grayscale(cv::Mat original, cv::Mat gray) {
  for (int i = 0; i < original.rows; i++) {
    for (int j = 0; j < original.cols; j++) {
       char* pOrig = (char*) original.data;
       char* pDest = (char*) gray.data;
       int idx = i * original.cols * 3 + j * 3;
       float pixel = 0.299f * pOrig[idx + 2] + 0.587f * pOrig[idx + 1] + 0.114f * pOrig[idx];
       pDest[idx] = (unsigned char) pixel;
       pDest[idx + 1] = (unsigned char) pixel;
       pDest[idx + 2] = (unsigned char) pixel;
    }
  }
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

  cv::Mat grayImage = image.clone();
  grayscale(image, grayImage);
    
  // cv::imshow("Grey", grayImage);
  // cv::waitKey(0);
  cv::imwrite("imagen_gris.png", grayImage);

  return 0;
}

