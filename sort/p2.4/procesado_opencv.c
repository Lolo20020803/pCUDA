#include <stdio.h>
#include <opencv2/opencv.hpp>

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
  cv::Mat grayImage, sobel;
  cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
  cv::Sobel(grayImage,sobel, CV_8U,1,1,3);
  cv::imshow("sobel",sobel);
  cv::waitKey(0);
  cv::imwrite("imagen_sobel.png", sobel);

  return 0;
}

