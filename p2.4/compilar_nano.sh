/usr/local/cuda/bin/nvcc -I/usr/include/opencv4 -o grayscale grayscale.cu -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lcuda -lcudart
