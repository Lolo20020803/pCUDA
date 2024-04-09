#------------------------------------------------------------------#
# PCM. Arquitecturas Avanzadas Curso 23/24 ETSISI        30/06/23  #
#                                                                  #
# Makefile para el desarrollo de las pruebas de los programas      #
#          relacionados con el aprendizaje de CUDA                 #
#------------------------------------------------------------------#

CC          = /usr/local/cuda-10.2/bin/nvcc -arch=sm_53

todo:	mulmatriz mulmatrizOpt mulMatriz mulMatrizOpt

mulmatriz: mulmatriz.c
	gcc -Wall mulmatriz.c -o mulmatriz

mulmatrizOpt: mulmatriz.c
	gcc -O3 -Wall mulmatriz.c -o mulmatrizOpt

mulMatriz: mulMatriz.cu
	$(CC) -O3 mulMatriz.cu -o mulMatriz

mulMatrizOpt: mulMatrizOpt.cu
	$(CC) -O3 mulMatrizOpt.cu -o mulMatrizOpt

borrar:
	rm *.o	 mulmatriz mulmatrizOpt mulMatriz mulMatrizOpt

