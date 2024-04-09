#------------------------------------------------------------------#
# PCM. Arquitecturas Avanzadas Curso 23/24 ETSISI        30/06/23  #
#                                                                  #
# Makefile para el desarrollo de las pruebas de los programas      #
#          relacionados con los algoritmos de ordenacion.          #
#          Nota: Deliberadamente sin optimizar (-O3)               #
#------------------------------------------------------------------#

CC	= /usr/local/cuda-10.2/bin/nvcc -arch=sm_53

todo:	bitonicSec bitonicSecO3 bitonicPar

bitonicSec:	bitonicSec.c
	gcc -Wall -ggdb bitonicSec.c -o bitonicSec

bitonicSecO3:	bitonicSec.c
	gcc -Wall -ggdb -O3 bitonicSec.c -o bitonicSecO3

bitonicPar: bitonicPar.cu
	$(CC) bitonicPar.cu -o bitonicPar

borrar:	
	rm *.o bitonicSec bitonicSecO3 bitonicPar

