#------------------------------------------------------------------#
# PCM. Arquitecturas Avanzadas Curso 22/23 ETSISI        30/06/23  #
#                                                                  #
# Makefile para el desarrollo de las pruebas de los programas      #
#          relacionados con el aprendizaje de CUDA                 #
#          Nota: Deliberadamente sin optimizar (-O3)               #
#------------------------------------------------------------------#

CC          = /usr/local/cuda-10.2/bin/nvcc -arch=sm_53

todo:	sumvector sumVectorMal sumVector sumVectorBis sumVectorTris  \
	sumvectores sumVectores sumVectoresBis sumVectoresBis1Thread \
        sumvectorO3

sumvector: sumvector.c
	gcc -Wall sumvector.c -o sumvector

sumvectorO3: sumvector.c
	gcc -Wall -O3 sumvector.c -o sumvectorO3

sumVectorMal: sumVectorMal.cu
	$(CC) sumVectorMal.cu -o sumVectorMal

sumVector: sumVector.cu
	$(CC) sumVector.cu -o sumVector

sumVectorBis: sumVectorBis.cu
	$(CC) sumVectorBis.cu -o sumVectorBis

sumVectorTris: sumVectorTris.cu
	$(CC) sumVectorTris.cu -o sumVectorTris

sumVectores: sumVectores.cu
	$(CC) sumVectores.cu -o sumVectores

sumVectoresBis: sumVectoresBis.cu
	$(CC) sumVectoresBis.cu -o sumVectoresBis

sumVectoresBis1Thread: sumVectoresBis1Thread.cu
	$(CC) sumVectoresBis1Thread.cu -o sumVectoresBis1Thread

sumvectores: sumvectores.c
	gcc -Wall sumvectores.c -o sumvectores

borrar:
	rm *.o	sumvector sumVectorMal sumVector sumVectorBis sumVectorTris \
		sumvectores sumVectores sumVectoresBis sumVectoresBis1Thread \
		sumvectorO3

