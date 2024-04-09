//------------------------------------------------------------------+
// PCM. Arquitecturas Paralelas Curso 15/16 EUI           04/03/16  |
//                                                                  |
// mandelbrot.c: Implementacion del modulo que calcula el valor de  |
//               divergencia del conjunto de mandelbrot para el     |
//               pixel (fila, columna).                             |
//------------------------------------------------------------------+

#include <stdio.h>

#include "mandelbrot.cuh"

static int MANDEL_MAX_ITER = 256;

void mandelInit (int maxIteraciones) {
  MANDEL_MAX_ITER = maxIteraciones;
}

int mandelbrot (double X, double Y) {

  double pReal = 0.0;    // Parte real       X
  double pImag = 0.0;    // Parte imaginaria Y
  double pRealAnt, pImagAnt, distancia;
  int i = 0;

  // Se evalua la formula de Mandelbrot
  do {
    pRealAnt = pReal;
    pImagAnt = pImag;
    pReal = ((pRealAnt*pRealAnt) - (pImagAnt*pImagAnt)) + X;
    pImag = (2.0 * (pRealAnt*pImagAnt)) + Y;
    i++;
    distancia = pReal*pReal + pImag*pImag;
  } while ((i < MANDEL_MAX_ITER) && (distancia <= 4.0));
  
  if (i == MANDEL_MAX_ITER)
     return 0;
  else
     return i;
}

