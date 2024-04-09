//------------------------------------------------------------------+
// PCM. Arquitecturas Paralelas Curso 15/16 EUI           30/03/16  |
//                                                                  |
// sumVectorMal.cu: Prueba de suma de un vector en la GPU usando    |
//                  varios bloques y comparar con ejecucion en CPU  |
//                           PRIMER INTENTO ERRONEO                 |
//------------------------------------------------------------------+

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define anchoBloque 1024  // Threads por bloque

//-------------------------------------------------------------------
void initVector (double *V, int cardinalidad, double valor) {
  int i;
  V[0] = valor;
  for (i=1; i<cardinalidad; i++) V[i] = V[i-1] + valor;
}

//-------------------------------------------------------------------
__global__ void sumVectorKernel (double *Ad, int cardinalidad) {
  int yo = blockIdx.x*anchoBloque+threadIdx.x;
  int salto;

  for (salto=1; salto < cardinalidad; salto *= 2) {
    if (yo % (2*salto) == 0) Ad[yo] += Ad[yo+salto];
    __syncthreads();
  }
}

//-------------------------------------------------------------------
int main (int argc, char *argv[])
{
  struct timeval t0, tf, t;
  double  *A, *Ad, sumaCPU;
  int    cardinalidadVectorA, sizeVectorAEnBytes, k, numBloques;

  cardinalidadVectorA = atoi(argv[1]);
  if ((cardinalidadVectorA%anchoBloque) != 0) {
    printf ("El numero de elementos debe ser multiplo de %d\n", anchoBloque);
    exit (0);
  }
  numBloques = cardinalidadVectorA / anchoBloque;
  sizeVectorAEnBytes = cardinalidadVectorA * sizeof(double);
  A = (double *) malloc (sizeVectorAEnBytes);
  initVector (A, cardinalidadVectorA, 0.00001f );
  sumaCPU = 0.0;
  for (k=0; k<cardinalidadVectorA; k++)  sumaCPU += A[k];

  assert (gettimeofday (&t0, NULL) == 0);
  // Transferir A a la GPU
  cudaMalloc ((void**) &Ad, sizeVectorAEnBytes);
  cudaMemcpy (Ad, A, sizeVectorAEnBytes, cudaMemcpyHostToDevice);
  // Invocar al kernel
  sumVectorKernel<<<numBloques, anchoBloque>>>(Ad, cardinalidadVectorA);
  assert (cudaDeviceSynchronize() == 0);
  // Transferir el resultado A[0] desde la GPU
  cudaMemcpy (A, Ad, sizeof(double), cudaMemcpyDeviceToHost);
  // Liberar matriz en la GPU
  cudaFree(Ad);

  assert (gettimeofday (&tf, NULL) == 0);
  timersub (&tf, &t0, &t);
  printf ("Tiempo total = %ld:%ld \n", t.tv_sec, t.tv_usec);
  printf ("Suma CPU = %f suma GPU = %f\n", sumaCPU, A[0]);
  return 0;
}
