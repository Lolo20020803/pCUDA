//------------------------------------------------------------------+
// PCM. Procesamiento Paralelo  Curso 13/14 EUI           14/11/13  |
//                                                                  |
// sumVectoresBis.cu: Prueba de suma de vector en la GPU usando     |
//                    varios bloques y comparar con ejecucion en CPU|
//------------------------------------------------------------------+

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define anchoBloque 256  // Threads por bloque

//-------------------------------------------------------------------
void initVector (float *V, int cardinalidad, float valor) {
  int i;
  V[0] = valor;
  for (i=1; i<cardinalidad; i++) V[i] = V[i-1] + valor;
}

//-------------------------------------------------------------------
__global__ void sumVectorKernel (float *Ad, float *Bd, float *Cd) {
  int yo = ; // Rellenar con la expresion adecuada

  Cd[yo] = Ad[yo] + Bd[yo];
}

//-------------------------------------------------------------------
int main (int argc, char *argv[])
{
  struct timeval t0, tf, t;
  float  *A,  *B,  *C;
  float  *Ad, *Bd, *Cd;
  int    cardinalidadVector, sizeVectorEnBytes, k;

  cardinalidadVector  = atoi(argv[1]);
  if ((cardinalidadVector%anchoBloque) != 0) {
    printf ("El numero de elementos debe ser multiplo de %d\n", anchoBloque);
    exit (0);
  }
  sizeVectorEnBytes = cardinalidadVector * sizeof(float);
  A = (float *) malloc (sizeVectorEnBytes);
  B = (float *) malloc (sizeVectorEnBytes);
  C = (float *) malloc (sizeVectorEnBytes);
  initVector (A, cardinalidadVector, 0.00001f );
  initVector (B, cardinalidadVector, 0.00002f);

  assert (gettimeofday (&t0, NULL) == 0);
  // Transferir A y B a la GPU
  cudaMalloc ((void**) &Ad, sizeVectorEnBytes);
  cudaMemcpy (Ad, A, sizeVectorEnBytes, cudaMemcpyHostToDevice);
  cudaMalloc ((void**) &Bd, sizeVectorEnBytes);
  cudaMemcpy (Bd, B, sizeVectorEnBytes, cudaMemcpyHostToDevice);
  // Ubicar C en la GPU
  cudaMalloc ((void**) &Cd, sizeVectorEnBytes);
  // Invocar al kernel
  // Poner la expresion adecuada
  assert (cudaDeviceSynchronize() == 0);
  // Transferir C desde la GPU
  cudaMemcpy (C, Cd, sizeVectorEnBytes, cudaMemcpyDeviceToHost);
  // Liberar matrices en la GPU
  cudaFree(Ad); cudaFree(Bd); cudaFree(Cd);
  assert (gettimeofday (&tf, NULL) == 0);

  timersub (&tf, &t0, &t);
  printf ("Tiempo total = %ld:%ld \n", t.tv_sec, t.tv_usec);
  for (k=0; k<cardinalidadVector; k++) assert (C[k] == (A[k] + B[k]));
  printf("OK\n");
  return 0;
}
