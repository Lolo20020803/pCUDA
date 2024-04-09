//------------------------------------------------------------------+
// PCM. Arquitecturas Paralelas Curso 15/16 EUI           30/03/16  |
//                                                                  |
// sumVectorTris.cu: Prueba de suma de un vector en la GPU usando   |
//                   varios bloques y comparar con ejecucion en CPU |
//                       VERSION OPTIMIZANDO ACCESOS A MGlobal      |
//                               Y SIN MALGASTAR LA MITAD DE Threads|
//------------------------------------------------------------------+

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define anchoBloque 256  // Threads por bloque

//-------------------------------------------------------------------
void initVector (double *V, int cardinalidadVector, double valor) {
  int i;
  V[0] = valor;
  for (i=1; i<cardinalidadVector; i++) V[i] = V[i-1] + valor;
}

//-------------------------------------------------------------------
__global__ void sumVectorKernel (double *Ad, double *Bd) {
  int yo = blockIdx.x*anchoBloque*2+threadIdx.x;
  int salto;

  for (salto = anchoBloque; salto > 0; salto /= 2) {
    if (threadIdx.x < salto) Ad[yo] += Ad[yo+salto];
    __syncthreads();
  }
  if (threadIdx.x == 0) Bd[blockIdx.x] = Ad[yo];
}

//-------------------------------------------------------------------
int main (int argc, char *argv[])
{
  struct timeval t0, t1, t2, tf, t;
  double  *A, *B, *Ad, *Bd, sumaGPU, sumaCPU;
  int    cardinalidadVector, sizeVectorEnBytes, k, numBloques;

  cardinalidadVector = atoi(argv[1]);
  if ((cardinalidadVector%anchoBloque) != 0) {
    printf ("El numero de elementos debe ser multiplo de %d\n", anchoBloque);
    exit (0);
  }
  numBloques = (cardinalidadVector / anchoBloque) / 2;
  sizeVectorEnBytes = cardinalidadVector * sizeof(double);
  A = (double *) malloc (sizeVectorEnBytes);
  initVector (A, cardinalidadVector, 0.00001f );
  sumaCPU = 0.0;
  for (k=0; k<cardinalidadVector; k++)  sumaCPU += A[k];
  B = (double *) malloc (numBloques * sizeof(double));

  assert (gettimeofday (&t0, NULL) == 0);
  // Transferir A a la GPU
  cudaMalloc ((void**) &Ad, sizeVectorEnBytes);
  cudaMemcpy (Ad, A, sizeVectorEnBytes, cudaMemcpyHostToDevice);
  // Ubicar B en la GPU
  cudaMalloc ((void**) &Bd, numBloques * sizeof(double));
  assert (gettimeofday (&t1, NULL) == 0);
  // Invocar al kernel
  sumVectorKernel<<<numBloques, anchoBloque>>>(Ad, Bd);
  assert (cudaDeviceSynchronize() == 0);
  assert (gettimeofday (&t2, NULL) == 0);
  // Transferir B desde la GPU
  cudaMemcpy (B, Bd, numBloques*sizeof(double), cudaMemcpyDeviceToHost);
  // Liberar matrices en la GPU
  cudaFree(Ad);
  cudaFree(Bd);
  assert (gettimeofday (&tf, NULL) == 0);

  timersub (&t2, &t1, &t);
  printf ("Tiempo sumar vector = %ld:%ld \n", t.tv_sec, t.tv_usec);
  timersub (&tf, &t0, &t);
  printf ("Tiempo total        = %ld:%ld \n", t.tv_sec, t.tv_usec);
  sumaGPU = 0.0;
  for (k=0; k<numBloques; k++) sumaGPU += B[k];
  printf ("Suma GPU = %f suma CPU = %f\n", sumaGPU, sumaCPU);
  return 0;
}
