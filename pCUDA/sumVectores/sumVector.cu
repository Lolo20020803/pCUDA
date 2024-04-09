//------------------------------------------------------------------+
// PCM. Arquitecturas Paralelas Curso 15/16 EUI           30/03/16  |
//                                                                  |
// sumVector.cu: Prueba de suma de un vector en la GPU usando       |
//               varios bloques y comparar con ejecucion en CPU     |
//------------------------------------------------------------------+

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define anchoBloque 256  // Threads por bloque

//-------------------------------------------------------------------
void initVector (double *V, int cardinalidad, double valor) {
  int i;
  V[0] = valor;
  for (i=1; i<cardinalidad; i++) V[i] = V[i-1] + valor;
}

//-------------------------------------------------------------------
__global__ void sumVectorKernel (double *Ad, double *Bd) {
  int yo = blockIdx.x*anchoBloque+threadIdx.x;
  int salto;

  for (salto=1; salto < anchoBloque; salto *= 2) {
    if (yo % (2*salto) == 0) Ad[yo] += Ad[yo+salto];
    __syncthreads();
  }
  if (threadIdx.x == 0) Bd[blockIdx.x] = Ad[yo];
}

//-------------------------------------------------------------------
int main (int argc, char *argv[])
{
  struct timeval t0, t1, t2, tf, t;
  double  *A, *Ad, *B, *Bd, sumaGPU, sumaCPU;
  int    cardinalidadVectorA, sizeVectorAEnBytes, sizeVectorBEnBytes, k, numBloques;

  cardinalidadVectorA = atoi(argv[1]);
  if ((cardinalidadVectorA%anchoBloque) != 0) {
    printf ("El numero de elementos debe ser multiplo de %d\n", anchoBloque);
    exit (0);
  }
  numBloques = cardinalidadVectorA / anchoBloque;
  sizeVectorAEnBytes = cardinalidadVectorA * sizeof(double);
  sizeVectorBEnBytes = numBloques  * sizeof(double);
  A = (double *) malloc (sizeVectorAEnBytes);
  B = (double *) malloc (sizeVectorBEnBytes);
  initVector (A, cardinalidadVectorA, 0.00001f );
  sumaCPU = 0.0;
  for (k=0; k<cardinalidadVectorA; k++)  sumaCPU += A[k];

  assert (gettimeofday (&t0, NULL) == 0);
  // Transferir A a la GPU y asignar B
  cudaMalloc ((void**) &Ad, sizeVectorAEnBytes);
  cudaMalloc ((void**) &Bd, sizeVectorBEnBytes);
  cudaMemcpy (Ad, A, sizeVectorAEnBytes, cudaMemcpyHostToDevice);
  assert (gettimeofday (&t1, NULL) == 0);
  // Invocar al kernel
  sumVectorKernel<<<numBloques, anchoBloque>>>(Ad, Bd);
  assert (cudaDeviceSynchronize() == 0);
  printf ("ErrorString = %s\n", cudaGetErrorString(cudaGetLastError()));
  assert (gettimeofday (&t2, NULL) == 0);
  // Transferir el vector B[] de sumas parciales desde la GPU
  cudaMemcpy (B, Bd, sizeVectorBEnBytes, cudaMemcpyDeviceToHost);
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
