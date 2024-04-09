//------------------------------------------------------------------+
// PCM. Arquitecturas Paralelas Curso 15/16 EUI           29/02/16  |
//                                                                  |
// sumVectores.cu: Prueba de suma de vector en la GPU para comparar |
//                 con los tiempos de ejecucion en la CPU           |
//                          version en 1D                           |
//------------------------------------------------------------------+

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

//-------------------------------------------------------------------
void initVector (float *V, int cardinalidad, float valor) {
  int i;
  V[0] = valor;
  for (i=1; i<cardinalidad; i++) V[i] = V[i-1] + valor;
}

//-------------------------------------------------------------------
__global__ void sumVectorKernel (float *Ad, float *Bd, float *Cd) {
  int yo = threadIdx.x;

  Cd[yo] = Ad[yo] + Bd[yo];
}

//-------------------------------------------------------------------
int main (int argc, char *argv[])
{
  struct timeval t0, tf, t;
  float  *A,  *B,  *C;
  float  *Ad, *Bd, *Cd;
  int    cardinalidadVector, sizeVectorEnBytes, k;
  double suma;

  cardinalidadVector = atoi(argv[1]);
  sizeVectorEnBytes  = cardinalidadVector * sizeof(float);
  A    = (float *) malloc (sizeVectorEnBytes);
  B    = (float *) malloc (sizeVectorEnBytes);
  C    = (float *) malloc (sizeVectorEnBytes);
  initVector (A, cardinalidadVector, 0.001f );
  initVector (B, cardinalidadVector, 0.002f);

  assert (gettimeofday (&t0, NULL) == 0);
  // Transferir A y B a la GPU
  cudaMalloc ((void**) &Ad, sizeVectorEnBytes);
  cudaMemcpy (Ad, A, sizeVectorEnBytes, cudaMemcpyHostToDevice);
  cudaMalloc ((void**) &Bd, sizeVectorEnBytes);
  cudaMemcpy (Bd, B, sizeVectorEnBytes, cudaMemcpyHostToDevice);
  // Ubicar C en la GPU
  assert (cudaMalloc ((void**) &Cd, sizeVectorEnBytes) == 0);
  // Invocar al kernel
  sumVectorKernel<<<1, cardinalidadVector>>>(Ad, Bd, Cd);
  printf("ERROR: %d\n", cudaGetLastError());
  cudaDeviceSynchronize() == 0;
  // Transferir C desde la GPU
  cudaMemcpy (C, Cd, sizeVectorEnBytes, cudaMemcpyDeviceToHost);
  // Liberar matrices en la GPU
  cudaFree(Ad); cudaFree(Bd); cudaFree(Cd);

  assert (gettimeofday (&tf, NULL) == 0);
  timersub (&tf, &t0, &t);
  printf ("Tiempo en GPU = %ld:%ld \n", t.tv_sec, t.tv_usec);

  // Comprobacion del resultado
  suma = 0.0;
  for (k=0; k<cardinalidadVector; k++) {
    assert (C[k] == A[k]+B[k]);
    suma += C[k];
  }
  printf("OK suma = %f\n", suma);
  return 0;
}
