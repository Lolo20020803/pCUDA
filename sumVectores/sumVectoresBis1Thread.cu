//------------------------------------------------------------------+
// PCM. Arquitecturas Paralelas Curso 15/16 EUI           30/03/16  |
//                                                                  |
// sumVectoresBis1Thread.cu: Prueba de suma de vector en la GPU     |
//                    usando un unico thread para ver aceleracion   |
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
__global__ void sumVectorKernel (float *Ad, float *Bd, float *Cd, int cardinalidadVector) {
  int i=0;
  for(;i<cardinalidadVector;i++){
	  Cd[i]=Ad[i]+Bd[i];
  }

  // Rellenar adecuadamente
}

//-------------------------------------------------------------------
int main (int argc, char *argv[])
{
  struct timeval t0, tf, t;
  float  *A,  *B,  *C;
  float  *Ad, *Bd, *Cd;
  int    cardinalidadVector, sizeVectorEnBytes, k;

  cardinalidadVector = atoi(argv[1]);
  sizeVectorEnBytes  = cardinalidadVector * sizeof(float);
  A = (float *) malloc (sizeVectorEnBytes);
  B = (float *) malloc (sizeVectorEnBytes);
  C = (float *) malloc (sizeVectorEnBytes);
  initVector (A, cardinalidadVector, 0.00001f );
  initVector (B, cardinalidadVector, 0.00002f);

  // Transferir A y B a la GPU
  cudaMalloc ((void**) &Ad, sizeVectorEnBytes);
  cudaMemcpy (Ad, A, sizeVectorEnBytes, cudaMemcpyHostToDevice);
  cudaMalloc ((void**) &Bd, sizeVectorEnBytes);
  cudaMemcpy (Bd, B, sizeVectorEnBytes, cudaMemcpyHostToDevice);
  // Ubicar C en la GPU
  cudaMalloc ((void**) &Cd, sizeVectorEnBytes);
  assert (gettimeofday (&t0, NULL) == 0);
  // Invocar al kernel
  sumVectorKernel<<<1,1>>>(Ad,Bd,Cd,cardinalidadVector);
  assert (cudaDeviceSynchronize() == 0);
  assert (gettimeofday (&tf, NULL) == 0);
  // Transferir C desde la GPU
  cudaMemcpy (C, Cd, sizeVectorEnBytes, cudaMemcpyDeviceToHost);
  // Liberar matrices en la GPU
  cudaFree(Ad); cudaFree(Bd); cudaFree(Cd);
  timersub (&tf, &t0, &t);

  printf ("Tiempo de ejecucion del kernel = %ld:%ld \n", t.tv_sec, t.tv_usec);
  for (k=0; k<cardinalidadVector; k++) {
    assert (C[k] == (A[k] + B[k]));
  }
  printf("OK\n");
  return 0;
}
