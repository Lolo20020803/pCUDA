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
  int yo = blockIdx.x * anchoBloque + threadIdx.x; // Rellenar con la expresion adecuada

  Cd[yo] = Ad[yo] + Bd[yo];
}

//-------------------------------------------------------------------
int main (int argc, char *argv[])
{
  struct timeval t0, tf, t;
  float  *A,  *B,  *C;
  float  *Ad, *Bd, *Cd;
  int    cardinalidadVector, sizeVectorEnBytes, k;
  //tiempos de alocación
  struct timeval ta, tb, t0b, tc, t0c, tk, t0k, tcp, t0cp,ttf;

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
  assert (gettimeofday(&t0,NULL)==0);
  // Transferir A y B a la GPU
  cudaMalloc ((void**) &Ad, sizeVectorEnBytes);
  cudaMemcpy (Ad, A, sizeVectorEnBytes, cudaMemcpyHostToDevice);
  //tiempos de alocación 
  assert (gettimeofday (&ta, NULL) == 0);
  timersub(&ta, &t0, &t);
  printf("Tiempo de alocar A en GPU: %ld:%ld\n", t.tv_sec, t.tv_usec);
  assert(gettimeofday (&t0b, NULL) == 0);
  cudaMalloc ((void**) &Bd, sizeVectorEnBytes);
  cudaMemcpy (Bd, B, sizeVectorEnBytes, cudaMemcpyHostToDevice);
  //tiempos de alocaión
  assert (gettimeofday (&tb, NULL) == 0);
  timersub(&tb, &t0b, &t);
  printf("Tiempo de alocar B en GPU: %ld:%ld\n", t.tv_sec, t.tv_usec);
  // Ubicar C en la GPU
  assert (gettimeofday (&t0c, NULL) == 0);
  cudaMalloc ((void**) &Cd, sizeVectorEnBytes);
  //tiempos de alocación
  assert (gettimeofday (&tc, NULL) == 0);
  timersub(&tc, &t0c, &t);
  printf("Tiempo de alocar C en GPU: %ld:%ld\n", t.tv_sec, t.tv_usec);
  assert (gettimeofday (&ttf,NULL)==0);
  timersub(&ttf,&t0,&t);
  printf("Tiempo total de enviar A y b y asignar c: %ld:%ld\n", t.tv_sec,t.tv_usec);
  // Invocar al kernel
  // Poner la expresion adecuada
  assert (gettimeofday (&t0k, NULL) == 0);
  sumVectorKernel<<<cardinalidadVector/anchoBloque, anchoBloque>>>(Ad, Bd, Cd);
  assert (cudaDeviceSynchronize() == 0);
  //tiempo de ejecución de kernel
  assert (gettimeofday (&tk, NULL) == 0);
  timersub(&tk, &t0k, &t);
  printf("Tiempo de ejecución del kernel: %ld:%ld\n", t.tv_sec, t.tv_usec);
  // Transferir C desde la GPU
  assert(gettimeofday (&t0cp, NULL) == 0);
  cudaMemcpy (C, Cd, sizeVectorEnBytes, cudaMemcpyDeviceToHost);
  assert(gettimeofday (&tcp, NULL) == 0);
  //tiempo en realocar C en la CPU
  timersub(&tcp, &t0cp, &t);
  printf("Tiempo en realocar C en el CPU: %ld:%ld\n", t.tv_sec, t.tv_usec);
  // Liberar matrices en la GPU
  cudaFree(Ad); cudaFree(Bd); cudaFree(Cd);
  assert (gettimeofday (&tf, NULL) == 0);

  timersub (&tf, &t0, &t);
  printf ("Tiempo total = %ld:%ld \n", t.tv_sec, t.tv_usec);
  for (k=0; k<cardinalidadVector; k++) assert (C[k] == (A[k] + B[k]));
  printf("OK\n");
  return 0;
}
