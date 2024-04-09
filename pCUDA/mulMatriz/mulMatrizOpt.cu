//------------------------------------------------------------------+
// PCM. Arquitecturas Paralelas Curso 20/21 ETSISI        22/04/21  |
//                                                                  |
// mulmatrizOpt.c: Prueba de multiplicacion de matriz x matriz en la|
//                 CPU para luego comparar con version GPU          |
//                    VERSION OPTIMIZADA [Acceso a memoria]         |
//------------------------------------------------------------------+

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define anchoBloque 16  // En 2D => 16x16 = 256 threads
//-------------------------------------------------------------------
void initMatriz (double *M, int card, double valor) {
  int i;
  for (i=0; i<card; i++) {
    M[i] = valor;
    valor += 0.1;
  }
}

//-------------------------------------------------------------------
__global__ void mulMatrizKernel (double *Ad, double *Bd, double *Cd, int card) {
  __shared__ double Ads[anchoBloque][anchoBloque];
  __shared__ double Bds[anchoBloque][anchoBloque];
  int tx  = threadIdx.x;
  int ty  = threadIdx.y;
  int fil = blockIdx.y*anchoBloque+ty;
  int col = blockIdx.x*anchoBloque+tx;
  int k,m;
  double Cvalor = 0.0;

  for (m=0; m<card/anchoBloque; m++) {
    Ads[ty][tx] = Ad[fil*card + (m*anchoBloque + tx)];
    Bds[ty][tx] = Bd[(m*anchoBloque + ty)*card + col];
    __syncthreads();
    for (k=0; k<anchoBloque; k++)
      Cvalor += Ads[ty][k] * Bds[k][tx];
    __syncthreads();
  }
  Cd[fil * card + col] = Cvalor;
}

//-------------------------------------------------------------------
int main (int argc, char *argv[])
{
  int filA, colA, filB, colB, filC, colC;
  struct timeval t0, tf, t;
  double *A, *B, *C;
  double *Ad, *Bd, *Cd;
  int    sizeA, sizeB, sizeC, f, k;
  double valor;

  filA = atoi(argv[1]);
  colA = filA;
  filB = filA;
  colB = filA;
  filC = filA;
  colC = filA;
  sizeA = filA*colA*sizeof(double);
  sizeB = filB*colB*sizeof(double);
  sizeC = filC*colC*sizeof(double);
  A = (double *) malloc (sizeA);
  B = (double *) malloc (sizeB);
  C = (double *) malloc (sizeC);
  initMatriz (A, filA*colA, 1.0f );
  initMatriz (B, filB*colB, 0.01f);

  assert (gettimeofday (&t0, NULL) == 0);
  // Transferir A y B a la GPU
  cudaMalloc ((void**) &Ad, sizeA);
  cudaMemcpy (Ad, A, sizeA, cudaMemcpyHostToDevice);
  cudaMalloc ((void**) &Bd, sizeB);
  cudaMemcpy (Bd, B, sizeB, cudaMemcpyHostToDevice);
  // Ubicar C en la CPU
  cudaMalloc ((void**) &Cd, sizeC);
  // Invocar al kernel
  dim3 dimGrid (filA/anchoBloque, filA/anchoBloque);
  dim3 dimBlock(anchoBloque, anchoBloque);
  mulMatrizKernel<<<dimGrid, dimBlock>>>(Ad, Bd, Cd, filA);
  cudaDeviceSynchronize();
  // Transferir C desde la GPU
  cudaMemcpy (C, Cd, sizeC, cudaMemcpyDeviceToHost);
  // Liberar matrices en la GPU
  cudaFree(Ad); cudaFree(Bd); cudaFree(Cd);
  assert (gettimeofday (&tf, NULL) == 0);

  timersub (&tf, &t0, &t);
  printf ("Tiempo = %ld:%ld \n", t.tv_sec, t.tv_usec);

  // Comprobamos diagonal principal
  for (f=0; f<filA; f++) {
    valor = 0.0;
    for (k=0; k<filA; k++)
      valor += A[f * colA + k] * B[k * colA + f];
    if (fabs(valor - C[f*colA+f]) > 0.1) {
      printf ("Error f=%d c=%d CPU = %lf GPU = %lf diff = %lf\n",
      f, f, valor, C[f*colA+f], valor-C[f*colA+f]);
      return 0;
    }
  }
  return 0;
}
