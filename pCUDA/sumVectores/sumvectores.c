//------------------------------------------------------------------+
// PCM. Arquitecturas Paralelas Curso 15/16 EUI           29/02/16  |
//                                                                  |
// sumvectores.c: Prueba de suma de dos vectores en la CPU          |
//                para luego comparar con version GPU               |
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
int main (int argc, char *argv[])
{
  struct timeval t0, tf, t;
  float  *A, *B, *C;
  double suma;
  int    i, n;

  n = atoi(argv[1]);
  A = (float *) malloc (n*sizeof(float));
  B = (float *) malloc (n*sizeof(float));
  C = (float *) malloc (n*sizeof(float));
  initVector (A, n, 0.001f );
  initVector (B, n, 0.002f);

  assert (gettimeofday (&t0, NULL) == 0);
  for (i=0; i<n; i++) C[i] = A[i] + B[i];
  assert (gettimeofday (&tf, NULL) == 0);

  suma = 0.0;
  for (i=0; i<n; i++) suma += (double ) C[i];
  timersub (&tf, &t0, &t);
  printf ("Tiempo en CPU = %ld:%ld \n", t.tv_sec, t.tv_usec);
  printf ("Suma = %f\n", suma);
  return 0;
}
