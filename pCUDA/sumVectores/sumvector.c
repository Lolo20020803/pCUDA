//------------------------------------------------------------------+
// PCM. Arquitecturas Paralelas Curso 15/16 EUI           30/03/16  |
//                                                                  |
// sumvector.c: Prueba de suma de un vector en la CPU               |
//              para luego comparar con version GPU                 |
//------------------------------------------------------------------+

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

//-------------------------------------------------------------------
void initVector (double *V, int cardinalidad, double valor) {
  int i;
  V[0] = valor;
  for (i=1; i<cardinalidad; i++) V[i] = V[i-1] + valor;
}

//-------------------------------------------------------------------
int main (int argc, char *argv[])
{
  struct timeval t0, tf, t;
  double  *A;
  double suma;
  int    i, n;

  n = atoi(argv[1]);
  A = (double *) malloc (n*sizeof(double));
  initVector (A, n, 0.00001f );

  assert (gettimeofday (&t0, NULL) == 0);
  suma = 0.0;
  for (i=0; i<n; i++) {
    suma += (double ) A[i];
  }
  assert (gettimeofday (&tf, NULL) == 0);
  timersub (&tf, &t0, &t);
  printf ("Tiempo en CPU = %ld:%ld \n", t.tv_sec, t.tv_usec);
  printf ("Suma = %f\n", suma);
  return 0;
}
