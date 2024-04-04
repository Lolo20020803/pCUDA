//------------------------------------------------------------------+
// PCM. Arquitecturas Paralelas Curso 20/21 ETSISI        22/04/21  |
//                                                                  |
// mulmatriz.c: Prueba de multiplicacion de matriz x matriz en la   |
//              CPU para luego comparar con version GPU             |
//------------------------------------------------------------------+

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

//-------------------------------------------------------------------
void initMatriz (double *M, int card, double valor) {
  int i;
  for (i=0; i<card; i++) {
    M[i] = valor;
    valor += 0.1;
  }
}

//-------------------------------------------------------------------
int main (int argc, char *argv[])
{
  int filA, colA, filB, colB, filC, colC;
  struct timeval t0, tf, t;
  double *A, *B, *C, suma;
  int    f, c, k;

  filA = atoi(argv[1]);
  colA = filA;
  filB = filA;
  colB = filA;
  filC = filA;
  colC = filA;
  A = (double *) malloc (filA*colA*sizeof(double));
  B = (double *) malloc (filB*colB*sizeof(double));
  C = (double *) malloc (filC*colC*sizeof(double));
  initMatriz (A, filA*colA, 1.0f );
  initMatriz (B, filB*colB, 0.01f);

  assert (gettimeofday (&t0, NULL) == 0);
  for (f=0; f<filA; f++)
    for (c=0; c<colB; c++) {
      suma = 0;
      for (k=0; k<colA; k++)
        suma += A[f*colA+k] * B[k*filB+c];
      C[f*colA+c] = suma;
    }
  assert (gettimeofday (&tf, NULL) == 0);

  timersub (&tf, &t0, &t);
  printf ("Tiempo = %ld:%ld \n", t.tv_sec, t.tv_usec);
  return 0;
}
