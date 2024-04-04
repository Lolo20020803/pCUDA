/*-----------------------------------------------------------------*/
/* PCM. Arquitecturas Paralelas Curso 15/16 EUI           29/03/16 */
/*                                                                 */
/* bitonicSec.c: Version secuencial del algoritmo de ordenacion    */
/*               por ordenacion y mezcla bitonic                   */
/*-----------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>

static int cardinalidad, ascendente;

void visualizar (int *v) {
  int i;

  printf ("\n--------------------------------------------\n");
  for (i=0; i<cardinalidad; i++) {
    printf ("%5d ", v[i]);
    if (((i+1)%10) == 0) printf ("\n");
  }
}

int main (int argc, char *argv[])
{
  struct timeval t0, tf, t;
  int i, j, iA, iB, maxEntero;
  int paso, salto, ultimoPaso, numComparaciones;
  int *vOrg; 

  void comparar (int indA, int indB) {
    int tmp;
  
    if (   ( ascendente && (vOrg[indA] > vOrg[indB]))
        || (!ascendente && (vOrg[indA] < vOrg[indB])) ) {
      tmp        = vOrg[indA];
      vOrg[indA] = vOrg[indB];
      vOrg[indB] = tmp;
    }
  }

  if (argc != 2) {
    printf ("Uso: bitonicSec cardinalidad\n");
    exit (0);
  }
  cardinalidad = atoi (argv[1]);
  ultimoPaso = cardinalidad / 2;
  maxEntero = cardinalidad * 2;
  vOrg = malloc (cardinalidad * sizeof(int));
  for (i=0; i<cardinalidad; i++)
      vOrg[i] = random() % maxEntero;
  visualizar(vOrg);

  assert (gettimeofday (&t0, NULL) == 0);
  // Construir secuencia bitonica y ordenarla
  for (paso=1; paso<=ultimoPaso; paso*=2) {
    numComparaciones = 0;
    for (salto=paso; salto>0; salto/=2) {
      ascendente = 1;
      for (i=0; i<cardinalidad; i+=salto*2) {
        iA = i; iB = iA+salto;
        for (j=0; j<salto; j++) {
          comparar(iA, iB);
          iA++; iB++;
          numComparaciones++;
        }
        if ((numComparaciones % paso) == 0)
          ascendente = (ascendente ? 0 : 1);
      }
    }
  }
  assert (gettimeofday (&tf, NULL) == 0);

  timersub (&tf, &t0, &t);
  visualizar(vOrg);
  printf ("\nTiempo (seg:mseg): %ld:%ld\n", t.tv_sec, t.tv_usec/1000);
  // Comprobar OK
  for (i=1; i<cardinalidad; i++) {
    if (vOrg[i] < vOrg[i-1]) {
      printf ("Error vOrg[%d] < vOrg[ant] => %d y %d\n", i, vOrg[i], vOrg[i-1]);
      exit (0);
    }
  }
  printf ("OK\n");
  exit (0);
}
