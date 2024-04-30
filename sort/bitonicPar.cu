/*-----------------------------------------------------------------*/
/* PCM. Arquitecturas Paralelas Curso 15/16 EUI           17/03/16 */
/*                                                                 */
/* bitonicPar.cu: Version paralela del algoritmo de ordenacion     */
/*                por ordenacion y mezcla bitonic                  */
/*-----------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>

#define anchoBloque 1024  // Threads por bloque

static int cardinalidad, *vOrg, *vOrgd; 

//-----------------------------------------------------------------
void visualizar (int *v) {
  int i;

  printf ("\n--------------------------------------------\n");
  for (i=0; i<cardinalidad; i++) {
    printf ("%5d ", v[i]);
    if (((i+1)%10) == 0) printf ("\n");
  }
}

//-----------------------------------------------------------------
__global__ void bitonicKernel (int *Ad, int paso, int salto) {
  int yo = blockIdx.x*anchoBloque+threadIdx.x;
  int iA, iB, ascendente, tmp;
  

  int miAgrupacion = yo / salto; // yo = 7 en salto 2 => agrup =3

  // iA = saltoAnt * miAgrup + yoEnLaAgrup
  iA = ((salto * 2) * miAgrupacion) + (yo % salto);        // Expresion que solo depende yo y de salto
  iB = iA + salto;
  ascendente =((iA /(paso*2))%2==0) ; // Expresion que solo depende iA y de paso
  if (   ( ascendente && (Ad[iA] > Ad[iB]))
      || (!ascendente && (Ad[iA] < Ad[iB])) ) {
    tmp    = Ad[iA];
    Ad[iA] = Ad[iB];
    Ad[iB] = tmp;
  }
}

//-----------------------------------------------------------------
int main (int argc, char *argv[])
{
  struct timeval t0, tf, t;
  int i, maxEntero;
  int paso, salto, ultimoPaso, sizeVectorEnBytes, numBloques;

  if (argc != 2) {
    printf ("Uso: bitonicPar cardinalidad\n");
    exit (0);
  }

  cardinalidad = atoi (argv[1]);
  if (cardinalidad < (anchoBloque*2)) {
    printf ("cardinalidad debe ser >= %d\n", anchoBloque*2);
    exit (0);
  }
  numBloques        = (cardinalidad / anchoBloque) / 2;
  sizeVectorEnBytes = cardinalidad * sizeof(int);
  ultimoPaso        = cardinalidad / 2;
  maxEntero         = cardinalidad * 2;
  vOrg = (int *) malloc (sizeVectorEnBytes);
  for (i=0; i<cardinalidad; i++)
      vOrg[i] = random() % maxEntero;
  //visualizar(vOrg);

  assert (gettimeofday (&t0, NULL) == 0);
  // Transferir vOrg a la GPU
  cudaMalloc ((void**) &vOrgd, sizeVectorEnBytes);
  cudaMemcpy (vOrgd, vOrg, sizeVectorEnBytes, cudaMemcpyHostToDevice);
  // Construir y ordenar secuencia bitonica
  for (paso=1; paso<=ultimoPaso; paso*=2) {
    for (salto=paso; salto>0; salto/=2) {
      // Invocar al kernel
      bitonicKernel<<<numBloques, anchoBloque>>>(vOrgd, paso, salto);
      assert (cudaDeviceSynchronize() == 0);
    }
  }
  // Transferir vOrg desde la GPU
  cudaMemcpy (vOrg, vOrgd, sizeVectorEnBytes, cudaMemcpyDeviceToHost);
  // Liberar vector en la GPU
  cudaFree(vOrgd);
  assert (gettimeofday (&tf, NULL) == 0);

  //visualizar(vOrg);
  timersub (&tf, &t0, &t);
  printf ("\nTiempo total (seg:mseg): %ld:%ld\n", t.tv_sec, t.tv_usec/1000);
  // Comprobar OK
  for (i=1; i<cardinalidad; i++) {
    if (vOrg[i] < vOrg[i-1]) {
      printf ("Error vOrg[%d] < vOrg[ant] => %d y %d\n", i, vOrg[i], vOrg[i-1]);
      exit (0);
    }
  }
  printf ("Ordenacion OK\n");
  exit (0);
}
