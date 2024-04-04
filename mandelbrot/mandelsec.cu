//------------------------------------------------------------------+
// PCM. Arquitecturas Paralelas Curso 15/16 EUI           04/03/16  |
//                                                                  |
// mandelsec.c: Programa que pinta el conjunto de Mandelbrot al     |
//              hacer click con el boton izquierdo o el derecho.    |
//              Haciendo click con el boton izquierdo, se recalcula |
//              y repinta el dibujo pero aumentando la imagen por   |
//              2 y centrandola donde se pulso. Si el click se hace |
//              con el boton derecho, se disminuye por dos.         |
//              Si se pulsa el boton de accion, se recalcula el     |
//              conjunto de Mandelbrot.                             |
//------------------------------------------------------------------+

#include <assert.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

#include "mapapixel.cuh"
#include "plano2D.cuh"
#include "mandelbrot.cuh"

#define TRUE  1
#define FALSE 0

static int dibujado = FALSE;
static tipoMapa miMapa;

#define FILAS     960
#define COLUMNAS  960

// Coordenadas del mandelbrot inicial en miniatura
static double miAltura = 0.000305;
static double centroX  = -0.813997;
static double centroY  =  0.194129;

//--------------------------------------------------------------------
static void dibujar()
{
  int fila, columna, color;
  double X, Y;
  struct timeval t0, t1, t;

  assert (gettimeofday(&t0, NULL) == 0);
  planoMapear (FILAS, COLUMNAS,
               centroX, centroY, miAltura);
  for (fila=0; fila<FILAS; fila++) {
    for (columna=0; columna<COLUMNAS; columna++) {
      planoPixelAPunto (fila, columna, &X, &Y);
      color = mandelbrot(X, Y);
      mapiDibujarPunto (fila, columna, color);
    }
  }
  assert (gettimeofday(&t1, NULL) == 0);
  timersub(&t1, &t0, &t);
  printf("Tiempo => %ld:%ld (seg:mseg)\n", t.tv_sec, t.tv_usec/1000);
}

//--------------------------------------------------------------------
static void clickRaton (short fila, short columna, int botonIzquierdo)
{
  if (!dibujado) {
      dibujar();
      dibujado = TRUE;
  } else {
      planoPixelAPunto (fila, columna, &centroX, &centroY);
      if (botonIzquierdo)
        miAltura = miAltura / 2.0;  // Profundizar
      else
        miAltura = miAltura * 2.0;  // Alejarse
      dibujar ();
  }
}

//--------------------------------------------------------------------
int main(int argc, char *argv[]) {

  int profundidadColor;

  if      (argc == 1) profundidadColor = 3;
  else if (argc == 2) profundidadColor = atoi(argv[1]);
  else {
    printf ("Uso: mandelsec [profundidadColor]\n");
    return(0);
  }
  mapiProfundidadColor(profundidadColor);
  mandelInit (mapiNumColoresDefinidos());
  miMapa.elColor  = colorRGB;
  miMapa.filas    = FILAS;
  miMapa.columnas = COLUMNAS;
  mapiCrearMapa (&miMapa);
  mapiInicializar (FILAS, COLUMNAS, dibujar, clickRaton, NULL);
  return 0;
}

