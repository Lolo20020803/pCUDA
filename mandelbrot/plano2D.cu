//------------------------------------------------------------------+
// PCM. Arquitecturas Paralelas Curso 15/16 EUI           04/03/16  |
//                                                                  |
// plano2D.c: Implementacion del modulo que correlaciona ventana    |
//            grafica y plano bidimensional y coordenadas carte-    |
//            sianas.                                               |
//------------------------------------------------------------------+

#include "plano2D.cuh"

static double planoVx; // Coordenadas del vertice inferior izquierdo
static double planoVy;

static short  planoFILAS;
static short  planoCOLUMNAS;
static double planoALTURA;
static double planoANCHURA;

static double planoFactorX;
static double planoFactorY;

//-------------------------------------------------------------------
void planoMapear (short filas, short columnas,
                  double Cx, double Cy, double laAltura)
{
  planoFILAS    = filas;
  planoCOLUMNAS = columnas;
  planoALTURA   = laAltura;
  planoANCHURA  = (planoCOLUMNAS * planoALTURA) / planoFILAS;
  planoVx       = Cx - (planoANCHURA / 2.0);
  planoVy       = Cy - (planoALTURA  / 2.0);
  planoFactorX  = planoANCHURA / (double) (planoCOLUMNAS - 1);
  planoFactorY  = planoALTURA  / (double) (planoFILAS    - 1);
}

//-------------------------------------------------------------------
void planoPixelAPunto (short fila, short columna, double *X, double *Y) {
  *X = (planoFactorX * (double) columna) + planoVx;
  *Y = (planoFactorY * ((double) (planoFILAS - 1) - (double) fila)) + planoVy;
}

