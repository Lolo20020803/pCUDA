//------------------------------------------------------------------+
// PCM. Arquitecturas Paralelas Curso 15/16 EUI           04/03/16  |
//                                                                  |
// mandelbrot.h: Modulo que obtiene el valor de divergencia del     |
//               conjunto de mandelbrot para el punto X,Y iterando  |
//               hasta un maximo definido por "mandelInit"          |
//------------------------------------------------------------------+


void mandelInit (int maxIteraciones);

int  mandelbrot (double X, double Y);

