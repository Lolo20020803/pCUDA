//------------------------------------------------------------------+
// PCM. Arquitecturas Paralelas Curso 15/16 EUI           04/03/16  |
//                                                                  |
// plano2D.h: Modulo que permite correlacionar una ventana grafica  |
//            formada por pixels (filas * columnas), con un plano   |
//            de dos dimensiones con coordenadas cartesianas, donde |
//            se determina la escala de los ejes y la ubicacion del |
//            centro de la ventana en el plano cartesiano.          |
//------------------------------------------------------------------+

//------------------------------------------------------------------+
// Se mapea la ventana grafica de tamanio filas * columnas, con un  |
// plano cuya altura se indica en "laAltura" y en el cual, las      |
// coordenadas del centro de la ventana se corresponden con las     |
// coordenadas Cx y Cy pasadas como parametro.                      |
// El ancho del plano visualizado se calcula a partir de los para-  |
// metros: filas, columnas y laAltura.                              |
//------------------------------------------------------------------+
void planoMapear (short filas, short columnas,
                  double Cx, double Cy, double laAltura);

//------------------------------------------------------------------+
// La siguiente funcion devuelve en X e Y, las coordenadas carte-   |
// sianas correspondientes al pixel (fila, columna) de la ventana   |
// grafica.                                                         |
//------------------------------------------------------------------+
void planoPixelAPunto (short fila, short columna, double *X, double *Y);

