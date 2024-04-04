//-------------------------------------------------------------------+
// PCM. Arquitecturas Avanzadas Curso 19/20 ETSISI         03/02/20  |
//                                                                   |
// mapapixel.c: Modulo que permite trabajar en modo grafico con GTK  |
//-------------------------------------------------------------------+

#include <sys/stat.h>
#include <fcntl.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <gtk/gtk.h>

#include "mapapixel.cuh"

#define MAX_LONG_LINEA 128
#define DELTA_COLOR    256

static GtkWidget *window;
static GtkWidget *pizarra;
static GtkWidget *vbox;
static GtkWidget *button;
static GtkWidget *buttonBox;

static cairo_t   *miCR;

static int laProfundidad    = 8;
static int elDesplazamiento = 0;
static int elNumColores     = 16777216;
static int mascara          = 0xFF;
static int ultimoColor      = -1;
static int colorVerdadero   = TRUE;

static mapiFuncionClick  userfClick;
static mapiFuncionCierre userfCierre;
static mapiFuncionAccion userfAccion;

typedef enum {Eninguno, Eclick, Ecierre, Eaccion} tEvento;
static tEvento elEvento = Eninguno;
static short filaClick, colClick;
static int   botonIzquierdoClick; 

//--------------------------------------------------------------------
void leerLinea (int fd, char *linea) {
  int i;

  for (i=0; i<MAX_LONG_LINEA; i++) {
    assert (read (fd, &linea[i], 1) == 1);
    if (linea[i] == '\n') {
      linea[i] = 0;
      return;
    }
  }
  assert (0);
}

//--------------------------------------------------------------------
static gint clickRaton (GtkWidget      *widget,
                        GdkEventButton *evento,
			gpointer	data)
{
  if ((evento->button == GDK_BUTTON_PRIMARY) || (evento->button == GDK_BUTTON_SECONDARY)) {
    elEvento  = Eclick;
    filaClick = evento->y;
    colClick  = evento->x;
    botonIzquierdoClick = (evento->button == 1);
    gtk_widget_queue_draw (window);
  }
  return 0;
}

//--------------------------------------------------------------------
static void quit ()
{
  if (userfCierre != NULL) {
    elEvento = Ecierre;
    gtk_widget_queue_draw (window);
  }
  exit (0);
}

//--------------------------------------------------------------------
static void accion ()
{
  elEvento = Eaccion;
  gtk_widget_queue_draw (window);
}

//--------------------------------------------------------------------
static void ponerColor (cairo_t *cr, int color) {
  tipoRGB  cRGB; 

  if (ultimoColor != color) {
    if (colorVerdadero) {
        mapiColorRGB (color, &cRGB);
    } else {
        cRGB.verde = color % 256;
        cRGB.rojo  = color % 256;
        cRGB.azul  = color % 256;
    }
    cairo_set_source_rgb (cr, cRGB.rojo/512.0, cRGB.verde/256.0 , cRGB.azul/128.0);
    //cairo_set_source_rgb (cr, cRGB.rojo/32768.0, cRGB.verde/1024.0 , cRGB.azul/32.0);
    ultimoColor = color;
  }
}

//--------------------------------------------------------------------
static gboolean dibujar (GtkWidget *widget, cairo_t *cr, gpointer user_data) {

  cairo_set_source_rgb (cr, 0.2, 0.02, 0.002);
  cairo_set_line_width (cr, 2.0);
  miCR = cr;
  switch (elEvento) {
    case Eaccion : userfAccion(); break;
    case Eclick  : userfClick (filaClick, colClick, botonIzquierdoClick);
                   break;
    case Ecierre : userfCierre (); break;
    case Eninguno: break;
  } 
  return FALSE;
}

//    FUNCIONES EXPORTADAS

//--------------------------------------------------------------------
int mapiInicializar (short filas, short columnas,
                     mapiFuncionAccion fAccion,
                     mapiFuncionClick  fClick,
                     mapiFuncionCierre fCierre)
{
  gtk_init (NULL, NULL);

  // Crear la ventana principal
  window = gtk_window_new (GTK_WINDOW_TOPLEVEL);
  gtk_window_set_title (GTK_WINDOW(window), "arqAva");
  gtk_window_set_default_size (GTK_WINDOW(window), columnas, filas);

  // Crear area de dibujo
  vbox = gtk_grid_new ();
  gtk_container_add (GTK_CONTAINER (window), vbox);

  pizarra = gtk_drawing_area_new();
  gtk_widget_set_size_request (pizarra, columnas, filas);
  gtk_grid_attach (GTK_GRID (vbox), pizarra, 0, 0, 1, 1);
  g_signal_connect (G_OBJECT(pizarra), "draw", G_CALLBACK (dibujar), NULL);

  if (fClick != NULL) {
    userfClick = fClick;
    g_signal_connect (pizarra, "button_press_event", G_CALLBACK (clickRaton), NULL);
    gtk_widget_set_events (pizarra, gtk_widget_get_events(pizarra) | GDK_BUTTON_PRESS_MASK);
  }

  // Incorporar boton de Accion
  buttonBox = gtk_button_box_new (GTK_ORIENTATION_HORIZONTAL);
  gtk_grid_attach (GTK_GRID (vbox), buttonBox, 0, filas, columnas, 1);
  button = gtk_button_new_with_label ("Accion");
  userfAccion = fAccion;
  g_signal_connect (button, "clicked", G_CALLBACK (accion), NULL);
  gtk_container_add (GTK_CONTAINER (buttonBox), button);

  // Incorporar accion de cierre de ventana
  userfCierre = fCierre;
  g_signal_connect (window, "destroy", G_CALLBACK (quit), NULL);

  gtk_widget_show_all (window);
  gtk_main();
  return 0;
}

//--------------------------------------------------------------------
void mapiProfundidadColor (unsigned short profundidad) {
  int i;

  laProfundidad    = profundidad;
  elDesplazamiento = 8 - profundidad;
  elNumColores  = 1 << (profundidad*3); 
  mascara = 1;
  for (i=1; i<profundidad; i++) mascara  = (mascara  << 1) | 1;
}

//--------------------------------------------------------------------
int mapiNumColoresDefinidos (void) {
  return elNumColores;
}

//--------------------------------------------------------------------
void mapiColorRGB (int color, tipoRGB *colorRGB) {
  colorRGB->rojo  = (color & mascara ) << elDesplazamiento;
  color = color >> (laProfundidad - 1);
  colorRGB->verde = (color & mascara ) << elDesplazamiento;
  color = color >> (laProfundidad - 1);
  colorRGB->azul  = (color & mascara ) << elDesplazamiento;
}

//--------------------------------------------------------------------
void mapiDibujarPunto (short fila, short columna, int color) {
  ponerColor (miCR, color);
  cairo_move_to (miCR, columna-1, fila);
  cairo_line_to (miCR, columna  , fila);
  cairo_stroke (miCR);

}

//--------------------------------------------------------------------
void mapiDibujarLinea (short fila1, short columna1,
                       short fila2, short columna2, int color) {

  ponerColor (miCR, color);
  cairo_line_to (miCR, columna1, fila1);
  cairo_line_to (miCR, columna2, fila2);
  cairo_stroke(miCR);
}

//--------------------------------------------------------------------
void mapiDibujarRectangulo (short fila1, short columna1,
                            short ancho, short largo, int color) {

  ponerColor (miCR, color);
  cairo_rectangle (miCR, columna1, fila1, ancho, largo);
  cairo_fill (miCR);
}


//--------------------------------------------------------------------
void mapiCrearMapa (tipoMapa *unMapa)
{
  int numBytes;

  assert (   (unMapa->filas <= MAX_FILAS)
	  && (unMapa->columnas <= MAX_COLUMNAS));
  numBytes = unMapa->filas * unMapa->columnas;
  if (unMapa->elColor == colorRGB) numBytes = numBytes * 3;
  unMapa->pixels = (char *) malloc (numBytes);
  bzero (unMapa->pixels, numBytes);
}

//--------------------------------------------------------------------
void mapiLeerMapa (char *fichero, tipoMapa *unMapa)
{
  int fd, numBytes, intColumnas, intFilas;
  char linea[MAX_LONG_LINEA];

  if (*fichero != '/') {
      strcpy (linea, getenv("PWD"));
      strcat (linea, "/");
      strcat (linea, fichero);
  } else
      strcpy (linea, fichero);

  assert ((fd = open (linea, O_RDONLY)) != -1);
  leerLinea(fd, linea);
  if      (strcmp (linea, "P5") == 0)
    unMapa->elColor = escalaGrises;
  else if (strcmp (linea, "P6") == 0)
    unMapa->elColor = colorRGB;
  else
    assert (0);
  leerLinea(fd, linea);
  assert (sscanf (linea, "%d %d", &intColumnas, &intFilas) == 2);
  assert ((intFilas <= MAX_FILAS) && (intColumnas <= MAX_COLUMNAS));
  unMapa->filas    = intFilas;
  unMapa->columnas = intColumnas;
  leerLinea(fd, linea);
  numBytes = unMapa->filas * unMapa->columnas;
  if (unMapa->elColor == colorRGB) numBytes = numBytes * 3;
  unMapa->pixels = (char *)malloc (numBytes);
  assert (read (fd, unMapa->pixels, numBytes) == numBytes);
  assert (close(fd) == 0);
}

//--------------------------------------------------------------------
void mapiPonerPuntoGris (tipoMapa *unMapa,
		         short fila, short columna, short tonalidad) {
  guchar *elPixel = (guchar *) unMapa->pixels;

  elPixel = elPixel + (fila * unMapa->columnas) + columna;
  *elPixel = tonalidad;
}

//--------------------------------------------------------------------
void mapiPonerPuntoRGB  (tipoMapa *unMapa,
		         short fila, short columna, tipoRGB color) {
  guchar *elPixel = (guchar *) unMapa->pixels;

  elPixel = elPixel + (((fila * unMapa->columnas) + columna) * 3);
  *elPixel++ = color.rojo;
  *elPixel++ = color.verde;
  *elPixel++ = color.azul;
}

//--------------------------------------------------------------------
void mapiDibujarMapa (tipoMapa *unMapa) {
  GdkPixbuf *pixbuf;

  if (unMapa->elColor == escalaGrises)
    //gdk_draw_gray_image (pizarra->window,
		         //pizarra->style->fg_gc[GTK_STATE_NORMAL],
			 //0, 0, unMapa->columnas, unMapa->filas,
			 //GDK_RGB_DITHER_NONE, (guchar *) unMapa->pixels,
			 //unMapa->columnas);
   pixbuf = gdk_pixbuf_new_from_data ( (unsigned char *) unMapa->pixels,
                         GDK_COLORSPACE_RGB, FALSE, 8, unMapa->columnas,
                         unMapa->filas, unMapa->columnas, NULL, NULL);
  else {
   pixbuf = gdk_pixbuf_new_from_data ( (unsigned char *) unMapa->pixels,
                         GDK_COLORSPACE_RGB, FALSE, 8, unMapa->columnas,
                         unMapa->filas, unMapa->columnas*3, NULL, NULL);
  }
  gdk_cairo_set_source_pixbuf(miCR, pixbuf, 0, 0);
  cairo_paint(miCR);

}

//--------------------------------------------------------------------
void mapiGrabarMapa (char *fichero, tipoMapa *unMapa)
{
  int  fd, numBytes;
  char linea[MAX_LONG_LINEA];

  if (*fichero != '/') {
      strcpy (linea, getenv("PWD"));
      strcat (linea, "/");
      strcat (linea, fichero);
  } else
      strcpy (linea, fichero);
  assert ((fd = open (linea, O_CREAT | O_WRONLY, S_IRWXU)) != -1);
  if (unMapa->elColor == escalaGrises) {
    numBytes = unMapa->filas * unMapa->columnas;
    sprintf (linea, "P5\n");
  } else {
    numBytes = unMapa->filas * unMapa->columnas * 3;
    sprintf (linea, "P6\n");
  }
  assert (write (fd, linea, 3) == 3); 
  assert (sprintf(linea, "%4d %4d\n", unMapa->columnas, unMapa->filas) == 10);
  assert (write (fd, linea, 10) == 10); 
  assert (sprintf(linea, "255\n") == 4);
  assert (write (fd, linea, 4) == 4); 
  assert (write (fd, unMapa->pixels, numBytes) == numBytes); 
  assert (close(fd) == 0);
}

