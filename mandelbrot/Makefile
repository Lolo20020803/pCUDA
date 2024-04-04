#------------------------------------------------------------------#
# PCM. Arquitecturas Paralelas Curso 15/16 EUI           01/03/16  #
#                                                                  #
# Makefile para el desarrollo de las pruebas de los programas      #
#          relacionados con el aprendizaje de CUDA                 #
#------------------------------------------------------------------#

CC      = /usr/local/cuda-10.2/bin/nvcc -arch=sm_53
CLINKER = /usr/local/cuda-10.2/bin/nvcc
LIBS	=	`pkg-config --libs   gtk+-3.0`
#CFLAGS	=	`pkg-config --cflags gtk+-3.0`

todo:	mandelsec mandelpar

mandelsec: mandelsec.o mandelbrot.o plano2D.o mapapixel.o
	$(CLINKER) $(LIBS) -o mandelsec \
	mandelsec.o mandelbrot.o plano2D.o mapapixel.o

mandelpar: mandelpar.o  mapapixel.o
	$(CLINKER) $(LIBS) -o mandelpar \
	mandelpar.o mapapixel.o

mandelpar.o: mandelpar.cu mapapixel.cuh
	$(CC) -O3 -c mandelpar.cu

mandelsec.o: mandelsec.cu mandelbrot.cuh plano2D.cuh mapapixel.cuh
	$(CC) -O3 -c mandelsec.cu

mandelbrot.o: mandelbrot.cu mandelbrot.cuh
	$(CC) -O3 -c mandelbrot.cu

plano2D.o: plano2D.cu plano2D.cuh
	$(CC) -O3 -c plano2D.cu

mapapixel.o: mapapixel.cu mapapixel.cuh
	$(CC) -O3 \
	-I/usr/include/gtk-3.0 -I/usr/include/at-spi2-atk/2.0 \
	-I/usr/include/at-spi-2.0 -I/usr/include/dbus-1.0 \
	-I/usr/lib/aarch64-linux-gnu/dbus-1.0/include \
	-I/usr/include/gtk-3.0 -I/usr/include/gio-unix-2.0/ \
	-I/usr/include/cairo -I/usr/include/pango-1.0 \
	-I/usr/include/harfbuzz -I/usr/include/pango-1.0 \
	-I/usr/include/atk-1.0 -I/usr/include/cairo \
	-I/usr/include/pixman-1 -I/usr/include/freetype2 \
	-I/usr/include/libpng16 -I/usr/include/freetype2 \
	-I/usr/include/libpng16 -I/usr/include/gdk-pixbuf-2.0 \
	-I/usr/include/libpng16 -I/usr/include/glib-2.0 \
       	-I/usr/lib/aarch64-linux-gnu/glib-2.0/include \
	-c mapapixel.cu

borrar:
	rm *.o	mandelsec mandelpar
