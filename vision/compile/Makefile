CFLAGS = -I. -Wall -Werror
LIBS = -I/usr/include/libdrm \
	-I/home/eileen/ane/ane/src/include \
	-I/home/eileen/ane/anelib/include

BUILD_DIR = /home/eileen/ane/build

.PHONY: pyane clean

pyane:
	bash gen_anec.sh
	gcc -shared -pthread -fPIC -fno-strict-aliasing \
		$(CFLAGS) $(LIBS) -I/usr/include/python3.10 \
		$(BUILD_DIR)/anelib.o \
		fcn.anec.o srgan.anec.o yolov5.anec.o \
		pyane.c -o pyane.so

clean:
	rm -f *.o *.so 
	rm -f *.anec anec_*.h pyane.c
