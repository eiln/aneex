
CFLAGS = -I. -Wall -Werror

ANE_DIR = /home/eileen/ane
BUILD_DIR = /home/eileen/ane/build

LIBS = -I/usr/include/libdrm \
	-I$(ANE_DIR)/ane/src/include \
	-I$(ANE_DIR)/anelib 

.PHONY: pyane clean

pyane:
	bash compile.sh
	gcc -shared -pthread -fPIC -fno-strict-aliasing \
		$(CFLAGS) $(LIBS) -I/usr/include/python3.10 \
		$(BUILD_DIR)/anelib.o \
		fcn.anec.o srgan.anec.o yolov5.anec.o \
		pyane.c -o pyane.so

clean:
	rm -f *.o *.so 
	rm -f *.anec anec_*.h pyane_*.h pyane.c
