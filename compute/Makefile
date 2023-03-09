CFLAGS = -I. -Wall -Werror

BUILD_DIR = /home/eileen/ane/build
LIBS = -I/usr/include/libdrm \
	-I/home/eileen/ane/ane/src/include \
	-I/home/eileen/ane/anelib/include

.PHONY: all clean

all: anec sqrt atan2 matmul
anec:
	bash gen_anec.sh
	python3 gen_data.py
sqrt:
	gcc $(CFLAGS) $(LIBS) $(BUILD_DIR)/anelib.o \
		sqrt.anec.o sqrt.c -o sqrt.out
atan2:
	gcc $(CFLAGS) $(LIBS) $(BUILD_DIR)/anelib.o \
		atan2.anec.o atan2.c -o atan2.out
matmul:
	gcc $(CFLAGS) $(LIBS) $(BUILD_DIR)/anelib.o \
		matmul.anec.o matmul.c -o matmul.out

clean:
	rm -f *.o *.out *.anec anec_*.h *.tile *.arr
