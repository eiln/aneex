
> pyane.c
for name in fcn srgan yolov5; do
	anec.py -a hwx/$name.hwx
	ld -r -b binary -o $name.anec.o $name.anec
	echo "#include \"pyane_$name.h\"" >> pyane.c
done
echo "int main(void){return 0;}" >> pyane.c
