
for name in fcn srgan yolov5; do
	anec.py -s -d hwx/$name.hwx
	ld -r -b binary -o $name.anec.o $name.anec
	echo "#include \"pyane_$name.h\""
done
