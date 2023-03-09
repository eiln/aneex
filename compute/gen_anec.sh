#!/usr/bin/bash

for name in atan2 matmul sqrt; do
	anec.py -s -d hwx/$name.hwx
	ld -r -b binary -o $name.anec.o $name.anec
done
