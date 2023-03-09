#!/usr/bin/python3

import numpy as np

np.random.seed(0)

A = np.random.random((1, 1, 1024, 2048)).astype(np.float16)
B = np.random.random((1, 1, 1024, 2048)).astype(np.float16)
C = np.random.random((1, 1, 2048, 4096)).astype(np.float16)

open("A.arr", "wb").write(A.tobytes(order='C'))
open("B.arr", "wb").write(B.tobytes(order='C'))
open("C.arr", "wb").write(C.tobytes(order='C'))
