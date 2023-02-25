# SPDX-License-Identifier: MIT
# Copyright 2022 Eileen Yoon <eyn@gmx.com>

import ctypes
from ctypes import c_void_p, pointer

import atexit
import numpy as np


def round_up(x, y):
	return ((x + (y - 1)) & (-y))


class ANE:
	def __init__(self, path="/home/eileen/aneex/compile/pyane.so"):
		self.lib = ctypes.cdll.LoadLibrary(path)
		print(self.lib)
		self.handles = {}
		atexit.register(self.cleanup)

	def cleanup(self):
		for handle in self.handles:
			self.handles[handle](handle)

	def init(self, model):
		handle = model.initf()
		if (handle == None): raise RuntimeError("uh oh")
		self.handles[handle] = model.freef
		model.handle = handle
		return handle


class ANE_MODEL:
	def __init__(self, ane):
		self.ane = ane
		self.handle = None

	def register(self):
		assert(self.input_count and self.output_count)
		if (self.input_count > 1): raise ValueError("todo")
		assert(self.input_count == len(self.input_nchw))
		assert(self.output_count == len(self.output_nchw) == len(self.output_size))
		self.outputs = [None] * self.output_count
		for n in range(self.output_count):
			self.outputs[n] = ctypes.create_string_buffer(self.output_size[n])
		self.initf.restype = c_void_p
		self.freef.argtypes = [c_void_p]
		self.execf.argtypes = [c_void_p] * (1 + self.input_count + self.output_count)
		self.ane.init(self)

	def predict(self, input):
		err = self.execf(self.handle, input, *[pointer(output) for output in self.outputs])
		return self.outputs

	def nchw_tile(self, arr, nchw):
		assert((arr.ndim == 4) and (arr.dtype == np.float16) and (arr.shape == nchw[:4]))
		N, C, H, W, pS, rS = nchw
		new_N, new_C, new_H, new_W = N, C, pS//rS, rS//2
		tarr = np.zeros((new_N, new_C, new_H, new_W), dtype=np.float16)
		tarr[:N, :C, :H, :W] = arr
		tile = tarr.tobytes(order='C')
		tile += b'\0' * (round_up(len(tile), 0x4000) - len(tile))
		return tile

	def nchw_untile(self, tile, nchw):
		N, C, H, W, pS, rS = nchw
		new_N, new_C, new_H, new_W = N, C, pS//rS, rS//2
		arr = np.frombuffer(tile, dtype=np.float16)[:new_N*new_C*new_H*new_W]
		arr = arr.reshape((new_N, new_C, new_H, new_W))[:N, :C, :H, :W]
		return arr

	def tile(self, arr, n=0):
		return self.nchw_tile(arr, self.input_nchw[n])

	def untile(self, outputs):
		return [self.nchw_untile(outputs[n], self.output_nchw[n]) for n in range(self.output_count)]

