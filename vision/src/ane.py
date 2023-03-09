# SPDX-License-Identifier: MIT
# Copyright 2022 Eileen Yoon <eyn@gmx.com>

import ctypes
from ctypes import c_void_p, c_char_p, pointer, addressof

import atexit
import numpy as np


def round_up(x, y):
	return ((x + (y - 1)) & (-y))


class ANE:
	def __init__(self, path="/home/eileen/aneex/vision/compile/pyane.so"):
		self.lib = ctypes.cdll.LoadLibrary(path)
		self.lib.pyane_free.argtypes = [c_void_p]
		self.lib.pyane_exec.argtypes = [c_void_p, c_void_p, c_void_p]
		self.handles = {}
		atexit.register(self.cleanup)

	def cleanup(self):
		for handle in self.handles:
			self.lib.pyane_free(handle)

	def init(self, model):
		handle = model.initf()
		if (handle == None): raise RuntimeError("uh oh")
		self.handles[handle] = 0
		model.handle = handle

class ANE_MODEL:
	def __init__(self, ane):
		self.ane = ane
		self.initf = None
		self.handle = None

	def register(self):
		assert(self.input_count and self.output_count)
		assert(self.input_count == len(self.input_nchw))
		assert(self.output_count == len(self.output_nchw) == len(self.output_size))
		self.outputs = [None] * self.output_count
		for n in range(self.output_count):
			self.outputs[n] = ctypes.create_string_buffer(self.output_size[n])
		self.initf.restype = c_void_p
		self.ane.init(self)

	def predict(self, inputs):
		inputs_p = (c_char_p * len(inputs))(*inputs)
		outputs = [c_char_p(addressof(output)) for output in self.outputs]
		outputs_p = (c_char_p * len(outputs))(*outputs)
		err = self.ane.lib.pyane_exec(self.handle, inputs_p, outputs_p)
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

	def tile(self, arrs):
		return [self.nchw_tile(arrs[n], self.input_nchw[n]) for n in range(self.input_count)]

	def untile(self, outputs):
		return [self.nchw_untile(outputs[n], self.output_nchw[n]) for n in range(self.output_count)]

