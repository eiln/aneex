# SPDX-License-Identifier: MIT
# Copyright 2022 Eileen Yoon <eyn@gmx.com>

from ane import ANE_MODEL

import numpy as np
import cv2

def rescale(a, low, high):
    return np.interp(a, (a.min(), a.max()), (low, high))


class SRGAN(ANE_MODEL):
	def __init__(self, ane):
		super(SRGAN, self).__init__(ane)
		self.input_count, self.output_count = 1, 1
		self.output_size = [0x1800000]
		self.input_nchw = [(1, 3, 512, 512, 0x80000, 0x400)]
		self.output_nchw = [(1, 3, 2048, 2048, 0x800000, 0x1000)]
		self.initf = self.ane.lib.pyane_init_srgan
		self.register()

	def preprocess(self, img):
		# (any, any, 3) cv2 RGB -> (1, 3, 512, 512) inarr
		resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
		transposed = np.expand_dims(resized.swapaxes(0, -1).swapaxes(1, -1), 0)
		normed = rescale(transposed, -1, +1).astype(np.float16)
		return normed

	def postprocess(self, outarrs):
		# (1, 3, 2048, 2048) outarr -> (2048, 2048, 3) cv2 RGB
		reshaped = np.swapaxes(outarrs[0].squeeze(), 0, -1).swapaxes(0, 1)
		clipped = np.round(reshaped).clip(0, 255).astype(np.uint8)
		return clipped
