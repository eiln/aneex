# SPDX-License-Identifier: MIT
# Copyright 2022 Eileen Yoon <eyn@gmx.com>

from ane import ANE_MODEL

import numpy as np
import cv2

def normalize(data): # https://stackoverflow.com/a/55141403/20891128
    return (data - np.min(data)) / (np.max(data) - np.min(data))


class YOLOV5(ANE_MODEL):
	def __init__(self, ane):
		super(YOLOV5, self).__init__(ane)
		self.input_count, self.output_count = 1, 3
		self.output_size = [0x2dc000, 0xb8000, 0x30000]
		self.input_nchw = [(1, 3, 576, 576, 0xa2000, 0x480)]
		self.output_nchw = [(1, 1, 15552, 85, 0x2d9000, 0xc0),
				    (1, 1, 3888, 85, 0xb6400, 0xc0),
				    (1, 1, 972, 85, 0x2d900, 0xc0)]
		self.initf = self.ane.lib.pyane_init_yolov5
		self.register()

	def preprocess(self, img):
		# (any, any, 3) cv2 RGB -> (1, 3, 576, 576) inarr
		resized = cv2.resize(img, (576, 576), interpolation=cv2.INTER_AREA)
		transposed = np.expand_dims(resized.swapaxes(0, -1).swapaxes(1, 2), 0)
		normed = normalize(transposed).astype(np.float16)
		return normed

	def postprocess(self, outarrs):
		arr = np.vstack([outarrs[0].squeeze(), outarrs[1].squeeze(), outarrs[2].squeeze()])
		return arr
