# SPDX-License-Identifier: MIT
# Copyright 2022 Eileen Yoon <eyn@gmx.com>

from ane import ANE_MODEL

import numpy as np
import cv2

from torchvision import transforms
from PIL import Image


class FCN(ANE_MODEL):
	def __init__(self, ane):
		super(FCN, self).__init__(ane)
		self.input_count, self.output_count = 1, 1
		self.output_size = [0x100000]
		self.input_nchw = [(1, 3, 1026, 1282, 0x291480, 0xa40)]
		self.output_nchw = [(1, 21, 129, 161, 0xc180, 0x180)]
		self.initf = self.ane.lib.pyane_init_fcn
		self.register()

	def preprocess(self, img):
		# (any, any, 3) cv2 RGB -> (1, 3, 1282, 1026) inarr
		resized = cv2.resize(img, (1282, 1026), interpolation=cv2.INTER_AREA)
		trans = transforms.Compose([
		    transforms.ToTensor(),
		    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
		batch = trans(Image.fromarray(resized)).unsqueeze(0).numpy()
		return batch.astype(np.float16)

	def postprocess(self, outarrs):
		# (1, 3, 2048, 2048) outarr -> (2048, 2048, 3) cv2 RGB
		return outarrs[0]
