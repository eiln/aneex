
# Sources


#### yolov5

	git clone https://github.com/ultralytics/yolov5.git
	cd yolov5
	python3 export.py --weights yolov5s.pt --include coreml --imgsz 576


##### srgan

	# download
	https://drive.google.com/file/d/1-076W2o0wCtoODptikX1eOnlFBx2s3qK/view?usp=sharing


##### fcn

	# Adopted from the pytorch site
	import torch
	import torch.nn as nn

	# Download an example image from the pytorch website
	import urllib
	url, filename = ("https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png")
	try: urllib.URLopener().retrieve(url, filename)
	except: urllib.request.urlretrieve(url, filename)

	from PIL import Image
	import cv2
	import numpy as np
	img = cv2.imread(filename)[:,:,::-1]
	input_image = Image.fromarray(img)


	from torchvision import transforms
	preprocess = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	input_tensor = preprocess(input_image)
	input_batch = input_tensor.unsqueeze(0)


	class WrappedFCN(nn.Module):
	    def __init__(self):
	        super(WrappedFCN, self).__init__()
	        self.model = torch.hub.load(
	            'pytorch/vision:v0.11.0',
	            'fcn_resnet50', pretrained=True,
	        ).eval()
	    def forward(self, x):
	        res = self.model(x)
	        x = res["out"]
	        return x
	traceable_model = WrappedFCN().eval()
	trace = torch.jit.trace(traceable_model, input_batch)


	import coremltools as ct
	mlmodel = ct.convert(trace, 
		inputs=[ct.TensorType(name='x', shape=input_batch.shape)])
	mlmodel.save("fcn.mlmodel")

