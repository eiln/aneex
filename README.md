

# ANE Examples



### Object Detection

[yolov5](https://github.com/ultralytics/yolov5) |
[notebook](notebooks/yolov5.ipynb)

<img src="assets/yolov5.jpg" width=60% height=60%>



### Super Resolution

[srgan](https://github.com/john-rocky/CoreML-Models#srgan) |
[notebook](notebooks/srgan.ipynb)

512 x 512             |  2048 x 2048
:-------------------------:|:-------------------------:
![](assets/srgan-512.jpg)  |  ![](assets/srgan-2048.jpg)



### Semantic Segmentation

[fcn](https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.fcn_resnet50.html) |
[notebook](notebooks/fcn.ipynb)

<img src="assets/fcn.jpg" width=70% height=70%>




# Compilation


the `compile` dir should look like:

	compile
	├── compile.sh
	├── hwx
	│   ├── fcn.hwx
	│   ├── srgan.hwx
	│   └── yolov5.hwx
	└── Makefile

from which

	cd compile
	make


To obtain hwx sources
either:


	bash download.sh  # curl from my dropbox


or:


[sources.md](sources.md)
