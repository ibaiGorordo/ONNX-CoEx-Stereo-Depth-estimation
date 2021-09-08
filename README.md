# ONNX-CoEx-Stereo-Depth-estimation
Python scripts form performing stereo depth estimation using the CoEx model in ONNX.

![CoEx stereo depth estimation ONNX](https://github.com/ibaiGorordo/ONNX-CoEx-Stereo-Depth-estimation/blob/main/doc/img/out.jpg)
*Stereo depth estimation on the cones images from the Middlebury dataset (https://vision.middlebury.edu/stereo/data/scenes2003/)*

# Requirements

 * **OpenCV**, **imread-from-url**, **onnx** and **onnxruntime**. Also, **pafy** and **youtube-dl** are required for youtube video inference. 
 
# Installation
```
pip install -r requirements.txt
pip install pafy youtube-dl
```

# ONNX model
The original models were converted to different formats (including .onnx) by [PINTO0309](https://github.com/PINTO0309), download the models from [his repository](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/135_CoEx) and save them into the **[models](https://github.com/ibaiGorordo/ONNX-CoEx-Stereo-Depth-estimation/tree/main/models)** folder. 

# Original Pytorch model
The Pytorch pretrained model was taken from the [original repository](https://github.com/antabangun/coex).
 
# Examples

 * **Image inference**:
 
 ```
 python imageDepthEstimation.py 
 ```
 
  * **Video inference**:
 
 ```
 python videoDepthEstimation.py
 ```
 
 * **DrivingStereo dataset inference**:
 
 ```
 python drivingStereoTest.py
 ```
 
# [Inference video Example](https://youtu.be/q1IfuHp0HR4) 
 ![CoEx stereo depth estimation ONNX](https://github.com/ibaiGorordo/ONNX-CoEx-Stereo-Depth-estimation/blob/main/doc/img/onnxCoExDepthEstimation.gif)

# References:
* CoEx model: https://github.com/antabangun/coex
* PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
* PINTO0309's model conversion tool: https://github.com/PINTO0309/openvino2tensorflow
* DrivingStereo dataset: https://drivingstereo-dataset.github.io/
* Original paper: 
*Correlate-and-Excite: Real-Time Stereo Matching via Guided Cost Volume Excitation
Authors: Antyanta Bangunharcana, Jae Won Cho, Seokju Lee, In So Kweon, Kyung-Soo Kim, Soohyun Kim
IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2021*
 


