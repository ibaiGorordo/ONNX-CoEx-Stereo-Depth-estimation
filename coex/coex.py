import time
import cv2
import numpy as np
import onnx
import onnxruntime

from coex.utils_coex import *

drivingStereo_config = CameraConfig(0.546, 1000)

class CoEx():

	def __init__(self, model_path, camera_config=drivingStereo_config):

		self.fps = 0
		self.timeLastPrediction = time.time()
		self.frameCounter = 0

		self.camera_config = camera_config

		# Initialize model
		self.model = self.initialize_model(model_path)

	def __call__(self, left_img, right_img):

		return self.estimate_disparity(left_img, right_img)

	def initialize_model(self, model_path):

		self.session = onnxruntime.InferenceSession(model_path)

		# Get model info
		self.getModel_input_details()
		self.getModel_output_details()

	def estimate_disparity(self, left_img, right_img):

		# Update fps calculator
		self.updateFps()

		left_input_tensor = self.prepare_input(left_img)
		right_input_tensor = self.prepare_input(right_img)

		self.disparity_map = self.inference(left_input_tensor, right_input_tensor)

		return self.disparity_map

	def get_depth(self):
		return self.camera_config.f*self.camera_config.baseline/self.disparity_map

	def prepare_input(self, image):

		img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		self.img_height, self.img_width, self.img_channels = img.shape

		# Input values should be from -1 to 1 with a size of 288 x 800 pixels
		img_input = cv2.resize(img, (self.input_width,self.input_height)).astype(np.float32)
		
		# Scale input pixel values to -1 to 1
		mean=[0.485, 0.456, 0.406]
		std=[0.229, 0.224, 0.225]
		
		img_input = ((img_input/ 255.0 - mean) / std)
		img_input = img_input.transpose(2, 0, 1)
		img_input = img_input[np.newaxis,:,:,:]        

		return img_input.astype(np.float32)


	def inference(self, left_input_tensor, right_input_tensor):

		left_input_name = self.session.get_inputs()[0].name
		right_input_name = self.session.get_inputs()[1].name
		output_name = self.session.get_outputs()[0].name

		output = self.session.run([output_name], {left_input_name: left_input_tensor, right_input_name: right_input_tensor})[0]

		return np.squeeze(output)

	def updateFps(self):
		updateRate = 1
		self.frameCounter += 1

		# Every updateRate frames calculate the fps based on the ellapsed time
		if self.frameCounter == updateRate:
			timeNow = time.time()
			ellapsedTime = timeNow - self.timeLastPrediction

			self.fps = int(updateRate/ellapsedTime)
			self.frameCounter = 0
			self.timeLastPrediction = timeNow

	def getModel_input_details(self):

		self.left_input_shape = self.session.get_inputs()[0].shape
		self.channes = self.left_input_shape[2]
		self.input_height = self.left_input_shape[2]
		self.input_width = self.left_input_shape[3]

		self.right_input_shape = self.session.get_inputs()[1].shape

	def getModel_output_details(self):

		self.output_shape = self.session.get_outputs()[0].shape
	






