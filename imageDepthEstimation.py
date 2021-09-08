import cv2
from coex import CoEx, draw_disparity, draw_depth, CameraConfig
import numpy as np
from imread_from_url import imread_from_url

if __name__ == '__main__':
		
	model_path = "models/coex_480x640.onnx"

	# Initialize model
	coex_stereo_depth = CoEx(model_path)

	# Load images
	left_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im2.png")
	right_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im6.png")

	# Estimate the depth
	disparity_map = coex_stereo_depth(left_img, right_img)

	color_disparity = draw_disparity(disparity_map)
	color_disparity = cv2.resize(color_disparity, (left_img.shape[1],left_img.shape[0]))

	cobined_image = np.hstack((left_img, right_img, color_disparity))

	cv2.imwrite("out.jpg", cobined_image)

	cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)	
	cv2.imshow("Estimated disparity", cobined_image)
	cv2.waitKey(0)

	cv2.destroyAllWindows()
