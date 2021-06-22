#openCV lane detection project

import cv2
import numpy as np
import matplotlib.pyplot as plt

#--------------------------------------------------------
#defining canny function
#--------------------------------------------------------

def canny(image):
	canny_image = cv2.Canny(blur_image,50,150) 
	cv2.imshow('result4',canny_image)
	cv2.waitKey(0)
	return canny_image

#--------------------------------------------------------
#function to define region of interest
#---------------------------------------------------------

def region_of_interest(image):
	#mask entire image and extract only region of interest as a polygon
	height = image.shape[0]
	polygons = np.array([
		[(200,height),(1000,height),(550,250)]])
	mask = np.zeros_like(image)
	cv2.fillPoly(mask, polygons, 255) #masked image with polygon created
	#using bitwise 'and' operator on masked image and edge image to find region of interest
	masked_image = cv2.bitwise_and(image,mask)
	return masked_image

#--------------------------------------------------------
#function to display lanes in the original image
#---------------------------------------------------------

def display_lane(image,lines_detect):
	plot_line_image = np.zeros_like(image) #create a black image with same dimension as original image
	#plot lines detected onto the black image
	if lines_detect is not None:
		for line in lines_detect:
			print(line) #printing 2D array of line coordintes
			x1, y1, x2, y2 = line.reshape(4) #reshape the 2D array to 1D array
			cv2.line(plot_line_image, (x1, y1), (x2, y2), (255,0,0), 10) #draw a line segment on the coordinates
	return plot_line_image

#--------------------------------------------------------
#fucntion to optimize detected lanes on the image
#---------------------------------------------------------
def average_slope_intercept(image,lines_detect):
	left_fit = []
	right_fit = []
	for line in lines_detect:
		x1,y1,x2,y2 = line.reshape(4)
		parameters = np.polyfit((x1,x2),(y1,y2),1)
		print(parameters)
		slope = parameters[0]
		intercept = parameters[1]
		if slope > 0:
			left_fit.append((slope,intercept))
		else:
			right_fit.append((slope,intercept))
	print(left_fit)
	print(right_fit)
	#average out all the slopes and intercepts
	left_fit_average = np.average(left_fit,axis=0)
	right_fit_average = np.average(right_fit,axis=0)

	left_fit_line = determine_coordinates(image,left_fit_average)
	right_fit_line = determine_coordinates(image,right_fit_average)
	return np.array([left_fit_line,right_fit_line])

#---------------------------------------------------------
#define a function to plot the lines with averaged out slope and intercept
#--------------------------------------------------------

def determine_coordinates (image,line_parameters):
	slope, intercept = line_parameters
	y1 = image.shape[0]
	y2 = int(3/5*y1)
	x1 = int((y1-intercept)/slope)
	x2 = int((y2-intercept)/slope)
	return np.array([x1,y1,x2,y2])

#--------------------------------------------------------			
#reading an road image file
#---------------------------------------------------------

image = cv2.imread('test_image.jpg')
#display original image
cv2.imshow('result1',image)
cv2.waitKey(0)

#-------------------------------------------------------
#edge detection algorithm
#-------------------------------------------------------

#determine gradients with sharp change in intensity - detects edges
#converting image to gray-scale image

lane_image = np.copy(image)
gray_scale = cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY)
#display grayscale image
cv2.imshow('result2',gray_scale)
cv2.waitKey(0)

#filtering image noise and smoothened image
blur_image = cv2.GaussianBlur(gray_scale,(5,5),0)
#display filtered image
cv2.imshow('result3',blur_image)
cv2.waitKey(0)

#-------------------------------------------------------------------
#calling canny function to determine gradient for edge detection
#--------------------------------------------------------------------

edge_image = canny(lane_image)

#------------------------------------------------------------------
#speciying region of interest in the edge_image
#------------------------------------------------------------------

#plot axis on the image
plt.imshow(edge_image)
plt.show()

#function call region of interest
#final cropped image with only the region of interest

cropped_image = region_of_interest(edge_image)
cv2.imshow('result5',cropped_image)

#--------------------------------------------------------------------
#detect straight lines in the cropped image using hough transform
#--------------------------------------------------------------------

lines_detect = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

#function call to display lane in the black image
plot_line_image = display_lane(lane_image, lines_detect)
cv2.imshow('result6', plot_line_image)
cv2.waitKey(0)

#blending the plot line image with the original image to shoe detected lanes

blend_with_original_image = cv2.addWeighted(lane_image,0.8,plot_line_image,1,1)
cv2.imshow('result7', blend_with_original_image)
cv2.waitKey(0)

#optmize detected lanes on the image

optimize_line = average_slope_intercept(lane_image,lines_detect)
optimize_lane_image = display_lane(lane_image, optimize_line)
lane_detection_image = cv2.addWeighted(lane_image,0.8,optimize_lane_image,1,1)
cv2.imshow('result8', lane_detection_image)
cv2.waitKey(0)

#------------------------------------------------------------------
#lane detection in a video
#------------------------------------------------------------------

video_data = cv2.VideoCapture("test2.mp4")
while (video_data.isOpened()):
	_, frame = video_data.read()
	edge_image = canny(frame)
	cropped_image = region_of_interest(edge_image)
	lines_detect = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
	plot_line_image = display_lane(frame, lines_detect)
	optimize_line = average_slope_intercept(frame,lines_detect)
	optimize_lane_image = display_lane(frame, optimize_line)
	lane_detection_video = cv2.addWeighted(frame,0.8,optimize_lane_image,1,1)
	cv2.imshow('result9', lane_detection_video)
	cv2.waitKey(0)
