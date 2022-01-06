# import the necessary packages
from detect import social_distancing_config as config
from detect.detection import detect_people

from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
import csv
from csv import DictWriter
import time

# 9FPS

Timecount = 0
maxpeople = []
mindistance = []
countround = 0
# path = '1.mp4'
path = 'rtsp://data_analytic:TcAnTaRa9721xx#@10.153.60.87'
# path = 'rtsp://data_analytic:TcAnTaRa9721881@10.158.8.19'
# path = 'rtsp://data_analytic:TcAnTaRa9721&&!@10.158.14.76:554/ch1-s1'

def csvWriter():
	with open('data2.csv', 'w', newline='') as file:
    		fieldnames = ['Time', 'Max' , 'Min']
    		writer = csv.DictWriter(file, fieldnames=fieldnames)

    		# writer.writeheader()
			#
    		# writer.writerow({'Time': time , 'Max':maxpeople , 'Min':mindistance})


def maxValue(value):
	max_value = max(value)
	return max_value

def minValue(value):
	min_value = min(value)
	return min_value

def openValue(value):
	open_value = value[0]
	return open_value

def closeValue(value):
	close_Value = value[-1]
	return close_Value

def append_dict_as_row(file_name, dict_of_elem, field_names):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        dict_writer = DictWriter(write_obj, fieldnames=field_names)
        # Add dictionary as wor in the csv
        dict_writer.writerow(dict_of_elem)

	# field_names = ['Time', 'Max' , 'Min']
	# row_dict = {'Time': time , 'Max':maxpeople , 'Min':mindistance}
	# # Append a dict as a row in csv file
	# append_dict_as_row('data.csv', row_dict, field_names)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov4-tiny-crowd.weights"])
# weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov4-p6.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov4-tiny-crowd.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
# check if we are going to use GPU
if config.USE_GPU:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
	# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
# vs = cv2.VideoCapture(args["input"] if args["input"] else path)
vs = cv2.VideoCapture(path,cv2.CAP_FFMPEG)

writer = None
# loop over the frames from the video stream

x = 1  # displays the frame rate every 1 second
counter = 0
# start_time = time.time()

# while True:
while True:

	value = []
	start_time = time.time()
	# read the next frame from the file
	for countround in range(1):
		# time1 = time.time()

		# fps = vs.get(cv2.CAP_PROP_FPS)
		# print(fps)
		(grabbed, frame) = vs.read()

		# frame = cv2.resize(frame1, (int(1280 / 2), int(720 / 2)))
		# if the frame was not grabbed, then we have reached the end
		# of the stream
		if not grabbed:
			break
		# resize the frame and then detect people (and only people) in it
		# frame = imutils.resize(frame, width=600)
		# results = detect_people(frame, net, ln,
		# 	personIdx=LABELS.index("person"))
		results = detect_people(frame, net, ln,
			personIdx=LABELS.index("person"))




		# initialize the set of indexes that violate the minimum social
		# distance
		violate = set()

		# ensure there are *at least* two people detections (required in
		# order to compute our pairwise distance maps)
		# for countround in range(10):
		if len(results) >= 2:
			# extract all centroids from the results and compute the
			# Euclidean distances between all pairs of the centroids
			centroids = np.array([r[2] for r in results])
			D = dist.cdist(centroids, centroids, metric="euclidean")
			# loop over the upper triangular of the distance matrix
			for i in range(0, D.shape[0]):
				for j in range(i + 1, D.shape[1]):
					# check to see if the distance between any two
					# centroid pairs is less than the configured number
					# of pixels
					if D[i, j] < config.MIN_DISTANCE:
						# update our violation set with the indexes of
						# the centroid pairs
						violate.add(i)
						violate.add(j)
		print("FPS (Jetson nano GPU) : ", 1/(time.time() - start_time))
		# counter += 1
		# if (time.time() - start_time) > x:
		# 	print("FPS (Jetson AGX Xavier with GPU): ", (time.time() - start_time))
		# 	counter = 0
		# 	start_time = time.time()


			# loop over the results
		for (i, (prob, bbox, centroid)) in enumerate(results):
			# extract the bounding box and centroid coordinates, then
			# initialize the color of the annotation
			(startX, startY, endX, endY) = bbox
			(cX, cY) = centroid
			color = (0, 255, 0)
			# if the index pair exists within the violation set, then
			# update the color
			if i in violate:
				color = (0, 0, 255)
			# draw (1) a bounding box around the person and (2) the
			# centroid coordinates of the person,
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			cv2.circle(frame, (cX, cY), 5, color, 1)

		# print(time2 - time1)
		# print(len(results) , len(violate))
		# draw the total number of social distancing violations on the
		# output frame
		# cv2.putText(frame, f'FPS: {int(net.GetNetworkFPS())}', (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
		text = "Social Distancing Violations: {}".format(len(violate))
		cv2.putText(frame, text, (10, frame.shape[0] - 25),
					cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
		# maxpeople = (len(results))
		# mindistance = (len(violate))
		# csvWriter()

		value.append(len(violate))
		# print(value)


		# check to see if the output frame should be displayed to our
		# screen
		if args["display"] > 0:
			# show the output frame
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF
			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break
		# if an output video file path has been supplied and the video
		# writer has not been initialized, do so now
		if args["output"] != "" and writer is None:
			# initialize our video writer
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 25,
								 (frame.shape[1], frame.shape[0]), True)
		# if the video writer is not None, write the frame to the output
		# video file
		if writer is not None:
			writer.write(frame)

		# counter += 1
		# if (time.time() - start_time) > x:
		# 	print("FPS (Jetson AGX Xavier with GPU): ", counter / (time.time() - start_time))
		# 	counter = 0
		# 	start_time = time.time()


		# time2 = time.time()
		# print("FPS: ", 1.0 / (time2 - time1))

	# Max = maxValue(value)
	# Min = minValue(value)
	# Open = openValue(value)
	# Close = closeValue(value)
	#
	# Timecount += 1
	# #
	# # print(time, maxpeople, mindistance)
	# print(Timecount,Open,Max,Min,Close)
	# field_names = ['Time','Open','High','Low','Close']
	# # #
	# row_dict = {'Time': Timecount,'Open': Open,'High': Max,'Low': Min,'Close': Close }
	# # # # Append a dict as a row in csv file
	# append_dict_as_row('data2.csv', row_dict, field_names)





