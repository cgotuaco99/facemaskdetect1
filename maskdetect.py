# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os

count = 0 # for checking every x seconds

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo",default="./yolo",
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

labelsPath = os.path.sep.join([args["yolo"], "mask.names"])
LABELS = open(labelsPath).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
weightsPath = os.path.sep.join([args["yolo"], "yolov3-obj_last.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3-obj_copy.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

vid = cv2.VideoCapture(0)

prev_frame_time = 0
new_frame_time = 0

while(True):
	ret, frame = vid.read()
	gray = frame
	#gray = cv2.resize(gray, (500,300))
	
	count = count + 1
	print("count: ", count)
 

	new_frame_time = time.time()
	fps = 1/(new_frame_time-prev_frame_time) 
	prev_frame_time = new_frame_time 
	# converting the fps into integer 
	
	fps = int(fps) 

	font = cv2.FONT_HERSHEY_SIMPLEX 
	
	# converting the fps to string so that we can display it on frame 
    # by using putText function 
	fps = str(fps) 
    # puting the FPS count on the frame 
	fpstxt = "FPS: {}".format(fps)
	cv2.putText(gray, fpstxt, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
  
    # displaying the frame with fps 
	cv2.imshow('frame', gray) 
  
    # converting the fps to string so that we can display it on frame 
    # by using putText function 
	fps = str(fps) 

	(H, W) = frame.shape[:2]
	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
 
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	# show timing information on YOLO
	# print("[INFO] YOLO took {:.6f} seconds".format(end - start))
	boxes = []
	confidences = []
	classIDs = []
	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > args["confidence"]:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],args["threshold"])   
	if len(idxs) > 0:
	# loop over the indexes we are keeping
		for i in idxs.flatten():

			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# draw a bounding box rectangle and label on the image
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, color, 2)
			#print("LABEL:", LABELS[classIDs[i]])
			if (LABELS[classIDs[i]] == "with mask"):
				if (count % 10 == 0):
					print("good job!!")
			elif (LABELS[classIDs[i]] == "without mask"):
				if (count % 20 == 0):
					print("hey! put a mask on!")		
					cv2.putText(gray, "hey! put a mask on!", (7, 200), font, 3, (100, 255, 0), 3, cv2.LINE_AA)	
			

		
		

	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
vid.release()
cv2.destroyAllWindows()
