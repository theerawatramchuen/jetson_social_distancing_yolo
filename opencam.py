# import cv2
# import os
#
# RTSP_URL = 'rtsp://data_analytic:TcAnTaRa9721&&!@10.158.14.76:554/Streaming/Channels/401'

import cv2
import numpy as np
import os
import time

# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
# vcap = cv2.VideoCapture("rtsp://data_analytic:TcAnTaRa9721&&!@10.158.14.76:554/ch1-s1", cv2.CAP_FFMPEG)
# vcap = cv2.VideoCapture("rtsp://data_analytic:TcAnTaRa9721881@10.158.8.19", cv2.CAP_FFMPEG)
vcap = cv2.VideoCapture("rtsp://data_analytic:TcAnTaRa9721xx#@10.153.60.87", cv2.CAP_FFMPEG)
# vcap = cv2.VideoCapture("http:://data_analytic:TcAnTaRa9721&&!@/10.158.14.76:8000/mjpeg.cgi?user=data_analytic&password=TcAnTaRa9721&&!&channel=0&.mjpg", cv2.CAP_FFMPEG)
while(1):
    # w = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # h = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    start_time = time.time()
    ret, frame = vcap.read()
    #h, w = frame.shape[:2]


    #cv2.namedWindow('VIDEO', cv2.WINDOW_AUTOSIZE)
    # if ret == False:
    #     print("Frame is empty")
    #     break
    # else:
    #     cv2.imshow('VIDEO', frame)
    #     cv2.waitKey(1)
    cv2.imshow('VIDEO', frame)
    print("FPS (Jetson nano GPU) : ", 1 / (time.time() - start_time))
    #print(w,h)
    cv2.waitKey(1)