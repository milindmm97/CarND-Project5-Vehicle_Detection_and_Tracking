# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 09:58:45 2018

@author: Milind
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 21:40:48 2018

@author: Milind
"""

import cv2
import matplotlib.pyplot as plt

# Read single frame avi
cap = cv2.VideoCapture('project_video.mp4')
rval, frame = cap.read()

# Attempt to display using cv2 (doesn't work)
cv2.namedWindow("Input")
cv2.imshow("Input", frame)