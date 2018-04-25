# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 15:24:23 2018

@author: Milind
"""
import cv2
frame_car1 = cv2.imread('frame1000.jpg',3)
cv2.imshow('image',frame_car1)
cv2.waitKey(0)
cv2.destroyAllWindows()