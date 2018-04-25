# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 07:18:33 2018

@author: Milind
"""

import numpy as np
import model
from scipy.ndimage.measurements import label
import cv2
import os.path

confidenceThrd=.7
diagKernel = [[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]]
veHiDepth = 30
vehicleBoxesHistory = []
gdroupThr=10
groupDiff=.1

count=0

crop=(400, 660)
imgInputShape=(720, 1280, 3)
bottomClip = imgInputShape[0] - crop[1]
inH = imgInputShape[0] - crop[0] - bottomClip
inW = imgInputShape[1]
inCh = imgInputShape[2]

hotPoints = []
detectionPointSize = 64

cnnModel,cnnModelName=model.poolerPico(inputShape=(inH, inW, inCh))
cnnModel.load_weights('{}.h5'.format(cnnModelName))

cnnModel.summary()
cnnModelName



def drawBoxes(img, bBoxes, color=(0, 255, 0), thickness=4):
    """
    Universal bounding box painter, regardless of bBoxes format 
    :param img: image of interest
    :param bBoxes: list of bounding boxes.
    :param color: 
    :param thickness: 
    :return: 
    """
    for bBox in bBoxes:

        bBox = np.array(bBox)
        bBox = bBox.reshape(bBox.size)

        cv2.rectangle(img=img, pt1=(bBox[0], bBox[1]), pt2=(bBox[2], bBox[3]),
                      color=color, thickness=thickness)
        
cap = cv2.VideoCapture('project_video.mp4')
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
      
      # Display the resulting frame
      #cv2.imshow('Frame',frame)
      count = count+1
      print(count)
      roi= frame[crop[0]:crop[1],:]
      roiW, roiH = roi.shape[1], roi.shape[0]
      
      roi = np.expand_dims(roi, axis=0)
      detectionMap = cnnModel.predict(roi)
      predictionMapH, predictionMapW = detectionMap.shape[1], detectionMap.shape[2]
      ratioH, ratioW = roiH / predictionMapH, roiW / predictionMapW
      detectionMap = detectionMap.reshape(detectionMap.shape[1], detectionMap.shape[2])
      detectionMap = detectionMap > confidenceThrd
      labels = label(detectionMap, structure= diagKernel)
      
      for vehicleID in range(labels[1]):
          nz = (labels[0] == vehicleID + 1).nonzero()
          nzY = np.array(nz[0])
          nzX = np.array(nz[1])
          xMin = np.min(nzX) - 32
          xMax = np.max(nzX) + 32
          
          yMin = np.min(nzY)
          yMax = np.max(nzY) + 64
          
          spanX = xMax - xMin
          spanY = yMax - yMin
          
          for x, y in zip(nzX, nzY):
              
              offsetX = (x - xMin) / spanX * detectionPointSize
              offsetY = (y - yMin) / spanY * detectionPointSize
              
              topLeftX = int(round(x * ratioW - offsetX, 0))
              topLeftY = int(round(y * ratioH - offsetY, 0))
              bottomLeftX = topLeftX + detectionPointSize
              bottomLeftY = topLeftY + detectionPointSize
              topLeft = (topLeftX, crop[0] + topLeftY)
              bottomRight = (bottomLeftX, crop[0] + bottomLeftY)
              hotPoints.append((topLeft, bottomRight))
              
      src= frame
      sampleMask = np.zeros_like(src[:, :, 0]).astype(np.float)
      bBoxes=hotPoints
      mask=sampleMask
         
         
      for box in bBoxes:
          
          topY = box[0][1]
          bottomY = box[1][1]
          leftX = box[0][0]
          rightX = box[1][0]
          
          mask[topY:bottomY, leftX:rightX] += 1
          mask = np.clip(mask, 0, 255)
              
        
             
        
      heatMap = mask 
      currentFrameBoxes = label(heatMap, structure=diagKernel)
      cmap=cv2.COLORMAP_JET
      heatMapInt = cv2.equalizeHist(heatMap.astype(np.uint8))
      heatColor = cv2.applyColorMap(heatMapInt, cmap)
      heatColor = cv2.cvtColor(heatColor, code=cv2.COLOR_BGR2RGB)
      
      
      for i in range(currentFrameBoxes[1]):
          nz = (currentFrameBoxes[0] == i + 1).nonzero()
          nzY = np.array(nz[0])
          nzX = np.array(nz[1])
          tlX = np.min(nzX)
          tlY = np.min(nzY)
          brX = np.max(nzX)
          brY = np.max(nzY)
          
          
          vehicleBoxesHistory.append([tlX, tlY, brX, brY])
          vehicleBoxesHistory = vehicleBoxesHistory[-veHiDepth:]

        
      boxes, _ = cv2.groupRectangles(rectList=np.array(vehicleBoxesHistory).tolist(),
                                           groupThreshold=gdroupThr, eps=groupDiff)
      img= frame
      drawBoxes(img,boxes)
      cv2.imshow('box',img)
        
        
    
 
    # Press Q on keyboard to  exit
  if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()

print('done')
'''
frame_car1 = cv2.imread('frame1000.jpg',3)
cv2.imshow('image',frame_car1)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''





# Going 4-D


# Single-Feature top convolutional layer, which represents a
# miniaturized (25x153) version of the ROI with the vehicle's probability at each point






        # Considering obtained labels as vehicles.



'''
cv2.imshow('heat',heatMap)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''



'''
cv2.imshow('heat',heatColor)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


#sizeof(boxes)
'''
boxes





cv2.waitKey(0)
cv2.destroyAllWindows()

'''
















