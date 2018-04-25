
import numpy as np
import model
from scipy.ndimage.measurements import label
import cv2
#import pickle

#import os.path



font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
interpolation=cv2.INTER_AREA
origin=(20, 20)

confidenceThrd=.5
diagKernel = [[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]]
veHiDepth = 45
vehicleBoxesHistory = []
gdroupThr= 10
groupDiff=.1

count=0

#generate cropped region 
crop=(400, 660)
imgInputShape=(720, 1280, 3)
bottomClip = imgInputShape[0] - crop[1]
inH = imgInputShape[0] - crop[0] - bottomClip
inW = imgInputShape[1]
inCh = imgInputShape[2]
########################

hotPoints = []
detectionPointSize = 64

pipAlpha = .7
pipScaleRatio = .35
pipResizeRatio=0.3

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


 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('proj_video.mp4')
 
# Check if camera opened successfully
if (cap.isOpened()== False):
    
    print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  
  if ret == True:
      
      #cv2.imshow('Frame',frame)
      
      
      roi= frame[crop[0]:crop[1],:]
      roiW, roiH = roi.shape[1], roi.shape[0]
      
      roi = np.expand_dims(roi, axis=0)
      detectionMap= []
      detectionMap = cnnModel.predict(roi)
      
      predictionMapH, predictionMapW = detectionMap.shape[1], detectionMap.shape[2]
      ratioH, ratioW = roiH / predictionMapH, roiW / predictionMapW
      
      detectionMap = detectionMap.reshape(detectionMap.shape[1], detectionMap.shape[2])
      detectionMap = detectionMap > confidenceThrd
      labels = label(detectionMap, structure= diagKernel)
      
      hotPoints = []
      
      for vehicleID in range(labels[1]):
          nz = (labels[0] == vehicleID + 1).nonzero()
          nzY = np.array(nz[0])
          nzX = np.array(nz[1])

    # +/-'s are manually derived adjustments for more appropriate boxes visualization
          xMin = np.min(nzX) - 32
          xMax = np.max(nzX) + 32

          yMin = np.min(nzY)
          yMax = np.max(nzY) + 64

    # Used to keep generated bounding boxes within a range of the label (a.k.a. vehicle) boundaries
          spanX = xMax - xMin
          spanY = yMax - yMin

          for x, y in zip(nzX, nzY):
              offsetX = (x - xMin) / spanX * detectionPointSize
              offsetY = (y - yMin) / spanY * detectionPointSize

        # Getting boundaries in ROI coordinates scale (multiplying by ratioW, ratioH)
              topLeftX = int(round(x * ratioW - offsetX, 0))
              topLeftY = int(round(y * ratioH - offsetY, 0))
              bottomLeftX = topLeftX + detectionPointSize
              bottomLeftY = topLeftY + detectionPointSize

              topLeft = (topLeftX, crop[0] + topLeftY)
              bottomRight = (bottomLeftX, crop[0] + bottomLeftY)

              hotPoints.append((topLeft, bottomRight))
    
              

        # Adjustment offsets for a box starting point.
        # Ranges from 0 for the left(upper)-most to detectionPointSize for right(bottom)-most
        
      
      mask = np.zeros_like(frame[:, :, 0]).astype(np.float)
      
           
      for box in hotPoints:
          
          
          topY = box[0][1]
          bottomY = box[1][1]
          leftX = box[0][0]
          rightX = box[1][0]
          
          mask[topY:bottomY, leftX:rightX] += 1
          mask = np.clip(mask, 0, 255)
          #print('haalaa2')
      
      cv2.imshow('heat',mask)
      currentFrameBoxes = label(mask)
      
      cmap=cv2.COLORMAP_JET
      heatMapInt = cv2.equalizeHist(mask.astype(np.uint8))
      heatColor = cv2.applyColorMap(heatMapInt, cmap)
      heatColor = cv2.cvtColor(heatColor, code=cv2.COLOR_BGR2RGB)
      heatWidth = int(mask.shape[1] *pipScaleRatio)
      
      for i in range(currentFrameBoxes[1]):
            nz = (currentFrameBoxes[0] == i + 1).nonzero()
            nzY = np.array(nz[0])
            nzX = np.array(nz[1])

            tlX = np.min(nzX)
            tlY = np.min(nzY)
            brX = np.max(nzX)
            brY = np.max(nzY)

            vehicleBoxesHistory.append([tlX, tlY, brX, brY])

            # Keeping last elements in history. Number defined by self.VeHiDepth
            vehicleBoxesHistory = vehicleBoxesHistory[-veHiDepth:]
      
      
      boxes, _ = cv2.groupRectangles(rectList=np.array(vehicleBoxesHistory).tolist(),
                                           groupThreshold=gdroupThr, eps=groupDiff)
      
      smallPip = cv2.resize(src=heatColor, dsize=(0, 0), fx=pipResizeRatio, fy=pipResizeRatio, interpolation=interpolation)
      pipHeight = smallPip.shape[0]
      pipWidth = smallPip.shape[1]
      backGround = frame[origin[1]:origin[1] + pipHeight, origin[0]:origin[0] + pipWidth]

      blend = np.round(backGround * (1 - pipAlpha), 0) + np.round(smallPip * pipAlpha, 0)

      blend = np.minimum(blend, 255)
      

      frame[origin[1]:origin[1] + pipHeight, origin[0]:origin[0] + pipWidth] = blend
      
      
      drawBoxes(frame,boxes)
      count= count+1
      play=str(count)
      cv2.putText(frame,play,bottomLeftCornerOfText,font,fontScale,
                  fontColor,lineType)
      
      
      cv2.imshow('box',frame)    
      
      
      
      
      
      
    # Press Q on keyboard to  exit
      if cv2.waitKey(25) & 0xFF == ord('q'):
          break
 
  # Break the loop
  else: 
    break
 

cap.release()
 

cv2.destroyAllWindows()



