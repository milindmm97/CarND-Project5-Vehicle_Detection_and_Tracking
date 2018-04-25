# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 15:52:57 2018

@author: Milind
"""
from moviepy.editor import ImageSequenceClip, VideoFileClip
import cv2
from tqdm import tqdm

resultFrames = []

clipFileName = input('Enter video file name: ')



clip = VideoFileClip(clipFileName)

for frame in tqdm(clip.iter_frames()):
    
    #dst = ld.embedDetections(src=frame, pipParams=pipParams)
    cv2.imshow('hala',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    #resultFrames.append(dst)
    
cv2.destroyAllWindows()
