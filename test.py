# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 10:04:21 2019

@author: achuiko
"""

import sys
import videostreamer
import cv2
import time
import numpy as np
import model
import detector
import engine
from config import get_config, save_config

if __name__ == '__main__':
    config, unparsed = get_config(sys.argv)

    if len(unparsed) > 0:
        raise RuntimeError("Unknown arguments were given! Check the command line!")
    print(config)

    norm_type = cv2.NORM_L1
    if config.norm_type == 0 : norm_type = cv2.NORM_L1
    elif config.norm_type == 1 : norm_type = cv2.NORM_L2
    elif config.norm_type == 2 : norm_type = cv2.NORM_L2SQR
    elif config.norm_type == 3 : norm_type = cv2.NORM_HAMMING
    else : norm_type = cv2.NORM_HAMMING2
      
    method = cv2.RANSAC
    if config.method == 0 : method = cv2.RANSAC
    elif config.method == 1 : method = cv2.LMEDS
    else : method = cv2.RHO
    # This class helps load input images from different sources.
    
    
    print("Build Networks")
    
    
    lift = engine.Engine(config)



    objDetector = detector.Detector(config.matcher_multiplier, norm_type, method, config.repr_threshold, config.max_iter, config.confidence)

    vs = videostreamer.VideoStreamer("camera", config.camid, config.H, config.W, 1, '')

    win = 'TF_LIFT Tracker'
    objwin = 'Object'
    cv2.namedWindow(win)
    cv2.namedWindow(objwin)
    
    print('Running Demo.')
  
    obj = model.ModelFile(config.object_path)
    greyObj = cv2.cvtColor(obj.image, cv2.COLOR_BGR2GRAY)
    
   
        
    key = cv2.waitKey() & 0xFF

    cv2.destroyAllWindows()

    print('==> Finshed Demo.')