# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:08:31 2017

@author: Yohan
"""

from PyQt4.QtGui import QFileDialog
import cv2
import matplotlib.image as mpimg
import numpy as  np
import ctypes

#change the file path accordingly or have the dll file in the same folder
uEyeDll = ctypes.cdll.LoadLibrary("ueye_api_64.dll")

def load_image_file(filePath):
    """
    loads image into GUI. Returns binarized image
    """
    image = cv2.imread(filePath,0)
    blur = cv2.GaussianBlur(image,(5,5),0)
    ret3, binaryImage = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return binaryImage
    
def load_video_file(filePath):
    """
    loads video into GUI
    """
    vid = cv2.VideoCapture(filePath)
    return vid
    
def get_new_image():
    """
    use the uEye camera to take picture
    """
    #connect camera
    cam = ctypes.c_uint32(0)
    hWnd = ctypes.c_voidp()
    msg=uEyeDll.is_InitCamera(ctypes.byref(cam),hWnd)
    ErrChk=uEyeDll.is_EnableAutoExit (cam, ctypes.c_uint(1))
    if ~ErrChk:
        print (' Camera Connected')
    IS_CM_SENSOR_RAW8  =ctypes.c_int(11)
    nRet = uEyeDll.is_SetColorMode(cam,IS_CM_SENSOR_RAW8)
    IS_SET_TRIGGER_SOFTWARE = ctypes.c_uint(0x1000)
    nRet = uEyeDll.is_SetExternalTrigger(cam, IS_SET_TRIGGER_SOFTWARE)
    
    #allocate memory
    width_py = 1600
    height_py = 1200
    pixels_py =8
    
    width = ctypes.c_int(width_py) #convert python values into c++ integers
    height = ctypes.c_int(height_py) 
    bitspixel=ctypes.c_int(pixels_py)
    pcImgMem = ctypes.c_char_p() #create placeholder for image memory
    pid=ctypes.c_int()
    
    ErrChk=uEyeDll.is_AllocImageMem(cam, width, height,  bitspixel, ctypes.byref(pcImgMem), ctypes.byref(pid))
    if ~ErrChk:
        print (' Success')
    else:
        print (' Memory allocation failed, no camera with value' +str(cam.value))
        
    # Get image data    
    uEyeDll.is_SetImageMem(cam, pcImgMem, pid)
    ImageData = np.ones((height_py,width_py),dtype=np.uint8)
    uEyeDll.is_FreezeVideo (cam, ctypes.c_int(0x0000))  #IS_DONT_WAIT  = 0x0000, or IS_GET_LIVE = 0x8000
    uEyeDll.is_CopyImageMem (cam, pcImgMem, pid, ImageData.ctypes.data)
    return ImageData
    
def get_new_video():
    
    
def preview_camera_view(mplwidget, item, idCode):
    """
    previews the image or video. Takes in an idCode: 0 for image, 1 for video
    """
    if(idCode == 0):
        mplwidget.axes.imshow(item, cmap ='gray')
        mplwidget.figure.canvas.draw()
    else if(idCode == 1):
        
    else:
        return None
        
def analyze_frame():
    
    
    
