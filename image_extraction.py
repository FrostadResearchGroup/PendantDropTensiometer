# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:08:31 2017

@author: Yohan
"""
import cv2
import os
import time
import ctypes
import numpy as np

def load_image_file(filePath):
    """
    loads image into GUI. Returns binarized image
    """
    image = cv2.imread(filePath,0)
    return image
    
def load_video_file(filePath):
    """
    loads video into GUI
    """
    vid = cv2.VideoCapture(filePath)
    return vid
    
def get_image(width, height, cameraNum=0):
    
    #include full path or copy dll into same folder as .py script
    uEyeDll = ctypes.cdll.LoadLibrary("C:\Windows\SysWOW64\ueye_api.dll") 
    
    #connect camera
    cam = ctypes.c_uint32(cameraNum)
    hWnd = ctypes.c_voidp()
    msg = uEyeDll.is_InitCamera(ctypes.byref(cam),hWnd)
    ErrChk = uEyeDll.is_EnableAutoExit (cam, ctypes.c_uint(1))
    if ~ErrChk:
        print (' Camera Connected')
    IS_CM_SENSOR_RAW8  = ctypes.c_int(11)
    nRet = uEyeDll.is_SetColorMode(cam,IS_CM_SENSOR_RAW8)
    IS_SET_TRIGGER_SOFTWARE = ctypes.c_uint(0x1000)
    nRet = uEyeDll.is_SetExternalTrigger(cam, IS_SET_TRIGGER_SOFTWARE)
    
    
    #allocate memory
    width_py = width
    height_py = height
    pixels_py = 8
    
    width = ctypes.c_int(width_py) #convert python values into c++ integers
    height = ctypes.c_int(height_py) 
    bitspixel = ctypes.c_int(pixels_py)
    pcImgMem = ctypes.c_char_p() #create placeholder for image memory
    pid = ctypes.c_int()
    
    ErrChk = uEyeDll.is_AllocImageMem(cam, width, height,  bitspixel, 
                                    ctypes.byref(pcImgMem), ctypes.byref(pid))
    if ~ErrChk:
        print (' Success')
    else:
        print (' Memory allocation failed, no camera with value' + str(cam.value))
    
    
    # Get image data    
    uEyeDll.is_SetImageMem(cam, pcImgMem, pid)
#    nRed = uEyeDll.IS_GET_RED_GAIN(cam)
#    nGreen = uEyeDll.IS_GET_GREEN_GAIN(cam)
#    nBlue = uEyeDll.IS_GET_BLUE_GAIN(cam)
#    nMaster = uEyeDll.IS_GET_MASTER_GAIN(cam)
#    print nMaster, nRed, nGreen, nBlue
#    nMaster = ctypes.c_int(60)
#    ret = uEyeDll.is_SetHardwareGain(cam, nMaster, nRed, nGreen, nBlue)
#    print nMaster, nRed, nGreen, nBlue
    ImageData = np.ones((height_py,width_py),dtype=np.uint8)
    print ImageData

    #put these lines inside a while loop to return continous images to the array "ImageData"  
    print np.max(ImageData)
    uEyeDll.is_FreezeVideo (cam, ctypes.c_int(0x8000))  #IS_DONT_WAIT  = 0x0000, or IS_GET_LIVE = 0x8000
    uEyeDll.is_CopyImageMem (cam, pcImgMem, pid, ImageData.ctypes.data)
    print np.max(ImageData)

    uEyeDll.is_ExitCamera(cam)
    
    return ImageData
    
def get_image_cv2(width, height, cameraNum=0):
    """
    Connect to camera and capture a single image with the current time and 
    return.
    """
    
    vid = cv2.VideoCapture(cameraNum)    
    vid.set(3,width)
    vid.set(4,height)

    ret = vid.grab()
    imgTime = time.time()
    print ret
    ret, im = vid.retrieve()
    print ret
    vid.release()

    return im#, imgTime
    
def get_image_VC():
    from VideoCapture import Device

    cam = Device()
    cam.setResolution(800, 400)

    img = cam.getImage()
    return img
        
#For Testing Purposes    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    #test flags: change to True when testing specific functions
    testLoadImage = False
    testLoadVideo = False
    testGetImage = True

    if testGetImage:
        im = get_image_cv2(500,500)
        cv2.imshow('Test',im)
#        im = get_image_VC()
#        im.show()
#        print im
        
    
    if testLoadImage:
        imagePath = "H2O in PDMS.jpg" #change paths accordingly
        if os.path.isfile(imagePath):
            image = load_image_file(imagePath)
            plt.imshow(image, cmap = "gray")
        else:
            print("file doesn't exist!")
        
    #flag2 = test for load_video_file()
    if testLoadVideo:
        videoPath = "SampleVideo_1280x720.mp4" #change paths accordingly
        if os.path.isfile(videoPath):
            cap = load_video_file(videoPath)
            ret, frame = cap.read()
            while(ret):
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow('outVideo', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                ret, frame = cap.read()
                
            cap.release()
            cv2.destroyAllWindows()
        else:
            print("file doesn't exist!")
