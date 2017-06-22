# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:08:31 2017

@author: Yohan
"""
import cv2
import os

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
    
#For Testing Purposes    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    #test flags: change to True when testing specific functions
    flag1 = False
    flag2 = True

    #flag1 = test for load_image_file()
    if(flag1 == True):
        imagePath = "H2O in PDMS.jpg"
        if os.path.isfile(imagePath):
            image = load_image_file(imagePath)
            plt.imshow(image, cmap = "gray")
        else:
            print("file doesn't exist!")
        
    #flag2 = test for load_video_file()
    if (flag2 == True):
        videoPath = "SampleVideo_1280x720.mp4"
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
        
        