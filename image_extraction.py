# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:08:31 2017

@author: Yohan
"""
import cv2
import ids

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
    cam = ids.Camera()
    cam.color_mode = ids.ids_core.COLOR_BGR8    # Get images in BGR format for JPEG compatibility
    cam.exposure = 5                            # Set initial exposure to 5ms
    cam.auto_exposure = True
    cam.continuous_capture = True               # Start image capture
    
    #get image
    meta = cam.next_save("testpic.jpg")
    return
    
#For Testing Purposes    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    #test flags: change to True when testing specific functions
    flag1 = False
    flag2 = False
    flag3 = False
    flag4 = False
    flag5 = False
    flag6 = False
    
    #flag1 = test for load_image_file()
    if(flag1 == True):
        imagePath = "H20 in PDMS.jpg"
        image = load_image_file(imagePath)
        plt.imshow(image, cmap = "gray")
        
    #flag2 = test for load_video_file()
    if (flag2 == True):
        videoPath = "SampleVideo_1280x720.mp4"
        cap = load_video_file(videoPath)
        while(cap.isOpened()):
            ret, frame = cap.read()
        
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            cv2.imshow('frame',gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    #flag3 = test for get_new_image()
    if(flag3 == True):
        get_new_image()
        
        