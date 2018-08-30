# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 17:53:39 2018

@author: Frostad
"""

#import python modules
from PyQt4 import QtCore, QtGui, uic
from matplotlib.backends.qt_compat import QtWidgets
import cPickle as pkl
from pyueye import ueye
import sys
import cv2
import numpy as np
import copy
import threading
import Queue
import csv
import time
from Tkinter import *
import tkFileDialog

#import Custom modules
import image_extraction as ie
import dropletQuickAnalysis

#load GUI class
filename = "pendantDropAnalysis.ui"
form_class = uic.loadUiType(filename)[0]

class OwnImageWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None
        
    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()
        
class ApplicationWindow(QtGui.QMainWindow,form_class):
    
    def __init__(self, parent=None):
        """
        Initialize the instance of the GUI object.
        """
        QtGui.QMainWindow.__init__(self, parent)
        self.setupUi(self)
        
        # image resolution
        self.width = 2560
        self.height = 1920          
        
        #define image extension 
        self.imageExt = '.jpg'       
        
        #define physical parameters for droplet analysis
        self.deltaT = 2 #K
        self.thermalExpCoeff = 0.000214 #1/K or 1/C
        self.trueSyringeRotation = 0 #unsure of true syringe rotation       
        
        #Initialize thread and queue for generating live feed
        self.q = Queue.Queue()
        self.capture_thread = threading.Thread(target=self.grab)          

        # timers and threading
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)
        
        #Initializing toggle states
        self.running = False
        self.outline = False
        self.capillary = False
        self.recording = False
        self.dropletAnalysis = False
        
        #Initializing quick droplet calculations
        self.vol = None
        self.bond = None
        
        #Identify Toggle States
        self.selectFolder.clicked.connect(self.saveToFolder)
        self.startButton.clicked.connect(self.start_clicked)
        self.showOutline.clicked.connect(self.outline_on)
        self.capillaryButton.clicked.connect(self.get_capImage)
        self.beginDropAnalysisButton.clicked.connect(self.get_drop_features)
        self.runDippingTestButton.clicked.connect(self.start_dipping)
        self.recordingButton.clicked.connect(self.start_recording)
        self.stopButton.clicked.connect(self.stop_recording)
        self.save.clicked.connect(self.saveData)

        #Grabs the GUI geometrical properties
        self.window_width = self.ImgWidget.frameSize().width()
        self.window_height = self.ImgWidget.frameSize().height()
        self.ImgWidget = OwnImageWidget(self.ImgWidget)
        
        #Resets widgets and assigned values
        self.startButton.setEnabled(False)
        self.capillaryButton.setEnabled(False)
        self.recordingButton.setEnabled(False)
        self.stopButton.setEnabled(False)
        self.beginDropAnalysisButton.setEnabled(False)
        self.runDippingTestButton.setEnabled(False)
        self.dropVolText.setEnabled(False)
        self.dropVolUnits.setEnabled(False)
        self.bondNumText.setEnabled(False)
        self.motorSpeed.setEnabled(False)
        self.motorSpeedText.setEnabled(False)
        self.motorSpeedUnits.setEnabled(False)
        self.dipDist.setEnabled(False)
        self.dipDistText.setEnabled(False)
        self.dipDistUnits.setEnabled(False) 
        self.dipTime.setEnabled(False)
        self.dipTimeText.setEnabled(False)
        self.dipTimeUnits.setEnabled(False)        
        self.pressDiffText.setEnabled(False)
        self.capDiamText.setEnabled(False)
        self.capDiamUnits.setEnabled(False)
        self.save.setEnabled(False)

        #initialize dictionary and keys for storing data
        self.data = {}
        self.data['dropImage'] = {} 
        self.data['time'] = {}       
        
        #Initializes input values
        self.pressDiff.setPlainText('998')
        self.capDiam.setPlainText('2.10')

    def outline_on(self):
        """
        Toggle for tracking edge profile of image.
        """
        
        self.outline = self.showOutline.checkState()
                
    def get_capImage(self):
        """
        Toggle for taking a capillary image and importing input values.
        """
        self.capillary = True
        self.capillaryButton.setEnabled(False)
        self.beginDropAnalysisButton.setEnabled(True)
        
        #grab capillary diameter and pressure differential inputs
        self.capillaryDiameter = float(self.capDiam.toPlainText())
        self.deltaRho = float(self.pressDiff.toPlainText())
        self.capDiam.setEnabled(False)
        self.pressDiff.setEnabled(False)
        self.inputsText.setEnabled(False)
        
        # save input parameters to dictionary
        self.data['capDiam'] = self.capillaryDiameter
        self.data['pressDiff'] = self.deltaRho
                
    def get_drop_features(self):
        """
        Toggle for outputting geometrical features of droplet (quick analysis).
        """

        self.dropletAnalysis = True
        self.beginDropAnalysisButton.setEnabled(False)
        self.runDippingTestButton.setEnabled(True)

    def start_dipping(self):
        """
        Toggle for starting dipping tests.
        """
        
        self.dropletAnalysis = False
        self.runDippingTestButton.setEnabled(False)
        self.recordingButton.setEnabled(True)
        
        # store initial droplet parameters in dictionary (as first element)
        self.data['dropImage'][0] = self.dropImage
        self.data['time'][0] = 0

    def start_clicked(self):
        """
        Commands for pushing the preview button.
        """
        self.capture_thread.start()
        self.startButton.setEnabled(False)
        self.startButton.setText('Starting...')
        self.capillaryButton.setEnabled(True)
        self.running = True

    def update_frame(self):
        """
        Updating frames in a thread.
        """
        
        # Checks if camera is receiving images
        if not self.q.empty():
            self.startButton.setText('Camera is live')
            frame = self.q.get()
            img = frame["img"]
            
            if self.dropletAnalysis == True and self.vol is not None:
                self.dropVol.setPlainText(str(self.vol))
                self.bondNum.setPlainText(str(self.bond))  
                
            # Uses geometrical features to scale and resize image. 
            img_height, img_width, img_colors = img.shape
            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])

            if scale == 0:
                scale = 1
            
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, bpc = img.shape
            bpl = bpc * width
                               
            image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
            self.ImgWidget.setImage(image)

    def start_recording(self):
        """
        Start recording images.
        """

        self.recording = True
        self.recordingButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self.save.setEnabled(False)

    def stop_recording(self):
        """
        Stop recording images.
        """
        self.recording = False
        self.stopButton.setEnabled(False)
        self.recordingButton.setEnabled(True)
        self.save.setEnabled(True)

    def closeEvent(self, event):
        """
        Close window and disconnects camera.
        """
        self.running = False
        if self.initTrigger is not None:
            ret = ueye.is_ExitCamera(self.hcam)
    
    def saveData(self):
        """
        Saves data to file (.pkl), disconnects camera after and closes window 
        (need to figure out how to close GUI dialog box with save button).
        """
        self.startButton.setEnabled(False)
        self.recordingButton.setEnabled(False)
        self.save.setEnabled(False)
        self.save = True
        self.running = False

        
    def saveToFolder(self):
        """
        Opens dialog box for selecting folder for saving.
        """
        self.selectFolder.setEnabled(False)
        root = Tk()
        root.directory = tkFileDialog.askdirectory()
        self.folderName = root.directory
        root.destroy()
        self.startButton.setEnabled(True)
       
    def grab(self):
        """
        Grabs live image from camera feed. 
        """
           
        # init camera
        self.hcam = ueye.HIDS(0)
        self.initTrigger = ueye.is_InitCamera(self.hcam, None)

        # set color mode
        ret = ueye.is_SetColorMode(self.hcam, ueye.IS_CM_BGR8_PACKED)

        # set region of interest
        rect_aoi = ueye.IS_RECT()
        rect_aoi.s32X = ueye.int(0)
        rect_aoi.s32Y = ueye.int(0)
        rect_aoi.s32Width = ueye.int(self.width)
        rect_aoi.s32Height = ueye.int(self.height)
        ueye.is_AOI(self.hcam, ueye.IS_AOI_IMAGE_SET_AOI, rect_aoi, ueye.sizeof(rect_aoi))
     
        # allocate memory
        mem_ptr = ueye.c_mem_p()
        mem_id = ueye.int()
        bitspixel = 24 # for colormode = IS_CM_BGR8_PACKED
        ret = ueye.is_AllocImageMem(self.hcam, self.width, self.height, bitspixel,
                                    mem_ptr, mem_id)
      
        # set active memory region
        ret = ueye.is_SetImageMem(self.hcam, mem_ptr, mem_id)
    
        # continuous capture to memory
        ret = ueye.is_CaptureVideo(self.hcam, ueye.IS_DONT_WAIT)
       
        # get data from camera and display
        lineinc = self.width * int((bitspixel + 7) / 8)
        
        #initialize counter
        j = 1     
              
        while(self.running):
            
            frame = {} 

            if j == 1:
                
                startTime = time.time()       
            
            endTime = time.time()            
            
            img = ueye.get_data(mem_ptr, self.width, self.height, bitspixel, lineinc, copy=True)
            
            img = np.reshape(img, (self.height, self.width, 3))
            
            blkImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            #Check for edge detector toggle state
            if self.outline:
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                #create threshold on image to detect edges
                ret,thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)    
                edges = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
                
                if edges:
                    #change to size based on contour area
                    contour = max(edges,key=cv2.contourArea)  
                
                else:
                    
                    contour = None
                    
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                if contour is not None:
                               
                    cv2.drawContours(img,contour,-1,(0,255,0),6)

            #Check for event - taking capillary image
            if self.capillary:
                
                # load capillary image and store in dictionary
                self.capImage = copy.copy(blkImg)
                self.data['capImage'] = self.capImage
                
                # toggle capillary capture off
                self.capillary = False

            #Check for event - gather droplet data (volume and bond number)       
            if self.dropletAnalysis:
                
                # load drop image 
                self.dropImage = copy.copy(blkImg)  

                vals = np.array([self.deltaRho,self.capillaryDiameter,self.thermalExpCoeff,
                           self.trueSyringeRotation,self.deltaT])
                
                ret = dropletQuickAnalysis.get_droplet_geometry(vals,self.capImage,self.dropImage) 
                
                # output droplet geometry parameters 
                self.vol = ret[0]
                self.bond = ret[1]

            if self.recording:
    
                # grab timestamp
                timeVal = endTime - startTime
                
                # output droplet images and timestamp and store in dictionary
                self.data['dropImage'][j] = self.dropImage
                self.data['time'][j] = timeVal
                           
                j=j+1
            
            # write image to frame dictionary
            frame['img'] = img
    
            # sleep command to avoid build up in queue
            time.sleep(0.01)
            
            # write image to frame
            if self.q.qsize() < 10:
                self.q.put(frame)

            if self.save:
                
                saveFile = self.folderName + '/outputData.pickle'
                
                with open(saveFile, 'wb') as handle:
                    pkl.dump(self.data, handle)

def main():
    app = QtGui.QApplication(sys.argv)
    form = ApplicationWindow(None)
    form.setWindowTitle('Pendant Drop Analysis Software')
    form.show()
    app.exec_()
    

if __name__ == '__main__':
    main()
