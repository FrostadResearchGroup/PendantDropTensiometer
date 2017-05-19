# -*- coding: utf-8 -*-
"""
Created on Fri Jan 06 12:23:36 2017

@author: Yohan
"""
import sys
from PyQt4 import QtGui
import math
from scipy import stats
from PyQt4.QtGui import QFileDialog
from pendantdrop_gui import Ui_MainWindow
from skimage import feature
import matplotlib.image as mpimg
import cv2
import numpy as np

class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.line.axes: 
            return
        
        self.cutoff_line_coord_x = []
        self.cutoff_line_coord_y = []
            
        if len(self.xs)>1 or len(self.ys)>1:
            self.xs = []
            self.ys = []
        elif len(self.xs)==2 or len(self.ys)==2:   
            self.cutoff_line_coord_x = self.xs
            self.cutoff_line_coord_y = self.ys
        else:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            
        print self.xs, self.ys
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

class BoundaryDetect(QtGui.QMainWindow,Ui_MainWindow):
    def __init__(self,parent=None):
        super(BoundaryDetect,self).__init__(parent)
        self.setupUi(self)
        
        #set up buttons
        self.selectImageButton.clicked.connect(self.select_image)
        self.detectBoundaryButton.clicked.connect(self.detect_boundary)
        self.removeTubeButton.clicked.connect(self.isolate_drop)
        self.diameterSpinBox.valueChanged.connect(self.get_rotation_angle)
    
        #disable buttons until setup is complete
        self.reset_1st_stage()
    
    def reset_1st_stage(self):      
        self.mplwidget.axes.clear()
        self.mplwidget.figure.canvas.draw_idle()
        self.detectBoundaryButton.setEnabled(False)
        self.reset_2nd_stage()

    def reset_2nd_stage(self):
        self.edges = []
        self.intcoord = []
        self.dropCoord = []
        self.lineCoord = []
        self.adjLineCoord = []
        self.mplwidget2.axes.hold(False)
        self.mplwidget2.axes.clear()
        self.mplwidget2.figure.canvas.draw_idle()
        self.mplwidget3.axes.clear()
        self.mplwidget3.figure.canvas.draw_idle()
        self.diameterSpinBox.setEnabled(False)
        self.removeTubeButton.setEnabled(False)
        self.diameterSpinBox.setValue(0)
        self.diameterSpinBox.setEnabled(False)
        self.rotationAngleDisplay.setText("-")
        self.magRatioDisplay.setText("-")
        self.statusLabel.setText("")
        
    def select_image(self):
        #get image path from explorer, then clear previous image
        image_path = QFileDialog.getOpenFileName(self, 'Open file', 
         'c:\\',"Image files (*.jpg)")
        self.reset_1st_stage()
        
        #open image based on path, turn it into binary color using Otsu's method
        self.image = mpimg.imread(image_path,0)
        blur = cv2.GaussianBlur(self.image,(5,5),0)
        ret3,self.image_binary = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        #show image on mplwidget
        self.mplwidget.axes.imshow(self.image_binary,cmap='gray')
        self.mplwidget.figure.canvas.draw()
        self.detectBoundaryButton.setEnabled(True)
  
    def detect_boundary(self):
        #MAIN OBJ: detect image edges
        if(self.image_binary is not None):
            self.reset_2nd_stage()
            self.edges = feature.canny(self.image_binary,sigma=3)
            self.mplwidget2.axes.imshow(self.edges,cmap='gray')
            self.mplwidget2.figure.canvas.draw()
            self.get_interface_coordinates()

    def get_interface_coordinates(self):
        #MAIN OBJ: get the coordinates of the edges
        if(len(self.edges)!=0):
            self.intcoord = []
            for y in range(0,self.edges.shape[0]):
                edgecoord = []
                for x in range(0, self.edges.shape[1]): 
                    if self.edges[y, x] != 0:
                        edgecoord = edgecoord + [[x, y]]
                        
                #takes in only the outer most points
                if(len(edgecoord)>=2):
                    self.intcoord = self.intcoord + [edgecoord[0]] + [edgecoord[-1]]
                if(len(edgecoord)==1):
                    self.intcoord = self.intcoord + [edgecoord[0]]
                    
            self.intcoord = np.array(self.intcoord)
            self.mplwidget2.axes.scatter(self.intcoord[:,0],self.intcoord[:,1])
            self.mplwidget2.axes.hold(True)
            self.mplwidget2.figure.canvas.draw()
            self.diameterSpinBox.setEnabled(True)
            self.statusLabel.setText("Draw the boundary line on the second graph")
            self.draw_cutoff_line()
            
        else:
            self.statusLabel.setText("Analyze image first")
                
    def get_rotation_angle(self):
        #MAIN OBJ: get the rotation angle of the camera
        #get first few initial coordinates of the capillary tube
        self.diameter = self.diameterSpinBox.value()
        if (self.diameter > 0):
            self.lineCoord = self.intcoord[0:31:2]
            self.lineCoordX = [x[0] for x in self.lineCoord]
            self.lineCoordY = [x[1] for x in self.lineCoord]
                
            #calculate slope
            slope, intercept, r_value, p_value, std_err = stats.linregress(self.lineCoordX,self.lineCoordY)
                
            #use the inverse slope and flat line slope to calculate since
            #we can't compare infinite slope (vertical line) and actual slope        
            self.inverseSlope = -1/slope
            self.flatLineSlope = 0
                
            #trig function to calculate angle (in rads) using the slopes of 2 lines
            self.rotationAngle = math.atan((self.inverseSlope-self.flatLineSlope)/(1+(self.inverseSlope*self.flatLineSlope)))
            self.rotationAngleDisplay.setText(str(self.rotationAngle))
            self.get_magnification_ratio() 
        
        else:
            self.rotationAngleDisplay.setText("-")
            self.magRatioDisplay.setText("-")
        
    def get_magnification_ratio(self):
        #get the adjacent capillary tube line coordinates
        self.adjLineCoord = self.intcoord[1:31:2]
        
        #get the ends of both lines
        self.lineEnds1 = [self.lineCoord[0],self.lineCoord[-1]]
        self.lineEnds2 = [self.adjLineCoord[0],self.adjLineCoord[-1]]

        #calculate line distance and magnification ratio
        self.lineDistance = self.get_line_distance(self.lineEnds1,self.lineEnds2)
        self.magnificationRatio = self.diameter/self.lineDistance
        self.magRatioDisplay.setText(str(self.magnificationRatio))
        
    def get_line_distance(self, line1, line2):
        #step1: cross prod the two lines to find common perp vector
        (L1x1,L1y1),(L1x2,L1y2) = line1
        (L2x1,L2y1),(L2x2,L2y2) = line2
        L1dx,L1dy = L1x2-L1x1,L1y2-L1y1
        L2dx,L2dy = L2x2-L2x1,L2y2-L2y1
        commonperp_dx,commonperp_dy = (L1dy - L2dy, L2dx-L1dx)
    
        #step2: normalized_perp = perp vector / distance of common perp
        commonperp_length = math.hypot(commonperp_dx,commonperp_dy)
        commonperp_normalized_dx = commonperp_dx/float(commonperp_length)
        commonperp_normalized_dy = commonperp_dy/float(commonperp_length)
    
        #step3: length of (pointonline1-pointonline2 dotprod normalized_perp).
        # Note: According to the first link above, it's sufficient to
        #    "Take any point m on line 1 and any point n on line 2."
        #    Here I chose the startpoint of both lines
        shortestvector_dx = (L1x1-L2x1)*commonperp_normalized_dx
        shortestvector_dy = (L1y1-L2y1)*commonperp_normalized_dy
        mindist = math.hypot(shortestvector_dx,shortestvector_dy)
    
        #return results
        result = mindist
        return result
        
    def draw_cutoff_line(self):
        #draw the cutoff line
        self.cutoffLine, = self.mplwidget2.axes.plot([],[])
        self.lineBuilder = LineBuilder(self.cutoffLine)
        self.removeTubeButton.setEnabled(True)
        
    def calculate_dot_product(self, lineCoordX, lineCoordY, pointCoordX,pointCoordY):
        xp = (lineCoordX[1]-lineCoordX[0])*(pointCoordY-lineCoordY[0])-(lineCoordY[1]-lineCoordY[0])*(pointCoordX-lineCoordX[0])
        if xp < 0:
            return True
        else:
            return False

    def isolate_drop(self):
        #checks whether point is above or below line using the calculate_cross_product function above
        #if line_position is above (True), keep looping, else is below (False) therefore stop looping
       self.linePosition = True
       if self.cutoffLine is not None:
            self.lineCoordX = self.lineBuilder.xs
            self.lineCoordY = self.lineBuilder.ys
            
            #loop to find the outline coordinate where the outline coordinate is below the line
            for i in range (0, len(self.intcoord)):
                self.linePosition = self.calculate_dot_product(self.lineCoordX,self.lineCoordY,self.intcoord[i,0],self.intcoord[i,1])
                if self.linePosition is False:
                    self.cutoffPoint = i
                    break
            
            #self.dropcoord will be the coordinates used from now on
            self.dropCoord = self.intcoord
            
            #when it is found, remove all coordinates above that y-coordinate
            j = 0
            while j < self.cutoffPoint:
                self.dropCoord = np.delete(self.dropCoord,0,0)
                j += 1
            
            self.translate_drop()
            
       else:
            self.statusLabel.setText("Draw line first!")
            
    def translate_drop(self):
        #centers the drop   
        self.translationFactorX = (self.dropCoord[1,0]+self.dropCoord[0,0])*0.5
        for i in range(0,len(self.dropCoord)):
            self.dropCoord[i,0] = self.dropCoord[i,0] - self.translationFactorX
            
        #flip the drop vertically
        self.translationFactorY = self.dropCoord[0,1]
        for j in range(0,len(self.dropCoord)):
            self.dropCoord[j,1] = self.dropCoord[j,1] - self.translationFactorY
        self.dropMidPointY = (self.dropCoord[-1,1]+self.dropCoord[0,1])*0.5
        for k in range(0,len(self.dropCoord)):
            self.dropCoord[k,1] = self.dropMidPointY - (self.dropCoord[k,1]-self.dropMidPointY)
       
        np.savetxt('testfile.txt',self.dropCoord,delimiter=',',fmt="%i")
        
        self.mplwidget3.axes.scatter(self.dropCoord[:,0],self.dropCoord[:,1],color = 'red')
        self.mplwidget3.figure.canvas.draw()
        self.statusLabel.setText("Good to go for calculations!")
#        self.scale_rotate_drop()
        
#    def scale_rotate_drop(self):
#        self.nonDimCoord = self.dropCoord
#        self.x0 = self.y0 = 0
#        self.aspectRatio = self.aspRatioSpinBox.value()
#        self.rotationAngle = np.absolute(self.rotationAngle)
#        for i in range (0, len(self.dropCoord)-1):
#            self.nonDimCoord[i,0] = ((self.dropCoord[i,0]/self.magnificationRatio)-self.x0)*np.cos(self.rotationAngle) + ((self.aspectRatio*self.dropCoord[i,1]/self.magnificationRatio)-self.y0)*np.sin(self.rotationAngle)
#            self.nonDimCoord[i,1] = ((self.aspectRatio*self.dropCoord[i,1]/self.magnificationRatio)-self.x0)*np.cos(self.rotationAngle) - ((self.dropCoord[i,0]/self.magnificationRatio)-self.y0)*np.sin(self.rotationAngle)
#        print self.nonDimCoord
#        self.mplwidget3.axes.scatter(self.nonDimCoord[:,0],self.nonDimCoord[:,1],color = 'red')
#        self.mplwidget3.figure.canvas.draw()
        
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    MainWindow = BoundaryDetect()
    MainWindow.show()
    sys.exit(app.exec_())