# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:26:32 2017

@author: Yohan
"""

#import code blocks
import image_extraction as ie
import image_processing as ip
import data_processing as dp

#import other modules
import tkFileDialog
import matplotlib.pyplot as plt
import os

##############################################################################

#input parameters  
deltaRho = 998
actualDiameter = 2.108 #in mm
reloads = 1

##############################################################################

#function to display image on plt
def display_plot(obj):
    plt.cla()
    plt.imshow(obj, cmap='gray')
    plt.pause(1)
    
def display_scat(obj):
    plt.cla()
    plt.scatter(obj[:,0],obj[:,1])
    plt.pause(1)
    
#get image file path
filePath = tkFileDialog.askopenfilename()

#get image from file path
if os.path.isfile(filePath):
    image = ie.load_image_file(filePath)
    display_plot(image)
    print("image loaded")
else:
    print("file not loaded!")
    
#binarize image
binarizedImage = ip.binarize_image(image)
display_plot(binarizedImage)
print("image binarized")

#detect image boundary
edges = ip.detect_boundary(binarizedImage)
display_plot(edges)
print("boundary traced")

#get interface coordinates
interfaceCoordinates = ip.get_interface_coordinates(edges)
plt.cla()
plt.scatter(interfaceCoordinates[:,0],interfaceCoordinates[:,1])
plt.pause(1)
print("boundary coordinates acquired")

#get magnification ratio
magnificationRatio = ip.get_magnification_ratio(interfaceCoordinates,actualDiameter)
print ("magnification ratio is " + str(magnificationRatio))

#isolate drop
pts = plt.ginput(1)
x,y = pts[0]
xCoords = [min(interfaceCoordinates[:,0]),max(interfaceCoordinates[:,0])] 
yCoords = [y,y]
dropCoords = ip.isolate_drop(xCoords,yCoords,interfaceCoordinates)
print ("drop isolated")

#shift coordinates so apex is at 0,0
plt.gca().invert_yaxis()
newCenter = [0,0]
shiftedCoords = ip.shift_coords(dropCoords[:,0],dropCoords[:,1],newCenter)
display_scat(shiftedCoords)
print ("shifted apex to 0,0")

#scale drop
scaledCoords = ip.scale_drop(shiftedCoords,magnificationRatio)
print ("scaled coordinates according to magnification ratio")

#reorder data points
xData,zData = ip.reorder_data(scaledCoords)
s,xe,apexRadiusGuess = dp.bond_calc(xData,zData)
bondInit = dp.s_interp(s,xe)

#run through optimization routine
surfTen,apexRadius,thetaRotation,bondNumber = dp.optimize_params(xData,zData,bondInit,
                                                apexRadiusGuess,deltaRho,reloads)

#output surface tension
print "Surface tension = %.4g mN/m" %(surfTen*10**3)
