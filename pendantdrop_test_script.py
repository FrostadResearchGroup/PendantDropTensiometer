# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:26:32 2017

@author: Yohan
"""
#import code blocks
import image_extraction as ie
import image_processing as ip

#import other modules
import tkFileDialog
import matplotlib.pyplot as plt
import os

#function to display image on plt
def display_plot(obj):
    plt.cla()
    plt.imshow(obj, cmap='gray')
    
def display_scat(obj):
    plt.cla()
    plt.scatter(obj[:,0],obj[:,1])
    
"""
image extraction
"""
#get image file path
filePath = tkFileDialog.askopenfilename()

#get image from file path
if os.path.isfile(filePath):
    image = ie.load_image_file(filePath)
    display_plot(image)
    print("image loaded")
else:
    print("file not loaded!")
    
"""
image processing
"""
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
print("boundary coordinates acquired")

#get rotation angle
rotationAngle = ip.get_rotation_angle(interfaceCoordinates)
print ("rotation angle against vertical axis is " + str(rotationAngle))

#get magnification ratio
actualDiameter = 1.67 #in mm
magnificationRatio = ip.get_magnification_ratio(interfaceCoordinates, actualDiameter)
print ("magnification ratio is " + str(magnificationRatio))

#isolate drop
pts = plt.ginput(1)
x,y = pts[0]
xCoords = [min(interfaceCoordinates[:,0]),max(interfaceCoordinates[:,0])] # map applies the function passed as 
yCoords = [y,y]
dropCoords = ip.isolate_drop(xCoords,yCoords,interfaceCoordinates)
display_scat(dropCoords)
print ("drop isolated")

#shift coordinates so apex is at 0,0
plt.gca().invert_yaxis()
newCenter = [0,0]
oldCenter = dropCoords[-1]
shiftedCoords = ip.shift_coords(dropCoords,newCenter,oldCenter)
display_scat(shiftedCoords)
print ("shifted apex to 0,0")

#rotate drop
rotatedCoords = ip.rotate_coords(shiftedCoords ,rotationAngle*-1,'degrees')
display_scat(rotatedCoords)
print("corrected angle")

#scale drop
scaledCoords = ip.scale_drop(rotatedCoords,magnificationRatio)
display_scat(scaledCoords)
print ("scaled coordinates according to magnification ratio")