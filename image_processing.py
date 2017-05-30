# -*- coding: utf-8 -*-
"""
Created on Mon May 29 10:03:52 2017

@author: Yohan
"""
import math
from scipy import stats
from skimage import feature
import cv2
import numpy as np

def binarize_image(image):
    #binarize image to convert the image to purely black and white image
    blur = cv2.GaussianBlur(image,(5,5),0)
    ret3,binaryImage = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return binaryImage
    
def detect_boundary(binaryImage):
    #detect the outline of the binary image
    edges = feature.canny(binaryImage,sigma=3)
    return edges
    
def get_interface_coordinates(edges):
    #go through each pixel in the edges image to extract the coordinates of the edges
    interfaceCoords = []
    for y in range(0,edges.shape[0]):
        edgeCoords = []
        for x in range(0, edges.shape[1]): 
            if edges[y, x] != 0:
                edgeCoords = edgeCoords + [[x, y]]       
        #takes in only the outer most points
        if(len(edgeCoords)>=2):
            interfaceCoords = interfaceCoords + [edgeCoords[0]] + [edgeCoords[-1]]
        if(len(edgeCoords)==1):
            interfaceCoords = interfaceCoords + [edgeCoords[0]]      
    interfaceCoords = np.array(interfaceCoords)
    return interfaceCoords
    
def get_rotation_angle(interfaceCoords):
    #get coordinates of a few points along the capillary tube
    lineCoords = interfaceCoords[0:31:2]
    lineCoordsX = [x[0] for x in lineCoords]
    lineCoordsY = [x[1] for x in lineCoords]
        
    #use the inverse slope and flat line slope to calculate since
    #we can't compare infinite slope (vertical line) and actual slope
        
    #calculate inverse slope
    slope, intercept, r_value, p_value, std_err = stats.linregress(lineCoordsY,lineCoordsX)
    flatLineSlope = 0
        
    #trig function to calculate angle (in rads) using the slopes of 2 lines
    rotationAngle = math.atan((slope-flatLineSlope)/(1+(slope*flatLineSlope)))
    
    return rotationAngle
    
def get_min_distance(line1, line2):
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
    minDistance = math.hypot(shortestvector_dx,shortestvector_dy)

    return minDistance
    
def get_magnification_ratio(interfaceCoords, actualDiameter):
    #get a few of the adjacent capillary tube line coordinates
    lineCoords = interfaceCoords[0:31:2]
    adjLineCoords = interfaceCoords[1:31:2]
    
    #get the ends of both lines
    lineEnds1 = [lineCoords[0],lineCoords[-1]]
    lineEnds2 = [adjLineCoords[0],adjLineCoords[-1]]

    #calculate line distance and magnification ratio
    lineDistance = get_min_distance(lineEnds1,lineEnds2)
    magnificationRatio = actualDiameter/lineDistance
    
    return magnificationRatio
    
def calculate_dot_product(vectEndX, vectEndY, pointCoordX, pointCoordY):
    xp = (vectEndX[1]-vectEndX[0])*(pointCoordY-vectEndY[0])-(vectEndY[1]-vectEndY[0])*(pointCoordX-vectEndX[0])
    if xp < 0:
        return True
    else:
        return False
        
def isolate_drop(lineBuilder, interfaceCoords):
    #checks whether point is above or below line using the calculate_cross_product function above
    #if line_position is above (True), keep looping, else is below (False) therefore stop looping
    linePosition = True
    lineCoordX = lineBuilder.xs
    lineCoordY = lineBuilder.ys
    
    #loop to find the outline coordinate where the outline coordinate is below the line
    for i in range (0, len(interfaceCoords)):
        linePosition = calculate_dot_product(lineCoordX, lineCoordY,interfaceCoords[i,0], interfaceCoords[i,1])
        if linePosition is False:
            cutoffPoint = i
            break
    
    #self.dropcoord will be the coordinates used from now on
    dropCoords = interfaceCoords
    
    #when it is found, remove all coordinates above that y-coordinate
    j = 0
    while j < cutoffPoint:
        dropCoords = np.delete(dropCoords,0,0)
        j += 1

    return dropCoords
    
def translate_drop(dropCoords):
    #centers the drop   
    translationFactorX = (dropCoords[1,0] + dropCoords[0,0])*0.5
    for i in range(0,len(dropCoords)-1):
        dropCoords[i,0] = dropCoords[i,0] - translationFactorX
        
    #flip the drop vertically
    translationFactorY = dropCoords[0,1]
    for j in range(0,len(dropCoords)-1):
        dropCoords[j,1] = dropCoords[j,1] - translationFactorY
        
    dropMidPointY = (dropCoords[-1,1]+dropCoords[0,1])*0.5
    
    for k in range(0,len(dropCoords)-1):
        dropCoords[k,1] = dropMidPointY - (dropCoords[k,1] - dropMidPointY)
        
    translatedDropCoords = dropCoords
    return translatedDropCoords
    
def scale_drop(translatedDropCoords, magnificationRatio):
    #multiply the coordinates by the magnification ratio to get the true coordinates in mm
    scaledDropCoordsX = []
    scaledDropCoordsY = []
    scaledDropCoords = []
    
    for i in range(0,len(translatedDropCoords)-1):
        floatX = translatedDropCoords[i,0] * magnificationRatio
        floatY = translatedDropCoords[i,1] * magnificationRatio
        scaledDropCoordsX.append(floatX)
        scaledDropCoordsY.append(floatY)
        
    for j in range(0,len(scaledDropCoordsX)):
        scaledDropCoords[j] = [scaledDropCoordsX[j], scaledDropCoordsY[j]]
        
    return scaledDropCoords
    
#For Testing Purposes    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
    #test flags
    flag1 = False
    flag2 = False
    flag3 = False
    flag4 = False
    flag5 = True
    flag6 = False
    flag7 = False
    flag8 = False
    flag9 = False
    flag10 = False
    
    #test codes
    #flag1 = test for binarize_image()
    if (flag1 == True):
        img = mpimg.imread('H2O in PDMS.jpg')
        binarizedImage = binarize_image(img)
        plt.imshow(binarizedImage,cmap='gray')
    
    #flag2 = test for detect_boundary()
    if(flag2 == True):
        img = mpimg.imread('H2O in PDMS.jpg')
        binarizedImage = binarize_image(img)
        edges = detect_boundary(binarizedImage)
        plt.imshow(edges,cmap='gray')
        
    #flag3 = test for get_interface_coordinates()       
    if(flag3 == True):
        img = mpimg.imread('H2O in PDMS.jpg')
        binarizedImage = binarize_image(img)
        edges = detect_boundary(binarizedImage)
        interfaceCoordinates = get_interface_coordinates(edges)
        print interfaceCoordinates
        plt.scatter(interfaceCoordinates[:,0],interfaceCoordinates[:,1])
        
    #flag4 = test for get_rotation_angle()
    if(flag4 == True):
        #set up test array
        testArray = []
        for i in range(0,39):
            testArray.append([i,i+0.5*i])
        print testArray
        
        rotationAngle = get_rotation_angle(testArray)
        
        #convert to degrees
        rotationAngleDegrees = rotationAngle*360/(2*math.pi)
        print rotationAngleDegrees
        
    #flag5 = test for get_min_distance()
    if(flag5 == True):
        lineEnds1 = [[0,0],[0,15]]
        lineEnds2 = [[5,0],[6,15]]
        minDistance = get_min_distance(lineEnds1,lineEnds2)
        print minDistance
        