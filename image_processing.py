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
import data_processing as dp

def binarize_image(image):
    """
    binarize image to convert the image to purely black and white image
    """
    blur = cv2.GaussianBlur(image,(5,5),0)
    ret3,binaryImage = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return binaryImage
    
def detect_boundary(binaryImage):
    #detect the outline of the binary image
    edges = feature.canny(binaryImage,sigma=3.75)
    return edges
    
def get_interface_coordinates(edges):
    """
    go through each pixel in the edges image to extract the coordinates of the edges
    """
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
    """
    get coordinates of a few points along the capillary tube
    """
    lineCoords = interfaceCoords[0:51:2]
    lineCoordsX = [x[0] for x in lineCoords]
    lineCoordsY = [x[1] for x in lineCoords]
        
    #use the inverse slope and flat line slope to calculate since
    #we can't compare infinite slope (vertical line) and actual slope
        
    #calculate inverse slope
    slope, intercept, r_value, p_value, std_err = stats.linregress(lineCoordsY,lineCoordsX)
    flatLineSlope = 0
        
    #trig function to calculate angle (in rads) using the slopes of 2 lines
    rotationAngle = math.atan((slope-flatLineSlope)/(1+(slope*flatLineSlope)))
    rotationAngleDegrees = rotationAngle*360/(2*math.pi)
    
    #if rotational angle is being set to horizontal (quick and dirty fix)
    if rotationAngleDegrees > 45:
        rotationAngleDegrees = 90-rotationAngleDegrees

    return rotationAngleDegrees
    
def get_min_distance(line1, line2):
    """
    
    """
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
    """
    get a few of the adjacent capillary tube line coordinates
    """
    lineCoords = interfaceCoords[0:61:2]
    adjLineCoords = interfaceCoords[1:61:2]
    
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
        
def isolate_drop(lineCoordX, lineCoordY, interfaceCoords):
    """
    checks whether point is above or below line using the calculate_cross_product function above
    if line_position is above (True), keep looping, else is below (False) therefore stop looping
    """
    linePosition = True
    cutoffPoint = None    
    
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

def shift_coords(xCoords, zCoords, newCenter):
    """ 
    Shift the coordinates so that the orgin is at the specified center.
    
    coords = ndarray (N,i) where i is the dimensionality (i.e 2D).
    newCenter = ndarray (1,i)
    oldCenter (optional) = ndarray(1,i)
    """
    
    #determine offset of current droplet center
    xOffset =  (max(xCoords) + min(xCoords))/2
    zOffset =   zCoords[-1]
    print(zOffset)
    oldCenter = [xOffset,zOffset]
    
    #centers the drop
    xDifference = oldCenter[0] - newCenter[0]
    zDifference = oldCenter[1] - newCenter[1]
    xCoords -= xDifference
    zCoords -= zDifference
    
    coords = np.append([xCoords],[zCoords],axis=0).transpose()

    #flip the coordinates vertically
    coords *= [1,-1]
    return coords
    
def scale_drop(coords, magnificationRatio):
    """ 
    Scale the coordinate based on the magnification ratio
    
    coords = ndarray (N,i) where i is the dimensionality (i.e 2D).
    magnificationRatio = float
    """
    #changing units to meters
    scaledCoords = coords * [magnificationRatio*10**-3, 
                             magnificationRatio*10**-3] 
    return scaledCoords
    
def rotate_coords(coords,angle,format='radians'):
    """ 
    Shift the coordinates so that the orgin is at the specified center.
    Rotates COUNTER-CLOCKWISE respective to the angle given in the argument
    
    coords = ndarray (N,i) where i is the dimensionality (i.e 2D).
    angle = float in radians
    format (optional) = string specifying radians or degrees
    """
    x = coords[:,0]
    y = coords[:,1]
    if format is 'degrees':
        angle *= np.pi/180
    xRot = x*np.cos(angle) - y*np.sin(angle)
    yRot = x*np.sin(angle) + y*np.cos(angle)
    coords[:,0] = xRot
    coords[:,1] = yRot
    return coords    

def reorder_data(coords):
    """
    Re-order data into arrays that smoothly follow the drop profile (connected)
    """
    
    coords = coords[coords[:,1].argsort()]
    
    xData = coords[:,0]
    zData = coords[:,1]        
    
    xDataRight = ()
    zDataRight = ()
    xDataLeft = ()
    zDataLeft = ()
    
    # seperate left and right side of droplet by positive or negative x-coord value
    for i in range(len(xData)):
    
        if xData[i] < 0:
            xDataLeft = np.append(xDataLeft,xData[i])
            zDataLeft = np.append(zDataLeft,zData[i])
        elif xData[i] > 0:
            xDataRight = np.append(xDataRight,xData[i])
            zDataRight = np.append(zDataRight,zData[i])
        else:
            xDataLeft = np.append(xDataLeft,xData[i])
            zDataLeft = np.append(zDataLeft,zData[i])
            xDataRight = np.append(xDataRight,xData[i])
            zDataRight = np.append(zDataRight,zData[i])

                       
    # Reorders data in descending z order
    indexLeft = np.lexsort((xDataLeft,-1*zDataLeft))
    
    indexRight = np.lexsort((xDataRight,zDataRight))    
    print(indexRight)
    xDataLeft = xDataLeft[indexLeft]
    xDataRight = xDataRight[indexRight]
    zDataLeft = zDataLeft[indexLeft]
    zDataRight = zDataRight[indexRight]
    
    xCoords = np.append(xDataLeft,xDataRight[1:])
    zCoords = np.append(zDataLeft,zDataRight[1:])

    return xCoords,zCoords         
               

#For Testing Purposes    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
    #test array for flags 9, 10 ,11
    sigma = 0.06
    r0 = .0015
    deltaRho = 998
    L = 3.5
    nPoints = 1000
    Bond_actual = deltaRho*9.81*r0**2/sigma
    xActual,zActual = dp.get_test_data(sigma,r0,deltaRho,nPoints,L)
    
#    plt.figure()
#    plt.plot(xActual,zActual,'x')
#    plt.axis('equal')    
    
    testArray = np.append([xActual],[zActual],axis=0).transpose()    
    
    #test flags: change to True when testing specific functions
    
    #flag1 = test for binarize_image()
    flag1 = False
    
    #flag2 = test for detect_boundary()
    flag2 = False
    
    #flag3 = test for get_interface_coordinates()
    flag3 = False
    
    #flag4 = test for get_rotation_angle()
    flag4 = False
    
    #flag5 = test for get_min_distance()
    flag5 = False
    
    #flag6: test for get_magnification_ratio()
    flag6 = False
    
    #flag7: test for calculate_dot_product()
    flag7 = False
    
    #flag8: test for isolate_drop
    flag8 = False
    
    #flag9: test ofr shift_coords
    flag9 = True
    
    #flag10: test for scale_drop
    flag10 = False
    
    #flag11: test for rotate_coords
    flag11 = False
    
    #flag12: test for reordering data
    flag12 = False
    
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
        for i in range(0,59):
            testArray.append([i,i+0.6*i])
        print testArray
        
        rotationAngle = get_rotation_angle(testArray)
        print rotationAngle
        
    #flag5 = test for get_min_distance()
    if(flag5 == True):
        lineEnds1 = [[5,1],[14,1]]
        lineEnds2 = [[5,6],[15,6]]
        minDistance = get_min_distance(lineEnds1,lineEnds2)
        print minDistance
        
    #flag6: test for get_magnification_ratio()
    if(flag6 == True):
        #set up test array
        testArray = []
        for i in range(0,59):
            testArray.append([0 - (0.001*i),i])
            testArray.append([10 + (0.001*i),i])
        print testArray
        
        testActualDiameter = 1.63
        magRatio = get_magnification_ratio(testArray,testActualDiameter)
        print magRatio
     
    #flag7: test for calculate_dot_product()
    if(flag7 == True):
        #define test variables
        testVectEndX = [0,5]
        testVectEndY = [10,6]
        testPointCoordX = 5
        testPointCoordY = 2 #change value to test
        testBool = calculate_dot_product(testVectEndX, testVectEndY, testPointCoordX, testPointCoordY)
        print testBool
   
    #flag8: test for isolate_drop     
    if(flag8 == True):
        testLineX = [200, 1000]
        testLineY = [400,400]
        img = mpimg.imread('H2O in PDMS.jpg')
        binarizedImage = binarize_image(img)
        edges = detect_boundary(binarizedImage)
        interfaceCoordinates = get_interface_coordinates(edges)
        testDropCoords = isolate_drop(testLineX,testLineY,interfaceCoordinates)
        print testDropCoords
        plt.scatter(testDropCoords[:,0],testDropCoords[:,1])
        
    #flag9: test ofr shift_coords
    if(flag9 == True):
        newCenter = [2,2]
        shiftedCoords = shift_coords(testArray[:,0],testArray[:,1],newCenter)
        print shiftedCoords
        plt.scatter(shiftedCoords[:,0],shiftedCoords[:,1])
        
    #flag10: test for scale_drop
    if(flag10 == True):
        testMagnificationRatio = 10.0
        scaledDropCoords = scale_drop(testArray, testMagnificationRatio)
        print scaledDropCoords
        plt.scatter(scaledDropCoords[:,0],scaledDropCoords[:,1])
        
    #flag11: test for rotate_coords
    if(flag11 == True):
        angle = 300 #degrees0
        finalCoords = rotate_coords(testArray, angle, 'degrees')
        print finalCoords
        plt.scatter(finalCoords[:,0],finalCoords[:,1])
        
    if flag12:
        xReal,zReal = reorder_data(testArray)
