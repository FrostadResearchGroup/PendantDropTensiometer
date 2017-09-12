# -*- coding: utf-8 -*-
"""
Created on Mon May 29 10:03:52 2017

@author: Yohan
"""

from skimage import feature
import cv2
import numpy as np
import data_processing as dp

def binarize_image(image):
    """
    Binarize image to convert the image to purely black and white image.
    
    image = JPEG - image file of droplet 
    """
    #use Otsu's threshold after Gaussian filtering (Otsu's binarization)
    #filter image with 5x5 Gaussian kernel to remove noise
    
    #rotate image 90 degrees
    image = image.transpose()  
    
    blur = cv2.GaussianBlur(image,(5,5),0)
    ret3,binaryImage = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    
    return binaryImage

def detect_boundary(binaryImage):
    """
    Detects the outline of the binary image and outputs in boolean format.
    
    binaryImage = uint8 - black and white image of droplet
    """
    #use canny edge detection algorithm to find edges of image
    #edge dectection operator goes line-by-line horizontally
    edges = feature.canny(binaryImage)

#    edges = np.array(edges,dtype=np.uint8)
#    edges, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
#    #edges = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    return edges
    
def get_interface_coordinates(edges):
    """
    Goes through each pixel in the edges image to extract the coordinates of 
    the edges.
    
    edges = boolean - edge profile of droplet 
    """
    #creates array of edge coordinates with respect to pixel length
    interfaceCoords = []
    for y in range(0,edges.shape[0]):
        edgeCoords = []
        for x in range(0, edges.shape[1]): 
            if edges[y, x] != 0:
                edgeCoords = edgeCoords + [[x, y]]       
        #takes in only the outer most points
        if(len(edgeCoords)>=2):
            truthIndices = np.argwhere(edges)
            apexIndex = max(truthIndices[:,0])
            if y != apexIndex:    
                interfaceCoords = interfaceCoords + [edgeCoords[0]] + [edgeCoords[-1]]
            else:
                for i in range(len(edgeCoords)):
                    interfaceCoords = interfaceCoords + [edgeCoords[i]]
        if(len(edgeCoords)==1):
            interfaceCoords = interfaceCoords + [edgeCoords[0]]      
    interfaceCoords = np.array(interfaceCoords)

#    interfaceCoords, hierarchy = cv2.findContours(interfaceCoords, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)    

    return interfaceCoords

def get_translation_calibration(interfaceCoords,dropCoords):
    """
    Translates the interface and isolated droplet coordinates to same relative position
    
    """
    
    vertTranslation = min(interfaceCoords[:,1]) - min(dropCoords[:,1])   
    horzTranslation = (max(interfaceCoords[:,0]) - min(interfaceCoords[:,0]))/2 + (max(dropCoords[:,0]) - min(dropCoords[:,0]))/2
    return vertTranslation,horzTranslation

    
def get_capillary_diameter(line1, line2):
    """
    Defines capillary diameter in pixel length.
    
    line1 = int32 - first point on left edge of capillary
    line2 = int32 - second point on right edge of capillary
    """     
    #express first points on opposite side of capillary as x,z coordinates
    L1x,L1y = line1
    L2x,L2y = line2
    
    #find straight line distance between points
    dist = ((L2x-L1x)**2+(L2y-L1y)**2)**0.5

    #Assumption: rotation of image is very small such that the scaled straight
    #line distance is very close to true diameter of the capillary 
    return dist
    
def get_magnification_ratio(interfaceCoords, actualDiameter):
    """
    Determines magnification ratio through a comparison of capillary diameter.
    
    interfaceCoords = int32 - array of edge coordinates (x,z) in pixel length
    actualDiameter = float - actual outer diameter of capillary (user input)
    """
    
    #find first points on capillary edge  
    #assume that next point on same capillary edge is within a one pixel offset 
    #in x-direction - recall horizontal line-by-line output from canny edge dection 
    if interfaceCoords[0,0]-1 <= interfaceCoords[1,0] <= interfaceCoords[0,0]+1:
        lineCoords = interfaceCoords[0]
        adjLineCoords = interfaceCoords[2]
    else:
        lineCoords = interfaceCoords[0]
        adjLineCoords = interfaceCoords[1]
        
    #calculate line distance and magnification ratio
    lineDistance = get_capillary_diameter(lineCoords,adjLineCoords)
    magnificationRatio = actualDiameter/lineDistance
    
    return magnificationRatio
    
def calculate_dot_product(vectEndX, vectEndZ, pointCoordX, pointCoordZ):
    """
    Looks if the linePosition is above or below user defined point for 
    isolating drop.
    
    vectEndX = list - total x range (in pixels) of droplet
    vectEndZ = list - z-coordinate (in pixels) of user input for isolating droplet
    pointCoordX = int32 - x-coordinates of droplet
    pointCoordZ = int32 - z-coordinates of droplet 
    """
    
    #loop that returns boolean expressions depending on whether coordinate is 
    #below the line
    xp = (vectEndX[1]-vectEndX[0])*(pointCoordZ-vectEndZ[0])- \
                    (vectEndZ[1]-vectEndZ[0])*(pointCoordX-vectEndX[0])
    if xp < 0:
        return True
    else:
        return False
        
#def isolate_drop(lineCoordX, lineCoordZ, interfaceCoords,xBot,yBot):
#    """
#    Checks whether point is above or below line using the calculate_cross_product 
#    function to see if linePosition is: above (True) --> keep looping, else is 
#    below (False) --> stop looping.
#    
#    lineCoordX = list - total x range (in pixels) of droplet
#    lineCoordZ = list - z-coordinate (in pixels) of user input for isolating droplet
#    """
#    linePosition = True
#    cutoffPoint = None    
#    
#    #rotate image
#    xTop,yTop = interfaceCoords[0]
#    theta = get_rotate_coords(xTop,yTop,xBot,yBot)
#
#    print(theta)
#    xRotated = interfaceCoords[:,0]*np.cos(theta) - interfaceCoords[:,1]*np.sin(theta)
#    zRotated = interfaceCoords[:,0]*np.sin(theta) + interfaceCoords[:,1]*np.cos(theta)
#    
#    interfaceCoords = np.column_stack((xRotated,zRotated))
#    
#    #loop to find the outline coordinate where the outline coordinate is below the line
#    for i in range (0, len(interfaceCoords)):
#        linePosition = calculate_dot_product(lineCoordX, lineCoordZ,
#                                    interfaceCoords[i,0], interfaceCoords[i,1])
#        if linePosition is False:
#            cutoffPoint = i
#            break
#    
#    #self.dropcoord will be the coordinates used from now on
#    dropCoords = interfaceCoords
#    
#    #when it is found, remove all coordinates above that y-coordinate
#    j = 0
#    while j < cutoffPoint:
#        dropCoords = np.delete(dropCoords,0,0)
#        j += 1
#
#    return dropCoords

def shift_coords(xCoords, zCoords, newCenter):
    """ 
    Shift the coordinates so that the orgin is at the specified center.
    
    xCoords = int32 - x-coordinates of droplet pre-shift
    zCoords = int32 - z-coordinates of droplet pre-shift
    newCenter = list - desired new center (defined as (0,0))
    """
    
    #determine offset of current droplet center
    xOffset =  (max(xCoords) + min(xCoords))/2
    zOffset =   zCoords[-1]
    oldCenter = [xOffset,zOffset]
    
    #centers the drop
    xDifference = oldCenter[0] - newCenter[0]
    zDifference = oldCenter[1] - newCenter[1]
    xCoords -= xDifference
    zCoords -= zDifference
    
    #artifically rotate data (FOR TESTING PURPOSES)
#    xCoords = xCoords*np.cos(-5*2*np.pi/360) - zCoords*np.sin(-5*2*np.pi/360)
#    zCoords = xCoords*np.sin(-5*2*np.pi/360) + zCoords*np.cos(-5*2*np.pi/360)  
    
    coords = np.append([xCoords],[zCoords],axis=0).transpose()

    #flip the coordinates vertically
    coords *= [1,-1]
    
    return coords
    
def scale_drop(coords, magnificationRatio):
    """ 
    Scales the coordinate based on the magnification ratio.
    
    coords = ndarray (N,i) where i is the dimensionality (i.e 2D)
    magnificationRatio = float - pixel to meters scaling conversion 
    """
    #changing units to meters
    scaledCoords = coords * [magnificationRatio*10**-3, 
                             magnificationRatio*10**-3] 
    return scaledCoords
    
def reorder_data(coords):
    """
    Re-orders data into 2-D array that smoothly follows the drop profile.
    
    coords = ndarray (N,i) where i is the dimensionality (i.e 2D)
    """
    
    #sort in ascending z-value order
    coords = coords[coords[:,1].argsort()]
    
    xData = coords[:,0]
    zData = coords[:,1]        
    
    xDataRight = ()
    zDataRight = ()
    xDataLeft = ()
    zDataLeft = ()
    
    #seperate left and right side of droplet by positive or negative x-coord 
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

                       
    #reorders data in descending z order
    indexLeft = np.lexsort((xDataLeft,-1*zDataLeft))    
    indexRight = np.lexsort((xDataRight,zDataRight))    

    xDataLeft = xDataLeft[indexLeft]
    xDataRight = xDataRight[indexRight]
    zDataLeft = zDataLeft[indexLeft]
    zDataRight = zDataRight[indexRight]
    
    #append left and right side coordinates
    xCoords = np.append(xDataLeft,xDataRight[1:])
    zCoords = np.append(zDataLeft,zDataRight[1:])

    return xCoords,zCoords         


######################## Section for Laurie ##################################

def get_true_vertical(slopeEdges):
    """
    Finds the true vertical reference (gravity) through obtaining the slope of a
    vertically hanging string attached to a weight.
    
    slopeEdges - 2d array: 
    
    """
    
    return trueVerticalSlope
    
def get_syringe_rotation(capillaryEdges):
    """
    Finds the rotation angle of the syringe based off of the true vertical.
    """
    
    return dropletRotationAngle
    
def get_rotated_droplet(dropletCoords,trueVerticalSlope,dropletRotationAngle):
    """
    Calibrates the droplet coordinates based off the rotation angle of the syringe
    with respect to true vertical.
    """

    return rotatedDropletCoords

def get_droplet_isloation(rotatedDropletCoords,capillaryEdges):
    """
    Isolates the droplet from the capillary portion of the edge profile.
    """
    
    return isolatedDropletCoords

               
######################## For Testing Purposes ################################# 
   
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
    flag2 = True
    
    #flag3 = test for get_interface_coordinates()
    flag3 = False
   
    #flag5 = test for get_min_distance()
    flag5 = False
    
    #flag6: test for get_magnification_ratio()
    flag6 = False
    
    #flag7: test for calculate_dot_product()
    flag7 = False
    
    #flag8: test for isolate_drop
    flag8 = False
    
    #flag9: test ofr shift_coords
    flag9 = False
    
    #flag10: test for scale_drop
    flag10 = False
    

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
        edges,hierarchy = detect_boundary(binarizedImage)
        newEdges = []
        
        for i in range(len(edges)):
            if i == 0:
                newEdges = np.reshape(edges[i],(-1,2))
            else:
                newEdges = np.vstack((newEdges,(np.reshape(edges[i],(-1,2)))))
        
        plt.scatter(newEdges[:,0],newEdges[:,1])
        #y=cv2.drawContours(img,edges,-1,(0, 255, 0),-1)
        #cv2.imshow("window title", img)
        
        
    #flag3 = test for get_interface_coordinates()       
    if(flag3 == True):
        img = mpimg.imread('H2O in PDMS.jpg')
        binarizedImage = binarize_image(img)
        edges = detect_boundary(binarizedImage)
        interfaceCoordinates = get_interface_coordinates(edges)
        print interfaceCoordinates
        plt.scatter(interfaceCoordinates[:,0],interfaceCoordinates[:,1])
           
    #flag5 = test for get_min_distance()
    if(flag5 == True):
        lineEnds1 = [471,1]
        lineEnds2 = [656,1]
        minDistance = get_capillary_diameter(lineEnds1,lineEnds2)
        print minDistance
        
    #flag6: test for get_magnification_ratio()
    if(flag6 == True):
        #set up test array
        testArray = []
        for i in range(0,59):
            testArray.append([0 - (1*i),i])
            testArray.append([10 + (1*i),i])
        #print testArray
        testArray = np.array(testArray)
        
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
        testBool = calculate_dot_product(testVectEndX, testVectEndY, 
                                         testPointCoordX, testPointCoordY)
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
        

    if flag12:
        xReal,zReal = reorder_data(testArray)
