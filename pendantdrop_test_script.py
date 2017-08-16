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
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import os


##############################################################################

#input parameters  
deltaRho = 998          #absolute density difference between two fluids in g/L
actualDiameter = 2.108  #capillary diameter in mm
reloads = 1             #converge 

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
display_scat(interfaceCoordinates)
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
print ("shifted apex NEAR 0,0")

#scale drop
scaledCoords = ip.scale_drop(shiftedCoords,magnificationRatio)
print ("scaled coordinates according to magnification ratio")

#reorder data points
xData,zData = ip.reorder_data(scaledCoords)
s,xe,apexRadiusGuess = dp.bond_calc(xData,zData)
bondInit = dp.s_interp(s,xe)

#state guesses for apex shift
horizShiftGuess = 0
vertShiftGuess  = 0 

#run through optimization routine
surfTen,apexRadius,thetaRotation,bondNumber,horizTranslation,vertTranslation = dp.optimize_params(xData,zData,bondInit,
                                                      apexRadiusGuess,horizShiftGuess,
                                                      vertShiftGuess,deltaRho,reloads)


#output surface tension
print "Bond Number = %.4g" %(bondNumber)
print "Surface Tension = %.4g mN/m" %(surfTen*10**3)


#output error surface plot
#x,y,z = dp.get_response_surf(surfTen,apexRadius,thetaRotation,deltaRho,xData,zData,dp.objective_fun_v2)
#ax = Axes3D(plt.figure())
#ax.plot_surface(x,y,np.log10(z))

######################## For Testing Purposes ################################# 
   
if __name__ == "__main__":
    
    testProfiler = False
    
    
    if testProfiler:    
    
        #import profiler packages
        import cProfile
        import pstats
        
        cProfile.run('dp.optimize_params(xData,zData,bondInit,apexRadiusGuess,horizShiftGuess,vertShiftGuess,deltaRho,reloads)','timeStats')
        p = pstats.Stats('timeStats')
        
        #arrange output by largest culmulative tie
        p.sort_stats('time').print_stats(10)
