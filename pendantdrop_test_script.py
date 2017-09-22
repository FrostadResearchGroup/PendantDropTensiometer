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
deltaRho = 998 #absolute density difference between two fluids in g/L
actualDiameter = 2.108 #capillary diameter in mm
reloads = 1 #converge 
#unsure of true syringe rotation
trueSyringeRotation = 0

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

#user input to specify directory     
dirName = tkFileDialog.askdirectory()

for root,dirs,files in os.walk(dirName):

    capImage = files[0]
    capFullDirectory  = dirName+'/'+capImage
    capFullDirectory = capFullDirectory.encode('ascii','ignore')
    capillaryImage = ie.load_image_file(capFullDirectory)     
    
    capillaryRotation,xCoords,zCoords = ip.get_capillary_rotation(capillaryImage)

    
    surfaceTenVec = ()
    apexRadiusVec = ()
    bondNumberVec = ()
    changeDropVolVec = ()    
    
    mode = input('Is this an image or time lapse? Enter 1 for image, 2 for time lapse: ')
    numMethod = input('Algebraic or Iterative solution? Enter 1 for algebraic, 2 for iterative: ')
    
    #mode for analyszing image    
    if mode == 1:
        dropImage = files[1]
        dropFullDirectory  = dirName+'/'+dropImage
        dropFullDirectory = dropFullDirectory.encode('ascii','ignore')
        dropletImage = ie.load_image_file(dropFullDirectory) 
        
        #binarize image
        binarizedDropletImage = ip.binarize_image(dropletImage)
        display_plot(binarizedDropletImage)
        print("image binarized")
        
        #detect image boundary
        dropletEdges = ip.detect_boundary(binarizedDropletImage)
        display_plot(dropletEdges)
        print("boundary traced")
        
        #get interface coordinates
        interfaceCoordinates = ip.get_interface_coordinates(dropletEdges)
        plt.cla()
        plt.scatter(interfaceCoordinates[:,0],interfaceCoordinates[:,1])
        print("boundary coordinates acquired")
        
        #isolate drop
        xCoords = [min(interfaceCoordinates[:,0]),max(interfaceCoordinates[:,0])] 
        dropCoords = ip.isolate_drop(xCoords,zCoords,interfaceCoordinates)
        print ("drop isolated")
        
        #get magnification ratio
        magnificationRatio = ip.get_magnification_ratio(dropCoords,actualDiameter)
        print ("magnification ratio is " + str(magnificationRatio))
        
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
        surfTenInit = dp.s_interp(s,xe,deltaRho)
        if numMethod == 2:
            #run through optimization routine
            surfTen,apexRadius,bondNumber,dropVol = dp.optimize_params(xData,zData,surfTenInit,apexRadiusGuess,deltaRho,reloads,trueSyringeRotation)
            
            #output surface tension
            print "Bond Number = %.4g" %(bondNumber)
            print "Surface Tension = %.4g mN/m" %(surfTen*10**3)
        elif numMethod == 1:
            print "Surface Tension = %.4g mN/m" %(surfTenInit*10**3)
        else:
            print('Invalid Mode Selected! Re-Run Script and Enter 1 or 2')
    #mode for analyszing time lapse       
    elif mode == 2:
        for i in range(len(files)-2):
            if i != len(files)-1:
                
                dropImage = files[i+1]
                dropFullDirectory  = dirName+'/'+dropImage
                dropFullDirectory = dropFullDirectory.encode('ascii','ignore')
                dropletImage = ie.load_image_file(dropFullDirectory) 
                
                #binarize image
                binarizedDropletImage = ip.binarize_image(dropletImage)
                display_plot(binarizedDropletImage)
                print("image binarized")
                
                #detect image boundary
                dropletEdges = ip.detect_boundary(binarizedDropletImage)
                display_plot(dropletEdges)
                print("boundary traced")
                
                #get interface coordinates
                interfaceCoordinates = ip.get_interface_coordinates(dropletEdges)
                plt.cla()
                plt.scatter(interfaceCoordinates[:,0],interfaceCoordinates[:,1])
                print("boundary coordinates acquired")
                
                #isolate drop
                xCoords = [min(interfaceCoordinates[:,0]),max(interfaceCoordinates[:,0])] 
                dropCoords = ip.isolate_drop(xCoords,zCoords,interfaceCoordinates)
                print ("drop isolated")
                
                #get magnification ratio
                magnificationRatio = ip.get_magnification_ratio(dropCoords,actualDiameter)
                print ("magnification ratio is " + str(magnificationRatio))
                
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
                surfTenInit = dp.s_interp(s,xe,deltaRho)
                if numMethod == 2:
                    #run through optimization routine
                    surfTen,apexRadius,bondNumber,dropVol = dp.optimize_params(xData,zData,surfTenInit,apexRadiusGuess,deltaRho,reloads,trueSyringeRotation)
                    
                    #express dropletVolume in comparison to initial 
                    if i == 0:
                        initialDropVol = dropVol
                        changeDropVol  = 1
                    else:
                        changeDropVol = dropVol/initialDropVol
                    #output surface tension
                    print "Bond Number = %.4g" %(bondNumber)
                    print "Surface Tension = %.4g mN/m" %(surfTen*10**3)
                    
                    #output in vector format
                    surfaceTenVec = np.append(surfaceTenVec,surfTen)
                    apexRadiusVec = np.append(apexRadiusVec,apexRadius)
                    bondNumberVec = np.append(bondNumberVec,bondNumber)
                    changeDropVolVec = np.append(changeDropVolVec,changeDropVol)
                    
                elif numMethod == 1:
                    print "Surface Tension = %.4g mN/m" %(surfTenInit*10**3)
                    #output in vector format
                    surfaceTenVec = np.append(surfaceTenVec,surfTenInit)
                else:
                    print('Invalid Mode Selected! Re-Run Script and Enter 1 or 2')

 
    else:
        print('Invalid Mode Selected! Re-Run Script and Enter 1 or 2')
        
