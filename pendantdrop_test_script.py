# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:26:32 2017
@author: Yohan
"""

#import code blocks
import image_extraction as ie
import data_processing as dp

#import other modules
import numpy as np
import glob
import cv2

##############################################################################

#input parameters  
deltaRho = 998 #absolute density difference between two fluids in g/L
capillaryDiameter = 2.108 #capillary diameter in mm
reloads = 1 #converge 
trueSyringeRotation = 0 #unsure of true syringe rotation
folderName = 'Code Testing' # Must be located in Data folder
numMethod = 1 # 1 for 5 points (faster), 2 for all points

##############################################################################

#function to display image on plt
   
def get_image_time(fileName):
    ind1 = fileName.find('t=')
    ind2 = fileName.find('sec_')
    time = float(fileName[ind1+2:ind2])
    return time

# Parse user inputs
dirName = '../Data/' + folderName + '/' 
fileList = glob.glob(dirName + '*0*.jpg')
capillaryImage = ie.load_image_file(dirName + 'CapillaryImage.jpg')
N = len(fileList)

# Allocate arrays for storing data
surfaceTenVec = np.zeros((N,1))
dropVolVec = np.zeros((N,1))
timeVec = np.zeros((N,1))

for i in range(N):
    imageFile = fileList[i]
    dropletImage = ie.load_image_file(imageFile)     
    
    #get time vector from file
    timeVec[i] = get_image_time(imageFile)
    ret = dp.get_surf_tension(dropletImage, capillaryImage, deltaRho,
                                        capillaryDiameter, numMethod, 
                                        trueSyringeRotation, reloads)
    surfaceTenVec[i] = ret[0]
    dropVolVec[i] = ret[1]

                
#if numMethod == 1:
#    stats = np.vstack([timeVec,surfaceTenVec,bondNumberVec,dropVolVec,apexRadiusVec]).transpose()
#    headername = ["Surface Tension (N/m)", "Bond Number", "Relative Droplet Volume", "Apex Radius of Curvature (m)"]                
#else:
#    stats = np.vstack([timeVec,surfaceTenVec]).transpose()
#    headername = ["Time (s)","Surface Tension (N/m)"]
#    
#outputFileDir = dirName.encode('ascii','ignore')+'/'+'stats.csv'
#np.savetxt(outputFileDir, stats, delimiter=",",header = headername)



        
