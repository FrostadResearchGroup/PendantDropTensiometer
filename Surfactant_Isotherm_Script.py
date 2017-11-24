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
import matplotlib.pyplot as plt
import csv
from pylab import *

##############################################################################

#input parameters  
deltaRho = 998 #absolute density difference between two fluids in kg/m^3
capillaryDiameter = 2.10 #capillary diameter in mm
reloads = 1 #converge 
trueSyringeRotation = 0 #unsure of true syringe rotation
deltaT = 1 #volumetric thermal flucutations
thermalExpCoeff = 0.000214 #1/K or 1/C
folderName = 'TritonX-100/Nov22nd' # Must be located in Data folder
numMethod = 2 # 1 for 5 points (faster), 2 for all points
imageExtension = '.jpg'
numConc = 12  #total number of concentrations for isotherm
numPics = 15  #number of total droplet images
avgTemp= 24.36 #average local temperature from previous calculations in temperature.py script

##############################################################################

#function to display image on plt
   
#def get_image_time(fileName):
#    ind1 = fileName.find('t=')
#    ind2 = fileName.find('sec_')
#    time = float(fileName[ind1+2:ind2])
#    return time
   
def get_image_conc(fileName):
    ind1 = fileName.find('C=')
    ind2 = fileName.find('mM_')
    conc = float(fileName[ind1+2:ind2])
    return conc

# Parse user inputs
dirName = '../Data/' + folderName + '/' 
fileList = glob.glob(dirName + '*H2O*' + imageExtension)
saveFile = dirName + 'output.csv'
capillaryFile = glob.glob(dirName + '*Capillary*' + imageExtension)

N = len(capillaryFile)

# Allocate arrays for storing data
surfaceTenVec = np.zeros((numPics,1))
dropVolVec = np.zeros((numPics,1))
timeVec = np.zeros((numPics,1))
volErrorVec = np.zeros((numPics,1))
dropConcFile = np.zeros((numPics,1))
SurfaceTension_array=[]

stdDevSurfTenVec = np.zeros((N,1))
avgSurfaceTen = np.zeros((N,1))
concVec = np.zeros((N,1))
avgSTandSDev=[]
#avgSTandSDev= np.zeros(shape=(N,3))
##

##set up nested loop to segment images by concentration
#for i in range(N):
#    

#setting up loop for processing one capillary image per 15 pictures
#created nested loop
for i in range(N):
    capillaryImage = ie.load_image_file(capillaryFile[i])
    dropConcFile   = fileList[i*numPics:i*numPics+numPics] 
    
    for j in range(numPics):

        imageFile = dropConcFile[j]
        dropletImage = ie.load_image_file(imageFile)

#    #get time vector from file
#    timeVec[i] = get_image_time(imageFile)
         
    
        ret = dp.get_surf_tension(dropletImage, capillaryImage, deltaRho,
                                            capillaryDiameter, numMethod, 
                                            trueSyringeRotation, reloads,
                                            deltaT,thermalExpCoeff)
                        
        #returns values if not black image                                    
        if not isnan(ret[0]):
            surfaceTenVec[j] = ret[0]*10**3
            dropVolVec[j]    = ret[1]*10**9
            volErrorVec[j]   = ret[2]
#            concVec[j] = get_image_conc(imageFile)
            
        else:
            surfaceTenVec[j] = nan
            dropVolVec[j]    = nan
            volErrorVec[j]   = nan
            concVec[j]       = nan

    surfaceTenVec = surfaceTenVec[np.where(np.isfinite(surfaceTenVec))]
    dropVolVec    = dropVolVec[np.where(np.isfinite(dropVolVec))]
    volErrorVec    = volErrorVec[np.where(np.isfinite(volErrorVec))]
    #appending surface tension to pre-defined array
    SurfaceTension_array.append(surfaceTenVec)

    avgSurfaceTen[i] = np.mean(surfaceTenVec)
    concVec[i]       = get_image_conc(capillaryFile[i])
    stdDevSurfTenVec[i] = np.std(surfaceTenVec)  
    #appending strings to array
    avgSTandSDev.append(avgSurfaceTen[i])
    avgSTandSDev.append(stdDevSurfTenVec[i])

concVec  = concVec[np.where(np.isfinite(concVec))]
avgSTandSDev.append(concVec)
#converting method to name for plot title
if numMethod== 1:
    NumericalMethodName='5-Point Calculation, '
else: 
    NumericalMethodName= 'Numerical Method, '
    
# Plot surface tension vs concentration
plt.semilogx(concVec,avgSurfaceTen,'k^')
plt.title('TritonX-100 Isotherm, ' + NumericalMethodName + str(avgTemp) + ' degrees C')
plt. xlabel('Concentration (mM)')
plt.ylabel('Surface Tension, mN/m')
plt.errorbar(concVec,avgSurfaceTen,yerr=np.std(stdDevSurfTenVec),fmt=' ')

with open(saveFile, 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Concentration [mM]','Surface Tension [mN/m]'])
    for i in range(N):
        writer.writerow([concVec[i],avgSurfaceTen[i]])
    writer.writerow(['Average ST [mN/m]','Std Dev'])
    StandDev= stdDevSurfTenVec
    writer.writerow([avgSurfaceTen,StandDev])

print 'Average = %0.2f +/- %0.02f mN/m' %(np.mean(surfaceTenVec),np.std(surfaceTenVec))
average_surface_tension= np.mean(surfaceTenVec)
standard_deviation= np.std(surfaceTenVec)       