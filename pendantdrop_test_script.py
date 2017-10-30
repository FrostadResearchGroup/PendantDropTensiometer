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
folderName = 'Environmental Testing/Triton x-100 0.08 mM/Cuvette + Insolation, 10 min, 0.5 Hz' # Must be located in Data folder
numMethod = 1 # 1 for 5 points (faster), 2 for all points
imageExtension = '.jpg'

##############################################################################

#function to display image on plt
   
#def get_image_time(fileName):
#    ind1 = fileName.find('t=')
#    ind2 = fileName.find('sec_')
#    time = float(fileName[ind1+2:ind2])
#    return time
   
#def get_image_conc(fileName):
#    ind1 = fileName.find('C=')
#    ind2 = fileName.find('mM_')
#    conc = float(fileName[ind1+2:ind2])
#    return conc

# Parse user inputs
dirName = '../Data/' + folderName + '/' 
fileList = glob.glob(dirName + '*Test*' + imageExtension)
saveFile = dirName + 'output.csv'
capillaryFile = glob.glob(dirName + '*CapillaryImage*' + imageExtension)
capillaryImage = ie.load_image_file(capillaryFile[0])
N = len(fileList)-1

# Allocate arrays for storing data
surfaceTenVec = np.zeros((N,1))
dropVolVec = np.zeros((N,1))
volErrorVec = np.zeros((N,1))
timeVec = np.zeros((N,1))

# Read time vector
timeData = glob.glob(dirName + '*time*' + '.csv') 
with open(timeData[0], 'rb') as timeFile:
    timeLapse  = csv.reader(timeFile)    
    timeLapse  = list(timeLapse)    
#    for i in range(N):
#        timeVec[i] = np.array(timeLapse[i][0])



for i in range(N):
    imageFile = fileList[i]
    dropletImage = ie.load_image_file(imageFile)     
    
    ret = dp.get_surf_tension(dropletImage, capillaryImage, deltaRho,
                                        capillaryDiameter, numMethod, 
                                        trueSyringeRotation, reloads,
                                        deltaT,thermalExpCoeff)
                    
    #returns values if not black image                                    
    if not isnan(ret[0]):
        surfaceTenVec[i] = ret[0]*10**3
        dropVolVec[i]    = ret[1]*10**9
        volErrorVec[i]   = ret[2]
        timeVec[i]       = timeLapse[0][i]
        
    else:
        surfaceTenVec[i] = nan
        dropVolVec[i]    = nan
        volErrorVec[i]   = nan
        timeVec[i]       = nan

surfaceTenVec = surfaceTenVec[np.where(np.isfinite(surfaceTenVec))]
dropVolVec    = dropVolVec[np.where(np.isfinite(dropVolVec))]
volErrorVec    = volErrorVec[np.where(np.isfinite(volErrorVec))]
timeVec       = timeVec[np.where(np.isfinite(timeVec))]

avgSurfaceTen = np.average(surfaceTenVec)
relDropVolVec = dropVolVec/dropVolVec[0]
relVolErrorVec = volErrorVec/dropVolVec[0]  
                
#if numMethod == 1:
#    stats = np.vstack([timeVec,surfaceTenVec,bondNumberVec,dropVolVec,apexRadiusVec]).transpose()
#    headername = ["Surface Tension (N/m)", "Bond Number", "Relative Droplet Volume", "Apex Radius of Curvature (m)"]                
#else:
#    stats = np.vstack([timeVec,surfaceTenVec]).transpose()
#    headername = ["Time (s)","Surface Tension (N/m)"]
#    
#outputFileDir = dirName.encode('ascii','ignore')+'/'+'stats.csv'
#np.savetxt(outputFileDir, stats, delimiter=",",header = headername)

# Plot surface tension vs. time
plt.figure()
plt.plot(timeVec,surfaceTenVec,'ko')
plt.xlabel('Time (s)')
plt.ylabel('Surface Tension (mN/m)')
plt.title('Surface Tension vs. Time')

# Plot drop volume vs. time
plt.figure()
plt.plot(timeVec,dropVolVec,'ko')
plt.xlabel('Time (s)')
plt.ylabel('Drop Volume (mm$^3$)')
plt.title('Drop Volume vs. Time')
plt.errorbar(timeVec,dropVolVec,yerr=volErrorVec,fmt=' ')


# Plot drop volume vs. time
plt.figure()
plt.plot(timeVec,relDropVolVec,'ko')
plt.xlabel('Time (s)')
plt.ylabel('V/$V_0$')
plt.title('Relative Drop Volume vs. Time')
plt.errorbar(timeVec,relDropVolVec,yerr=relVolErrorVec,fmt=' ')


#with open(saveFile, 'wb') as csvfile:
#    writer = csv.writer(csvfile)
#    writer.writerow(['Time [s]','Surface Tension [mN/m]','Drop Volume [mm^3]',
#                                                     'Relative Drop Volume'])
#    for i in range(len(timeVec)):
#        writer.writerow([timeVec[i],surfaceTenVec[i],dropVolVec[i],relDropVolVec[i]])
#    writer.writerow(['Average ST [mN/m]','Std Dev'])
#    StandDev= np.std(surfaceTenVec)
#    writer.writerow([avgSurfaceTen,StandDev])

#print 'Average = %0.2f +/- %0.02f mN/m' %(np.mean(surfaceTenVec),np.std(surfaceTenVec))
        
