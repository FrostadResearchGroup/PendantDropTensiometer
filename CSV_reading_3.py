# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 17:36:37 2018

@author: frostad
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:50:31 2018

@author: Joanna
"""

##CSV Files reading version 2
###import of relevant modules
import numpy as np
import csv
import glob as glob
import matplotlib.pyplot as plt
import scipy as sci
from scipy import optimize
import os

#from tempfile import TemporaryFile

##################################################USER-DEFINED INPUTS###############################################
"""
This is a working version of script to get data plots in the meantime to writing a more refined version of data
management, plotting, and modular functions to that end.
Use this script to get initial data plots for soluble surfactant equilibrium surface tension isotherms obtinaed from measurements using
the Wilhelmy Rod with Langmuir Trough force tensiometer.
Based on the user-defined inputs in the subsequent section, you must indicate what data plot is desired and manually change the inputs.
For the multple runs on one plot, you must manually change the plot indicators (like 'ro', '--g') in between running the code to
differentiate the data points from different runs.
"""
#USER-DEFINED INPUTS
#filepath = 'C:\Research\Foam Density Joanna\Data\HTAB 16Feb2018./'
filepath = '..\Data\SDS 12March2018\\'
fileString = '*mM*.ntb'
outPath= '..\Data'
outFolder= '\Feb3rd_outfile./'  #manually create folder in Python outfiles folder- savig avg ST value arrays
startInd = 0
csv_reading= True
print_np_file= False
plot_Gibbs= False #decision to plot fitted Gibbs adsorption equation
multiple_plots= True #plotting one isotherm or multiple isotherms
TempAvg= np.mean([24.7,25.58, 25.53,25.59])   #manually input average temperatures from previous runs
SurfactantName= 'SDS'
plot_type= 1                          #change based on which run being plotted- values are 1, 2, or 3 for red circles, green triangles, or blue squares
surfactant_title_single= SurfactantName + ' isotherm,  '
surfactant_title_multiple= SurfactantName + ' isotherms '  + 'average temp = ' + np.array2string(TempAvg) + ' C'
###################################################################################################################
#constants
R= 8.314 #ideal gas constant in J/(mol*K)




def read_NTB(fileName):
    
    headerLines = 40
#    totArea = []
#    area = [] 
    tension = []
    time = []
    temp = []
    
    with open(fileName,'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter= ';', quotechar='|')
        
        # Parse through header lines
        for i in range(headerLines):
            row = next(reader)
            if i in [1,2,3,5]:
                print row
        # Read and store the date
        for row in reader:
#            totArea.append(row[1])
#            area.append(row[2])
            tension.append(row[6])
            time.append(row[8])
            temp.append(row[9])
            
    time = np.array(time)
    tension = np.array(tension)
    tension= tension.astype(float)
    print type(tension)
    T = np.array(temp)
#    rownum= 180
    Stvalues= tension[startInd:]
    AvgST=np.mean(Stvalues)
    Stdev= np.std(Stvalues)
#    for row in range(len(tension)):
#        row= rownum
#        print tension[row]
#        AvgST.append(tension[rownum])
#        row=row+1
#    AvgST=np.array(AvgST)   
#    area = np.array(area)
#    totArea - np.array(totArea)
    print('\n---------------------------------------------------------\n')    
    
    
    return time, tension, T, AvgST,Stdev

    
def get_conc(fileName):
    truncate = fileName[:-4]
    ind1 = truncate.find('c=')
    ind2 = truncate.find('mM')
    conc = truncate[ind1+2:ind2]
    conc = float(conc.replace(',','.'))
    return conc
    
def get_conc2(fileName):
    temp = os.path.basename(fileName)
    temp = temp[0:6]
    temp = temp.replace(',','.')
    conc = float(temp)
    return conc

def Gibbs(C,B,a):
    T=299
    y= 2*R*T*B*(np.log(1+(C/a)))
    return y
      
#plt.close('all')
fileList = glob.glob(filepath + fileString)
print fileList
N = len(fileList)
concNumber = N

#allocate lists/array for storing data
csv_files_reading= []
info= []
data_file= [] 
data_file_1= []
concentration= np.zeros((concNumber,))
time_data=np.zeros((concNumber,1))
tension_data=np.zeros((concNumber,1))
T_data= np.zeros((concNumber,1))
AvgSTvec= np.zeros_like(concentration)
STavg=[]
Stdev=[]



for i in range(len(fileList)):

    fileName = fileList[i]
    concentration[i]= get_conc(fileName)
    time, tension, T , avgST, Std = read_NTB(fileName)
    
#    plt.figure()
#    plt.plot(time,T,'.',time,tension,'o')
    time_data[i]= time[i] 
    #print time_data
    tension_data[i]=tension[i]
    #print tension_data
    T_data[i]=T[i]
    AvgTemp= np.mean(T_data)
    print ("%.2f" % round(AvgTemp,2))
    
    #print T_data
#    AvgSTvec[i]=AvgSTvec.append(avgST)
#    for i in range(len(AvgSTve)):
#        np.append(AvgSTvec[i],axis=0)
    STavg.append(avgST)
    #print STavg
    Stdev.append(Std)
#    STavg= np.array(STavg)
#plotting avg surface tension values
STavg= np.array(STavg)
Stdev= np.array(Stdev)
#plotting decision for single or multiple isotherms
if multiple_plots==False:
    plt.figure()
    plt.semilogx(concentration,STavg,'ro')  
    plt.title(surfactant_title_single + ' average temp (C) ' + np.array2string(AvgTemp))
    plt.xlabel('Concentration (mM)')
    plt.ylabel('Surface Tension (mN/m)') 
    plt.errorbar(concentration,STavg,yerr= Stdev, fmt= ' ') 
    plt.draw()
elif plot_type==1:
    plt.semilogx(concentration,STavg, 'ro') 
    plt.title(surfactant_title_multiple)
    plt.xlabel('Concentration (mM)')
    plt.ylabel('Surface Tension (mN/m)') 
    plt.errorbar(concentration,STavg,yerr= Stdev, fmt= ' ')
elif plot_type==2:
    plt.semilogx(concentration,STavg, 'g^') 
    plt.title(surfactant_title_multiple)
    plt.xlabel('Concentration (mM)')
    plt.ylabel('Surface Tension (mN/m)') 
    plt.errorbar(concentration,STavg,yerr= Stdev, fmt= ' ')
else:
    plt.semilogx(concentration,STavg, 'bs') 
    plt.title(surfactant_title_multiple)
    plt.xlabel('Concentration (mM)')
    plt.ylabel('Surface Tension (mN/m)') 
    plt.errorbar(concentration,STavg,yerr= Stdev, fmt= ' ')
    
    
##saving current avg ST vector    
if print_np_file:
    
    outfile = outPath + outFolder 
    np.savez(outfile, AvgSTvec=AvgSTvec, Temp=T_data)

if plot_Gibbs==True:   
##calculating fitted curve for Gibbs Equation
    pi= 72.8-STavg
#conc= concentration.resize((20,))
    plot_1,plot_2=sci.optimize.curve_fit(Gibbs,concentration,pi)
    print plot_1,plot_2

#for s in range(len(STavg)):
#    Y=STavg[s]
#    plot_1,plot_2=sci.optimize.curve_fit(Gibbs,concentration,STavg)
    plt.figure()
    plt.semilogx(concentration,pi, 'b-', label='surface pressure (reference value: sigma_0 = 72.2 mN/m)')
    plt.semilogx(concentration,Gibbs(concentration,*plot_1),'g----',label='Gibbs Adsorption Equation, where B=%5.3f,a=%5.3f' % tuple(plot_1))
    plt.title(surfactant_title_single, fontsize= 20)
    plt.xlabel('Concentration (mM)', fontsize=20)
    plt.ylabel('Surface Pressure, (sigma_0-sigma, mN/m)',fontsize=20)
    plt.legend(loc='best', fontsize=15)
    plt.draw()
    plt.show()
