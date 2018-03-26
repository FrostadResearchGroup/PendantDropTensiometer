# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 14:36:50 2018

@author: jlapucha
"""
"""
This file is a working version for the curve-fitting of adsorption isotherm models to the 
eqiulibrum surface tension change with concentrations.
"""
import numpy as np
import csv
import glob as glob
import matplotlib.pyplot as plt
import scipy as sci
from scipy import optimize
from scipy import odr
import os


#USER-DEFINED INPUTS
#filepath = 'C:\Research\Foam Density Joanna\Data\HTAB 16Feb2018./'
filepath = '..\Data\Curve Fitting\SDS 12March2018\\'   #change file folder name according to surfactant name
fileString = '*mM*.ntb'
startInd = 0
csv_reading= True
curve= 'Szyszkowski'
adsorption= 'Langmuir_ionic' #choices are Langmuir, Langmuir_ionic, Frumkin
Ind=0 #cell number of array containing reference surface tension value sigma_0 for calculating surface pressure
#print_np_file= False
#plot_Gibbs= False #decision to plot fitted Gibbs adsorption equation
#multiple_plots= True #plotting one isotherm or multiple isotherms
SurfactantName= 'SDS run 2'     
concentrations_new= np.linspace(1E-2,100,200) #change based on which concentration range needed to predict surfactant adsorption behavior                
#TempAvg= np.mean([27.00, 26.00])   #manually input average temperatures from previous runs
curve_fit= 'Frumkin equation of state'  
surfactant_title_Szysz= curve + ' equation for  ' + SurfactantName 
surfactant_title_single= SurfactantName + ' isotherm fitted to  ' + curve_fit
#surfactant_title_multiple= SurfactantName + ' isotherms '   ### + 'average temp = ' + np.array2string(TempAvg) + ' C'
###################################################################################################################

startInd = 0

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

def Szyszkowski(C,B,a):
    t=AvgTemp+273
    y= R*t*B*(np.log(1+(C/a)))
    return y


def Langmuir(C,B,a,ionic):
    if ionic==False:
        y=(C*B)/(a+C)
    else:
        y=(C*B)/2*(a+C)
    return y 



def Frumkin_state(G,B):
    t=AvgTemp +273
    y=-1*R*t*(np.log(1-(G/B)))
    return y

def Frumkin(C,G,B,a,A):
    G=B*C*np.exp(A*(G/B))/(a+C*np.exp(A*(G/B)))
    return G


fileList = glob.glob(filepath + fileString)
print fileList
N = len(fileList)
concNumber = N


#allocate lists/array for storing data
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
print STavg
print Stdev

StartPi=STavg[Ind]
pi= StartPi-STavg
T_Kelvin= AvgTemp+273

#conc= concentration.resize((20,))
#plot_1,plot_2=sci.optimize.curve_fit(Gibbs_ionic,concentration,pi)
#print plot_1,plot_2

titleFont = {'family': 'Tahoma','weight': 'bold','size': 40}
axesFont = {'weight': 'bold','size': 25}
size= 12

#for s in range(len(STavg)):
#    Y=STavg[s]
#    plot_1,plot_2=sci.optimize.curve_fit(Gibbs,concentration,STavg)
if curve=='Szyszkowski':
    plot_1,plot_2=sci.optimize.curve_fit(Szyszkowski,concentration,pi,sigma=Stdev)
    print plot_1,plot_2
    Beta=plot_1[0]
    alpha=plot_1[1]
    plot_1_tuple= tuple(plot_1)
    perr = np.sqrt(np.diag(plot_2))
#    perr=tuple(perr)
    plot_tuple=(plot_1[0], perr[0],plot_1[1],perr[1])
#    trial_2= (plot_1_tuple,perr)
    plt.figure()
    plt.title(surfactant_title_Szysz,titleFont)
    plt.semilogx(concentration,pi, 'ro', label='Experimental',ms=size)
    plt.semilogx(concentration,Szyszkowski(concentration,*plot_1),'g----',label='Szyszkowski eqn fit, where $\Gamma_{\infty}$ = %5.3f $\pm$ %5.3e , $K_{L}$ =%5.3f $\pm$  %5.3e ' % plot_tuple)
#    plt.title(surfactant_title_single)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Concentration (mM)',fontdict=axesFont)
    plt.ylabel(r"$\pi$",fontdict=axesFont)
    plt.legend(loc='best', fontsize=13)
    plt.errorbar(concentration,pi,yerr= Stdev, fmt= ' ')
    plt.draw()
    

if adsorption=='Langmuir_ionic':
   LangAdsorp=Langmuir(concentration,Beta,alpha,True)
   plt.figure()
   plt.title('Langmuir Adsorption isotherm, ionic surfactant', titleFont)
   plt.semilogx(concentration,LangAdsorp, 'b-',label= 'Langmuir ionic adsorption isotherm for ' + str(SurfactantName))
#   plt.semilogx(concentration,pi,Szyszkowski(concentration,*plot_1),'g----',label='Szyszkowski eqn fit')
   plt.xticks(fontsize=20)
   plt.yticks(fontsize=20)
   plt.xlabel('Concentration (mM)',fontdict=axesFont)
   plt.ylabel(r"$\Gamma_{2,1}$",fontdict=axesFont)
   plt.legend(loc='best', fontsize=20)
   plt.draw()
   plt.show()

elif adsorption=='Langmuir':
   LangAdsorp=Langmuir(concentrations_new,Beta,alpha,False)
   plt.figure()
   plt.title('Langmuir Adsorption isotherm', titleFont)
   plt.semilogx(concentrations_new,LangAdsorp, 'b-',label= 'Langmuir adsorption isotherm for ' + str(SurfactantName))
#   plt.semilogx(concentration,pi,Szyszkowski(concentration,*plot_1),'g----',label='Szyszkowski eqn fit')
   plt.xticks(fontsize=20)
   plt.yticks(fontsize=20)
   plt.xlabel('Concentration (mM)',fontdict=axesFont)
   plt.ylabel(r"$\Gamma_{2,1}$",fontdict=axesFont)
   plt.legend(loc='best', fontsize=20)
   plt.draw()
   plt.show()

else:
    LangAdsorp=Langmuir(concentration,Beta,alpha,True)
    plot_3,plot_4=sci.optimize.curve_fit(Frumkin_state,LangAdsorp,pi)
    print plot_3,plot_4
    Beta_1=plot_3[0]
    plt.figure()
    plt.title(surfactant_title_single,titleFont)
    plt.semilogx(concentration,pi, 'ro', label='Experimental',ms=size)
    plt.semilogx(concentration,Frumkin_state(concentration,*plot_3),'b--',label='Frumkin eqn of state, where B=%5.3f' %(plot_3))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Concentration (mM)', fontdict=axesFont)
    plt.ylabel('Surface Pressure $\pi$ ',fontdict=axesFont)
    plt.legend(loc='best', fontsize=20)
    plt.errorbar(concentration,pi,yerr= Stdev, fmt= ' ')
    plt.draw()
#    def Frumkin(G,C,B,a,A): question is whether Adsorption isotherm should use parameters from Szyszkowski eqn of state or Frumkin
    A_0=Beta_1    
    FrumAdsorp=sci.odr(Frumkin,A_0,LangAdsorp,concentration,Beta,alpha)
    print FrumAdsorp
    plt.show()
    




#if adsorption=='Frumkin':
#    plot_1,plot_2= sci.optimize.curve_fit(Langmuir,concentration,pi)
#    plt.figure()
#    plt.title(surfactant_title_single,titleFont)
#    plt.semilogx(concentration,pi, 'ro', label='Experimental',ms=size)
#    plt.semilogx(concentration,Szyszkowski(concentration,*plot_1),'g----',label='Gibbs Adsorption fit, where B=%5.3f,a=%5.3f' % tuple(plot_1))
#    plt.xticks(fontsize=20)
#    plt.yticks(fontsize=20)
#    plt.xlabel('Concentration (mM)', fontdict=axesFont)
#    plt.ylabel('Surface Pressure, (sigma_0-sigma, mN/m)',fontdict=axesFont)
#    plt.legend(loc='best', fontsize=20)
#    plt.errorbar(concentration,pi,yerr= Stdev, fmt= ' ')
#    plt.draw()
#    plt.show()
#    

#fig,ax1 = plt.subplots()
#        plt.title('DI Water, 24 Hr Test (0.1 Hz)',titleFont)
#        plt1 = ax1.plot(timeVec,dropVolVec,'bo',markeredgewidth=0.0,label='Drop Volume')
#        ax1.set_xlabel('Time (hr)',fontdict=axesFont)
#        ax1.set_xticks(np.arange(0,max(timeVec),6))
#        ax1.set_ylabel('Drop Volume (mm$^3$)',fontdict=axesFont)
#        ax1.tick_params('y',colors='k')
#plt.ylabel(r"$\alpha$")
