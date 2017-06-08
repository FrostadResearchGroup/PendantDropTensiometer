#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 18:26:21 2017

@author: vanessaebogan
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
#from numpy import loadtxt
from ylpsolver_fitting import odesolver


def workfile(params,deltaRho,xData,zData):
    gamma = params[0]
    R0 = params[1]
    #takes in initial bond number and optimizes through objective function as 
    #defined by N. Alvarez et. al
    bond= deltaRho*9.81*R0**2/gamma
    # print outputs of each section
    model=odesolver(bond)
    
    xModel=model[:,1]
    zModel=model[:,2]
    xModel_App=np.append(list(reversed(-xModel)),xModel[1:])
    zModel_App=np.append(list(reversed(zModel)),zModel[1:])
     
    xModelFit=xModel_App[:-1]
    zModelFit=zModel_App[:-1]
    xModelFit_plus=xModel_App[1:]
    zModelFit_plus=zModel_App[1:]
    
    xData*=R0
    zData*=R0
    
    xdatagrid=np.tile(xData,(len(xModelFit),1))
    zActualgrid=np.tile(zData,(len(zModelFit),1))
    
    xFit=np.tile(xModelFit,(len(xData),1)).transpose()
    zFit=np.tile(zModelFit,(len(zData),1)).transpose()
    xFit_plus=np.tile(xModelFit_plus,(len(xData),1)).transpose()
    zFit_plus=np.tile(zModelFit_plus,(len(zData),1)).transpose()
    
    a=(zFit-zActualgrid)*(xFit-xFit_plus)
    b=(xFit-xdatagrid)*(zFit-zFit_plus)
    c=np.power((xFit-xFit_plus),2)
    d=np.power((zFit-zFit_plus),2)
    
    output=np.power(((a+b)/np.power((c+d),0.5)),2)
    
    rsq=np.min(output,axis=0)
    
    return np.sum(rsq)

if __name__ == "__main__":
 
#### Acutal Data Points 
    Bond_actual=1  
    r0_actual=.8
    deltaRho=900
    sigmaActual=deltaRho*9.81*r0_actual**2/Bond_actual
    
    
    xActual=odesolver(Bond_actual)[:,1]
    zActual=odesolver(Bond_actual)[:,2]
    a
    xActual_App=np.append(list(reversed(-xActual)),xActual[1:])
    zActual_App=np.append(list(reversed(zActual)),zActual[1:])
    ################################################################  
    
    sigmaGuess=5
    R0Guess=.1
    
    initGuess=[sigmaGuess,R0Guess]
    
    r=scipy.optimize.minimize(workfile,initGuess,args=(deltaRho,xActual_App,
                              zActual_App),method='Nelder-Mead')
    
    sigmaFinal=r.x[0]
    R0Final=r.x[1]
    Bond_final=deltaRho*9.81*r.x[1]**2/r.x[0]
    
    fitted=odesolver(Bond_final)
    
    xCurveFit=fitted[:,1]*R0Final
    zCurveFit=fitted[:,2]*R0Final
    xCurveFit_App=np.append(list(reversed(-xCurveFit)),xCurveFit[1:])
    zCurveFit_App=np.append(list(reversed(zCurveFit)),zCurveFit[1:])
    
    
    #plt.plot(xActual_App,zActual_App,'ro')
    plt.plot(xCurveFit_App,zCurveFit_App,'bo')