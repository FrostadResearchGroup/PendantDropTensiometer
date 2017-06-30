#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 18:26:21 2017
@author: vanessaebogan
"""

import numpy as np
from scipy import optimize
from scipy.integrate import ode
import matplotlib.pyplot as plt
import time

t0 = time.time()

#set equations
def ode_system(t,y,bond):
    """
    Outputs system of ordinary differential equations with non-dimensionalized
    coordinates --> f = [d(fi)/ds , d(X)/ds , d(Z)/ds]
    t    = float - range of integration vector
    y    = float - desired solution to ODE system [fiVec, xVec, zVec]
    bond = float - Bond number
    """
    phi, r, z = y
    f = [2-bond*(z)-np.sin(phi)/(r), np.cos(phi), np.sin(phi)]
    return f
    
#define ode solver function
def young_laplace(Bo,nPoints,L):
    """
    Bo      = float - Bond number
    nPoints = int - number of integration points desired
    L       = float - final arc length for range of integration
    """
        
    # set solver
    solver = ode(ode_system)
    solver.set_integrator('dopri5')

    current_s=L
    N=nPoints

    bond=Bo
    
    #sets bond number to value inputted when ylp_func is called by solver
    solver.set_f_params(bond)

    #set initial values
    s0 = 0
    y0 = [0.00001,0.00001,0.00001]
    solver.set_initial_value(y0,s0)
    
    # Create the array `t` of time values at which to compute
    # the solution, and create an array to hold the solution.
    # Put the initial value in the solution array.
    s1 = current_s
    s = np.linspace(s0, s1, N)
    sol = np.empty((N, 3))
    sol[0] = y0
    
    # Shooter rountine: repeatedly call the `integrate` method to advance the
    # solution to time t[k], and save the solution in sol[k].
    k = 1
    while solver.successful() and solver.t < s1:
        solver.integrate(s[k])
        sol[k] = solver.y
        k += 1
        
    r = sol[:,1]
    z = sol[:,2]
        
    return r,z



def get_response_surf(p1Range,p2Range,obj_fun,xData,zData,deltaRho,N=100):
    """
    Plot the error surface for an objective function for a 2D optimization 
    problem.
    p1Range = List - [minVal,maxval]
    p2Range = List - [minVal,maxval]
    
    """
    
    # Create maxtrices of parameter value pairs to test
    p1List = np.linspace(p1Range[0],p1Range[1],N)
    p2List = np.linspace(p2Range[0],p2Range[1],N)
    X,Y = np.meshgrid(p1List,p2List)
    # Initialize matrix of error values
    Z = np.zeros_like(X)
    
    # Compute error for each parameter pair
    for i in range(len(X)):
        for j in range(len(Y)):
            p1 = X[i,j]
            p2 = Y[i,j]
            Z[i,j] = obj_fun([p1,p2],deltaRho,xData,zData)
            
    return X,Y,Z
    
    
def get_test_data(sigma,r0,deltaRho,N,L):
    
    """
    Generate drop profile data for testing purposes.
    sigma = float: surface tension in N/m
    r0 = 
    deltaRho
    N = int: number of data points on one side of droplet (give 2N-1 points)
    L = float: 
    """
    
    # Define Bond Number and solve Young Laplace Eqn.
    bond = deltaRho*9.81*r0**2/sigma
    
    xData,zData = young_laplace(bond,N,L)
    xData *= r0
    zData *= r0
    
    xData = np.append(list(reversed(-xData)),xData[1:])
    zData = np.append(list(reversed(zData)),zData[1:])
    
    # Convert to arrays and artificially pixelate
    xData = np.array(xData)
    zData = np.array(zData)
    xData = np.int64(xData*100000)/100000.0
    zData = np.int64(zData*100000)/100000.0
       
    return xData, zData
    
def get_data_apex(xData,zData):
    """
    Determines droplet apex from x and z coordinates provided through image
    processing
    
    xData = float -  x coordinates of droplet
    zData = float - z coordinates of droplet     
    """

    # finding index of all the minimum z locations to identify location of apex
    indices = np.argwhere(zData == np.min(zData))
    xApex = np.average(xData[indices])
    zApex = np.min(zData)
    
    apexIndex = np.average(indices)
    
    xApex = np.array([xApex])
    zApex = np.array([zApex])
    
    return xApex,zApex,apexIndex,indices

def add_interp_point(xData,zData,xApex,zApex,index):
    """
    Adds interpolated data point if the number of apex indices are an even number 
    (result of pixelation)
    
    xData = float - x coordinates of droplet
    zData = float - z coordinates of droplet
    xApex = float - x coordinates at droplet apex    
    zApex = float - z coordinates at droplet apex
    index = float - indices of droplet apex (from zData)      
    """
    newApexIndex = int(np.average(index))
    
    xData = np.insert(xData,newApexIndex,xApex)
    zData = np.insert(zData,newApexIndex,zApex)
    
    return xData,zData,newApexIndex
    
  
def tiling_matrix(zActualLeft,zActualRight,zModel):
    """
    Creates tiled matrices for subsequent model to data point comparison.
    
    zActualLeft  = float - z coordinates of droplet on left side
    zActualRight = float - x coordinates of droplet on right side
    zModel       = float - z coordinates of theorectical droplet (one side)
    """
    
    #building matrices for indexing based on z-value comparison
    zDatagridLeft=np.array([zActualLeft,]*len(zModel))
    zModelgridLeft=np.array([zModel,]*len(zActualLeft)).transpose()
    
    #building matrices for indexing based on z-value comparison
    zDatagridRight=np.array([zActualRight,]*len(zModel))
    zModelgridRight=np.array([zModel,]*len(zActualRight)).transpose()
    
    return zDatagridLeft,zModelgridLeft,zDatagridRight,zModelgridRight
    
def indexing(zDatagridLeft,zDatagridRight,zModelgridLeft,zModelgridRight):
    """
    Searches for closest distances between actual and theorectical points.
    
    zDatagridLeft = float - tiled matrix (same values column-wise) of z coordintates 
                            of droplet on left side, size = [len(zModel),len(zActualLeft)]
    zDatagridRight = float - tiled matrix (same values column-wise) of z coordintates 
                            of droplet on right side, size = [len(zModel),len(zActualRight)]
    zModelgridLeft = float - tiled matrix (same values row-wise) of theorectical z coordintates 
                            of droplet (one side), size = [len(zModel),len(zActualLeft)]
    zModelgridRight = float - tiled matrix (same values row-wise) of theorectical z coordintates 
                            of droplet (one side), size = [len(zModel),len(zActualRight)]
    """
    
    #indexing location of closest value
    indexLeft=np.argmin(np.abs((zModelgridLeft-zDatagridLeft)),axis=0)
    indexRight=np.argmin(np.abs((zModelgridRight-zDatagridRight)),axis=0)        
    
    return indexLeft,indexRight
    
def split_data(xActual,zActual):
    """
    Splits x and z droplet coordinates into two halves at the apex.
    
    xData = float - x coordinates of droplet
    zData = float - z coordinates of droplet   
    """

#    #find apex of droplet and the corresponding index (or indices) and values
#    xApex,zApex,apexIndex,indices = get_data_apex(xActual,zActual)
#        
#    if len(indices) % 2 == 0:
#        #add interpolated point if there does not exist one center apex point
#        xData,zData,newApexIndex = add_interp_point(xActual,zActual,xApex,zApex,apexIndex)
#        #parsing into left and right sections of data
#        xDataLeft = xData[:newApexIndex+1][::-1]
#        zDataLeft = zData[:newApexIndex+1][::-1]
#        xDataRight = xData[newApexIndex:]
#        zDataRight = zData[newApexIndex:]
#    else:
#        xDataLeft = xActual[:apexIndex+1][::-1]
#        zDataLeft = zActual[:apexIndex+1][::-1]
#        xDataRight = xActual[apexIndex:]
#        zDataRight = zActual[apexIndex:]
    xDataLeft = ()
    xDataRight = ()
    zDataLeft = ()
    zDataRight = ()
    
    for i in range(len(xActual)):
        if xActual[i] < 0:
            xDataLeft = np.append(xDataLeft,xActual[i])
            zDataLeft = np.append(zDataLeft,zActual[i])
        else:
            xDataRight = np.append(xDataRight,xActual[i])
            zDataRight = np.append(zDataRight,zActual[i])              

    
    return xDataLeft,zDataLeft,xDataRight,zDataRight



def objective_fun_v2(params,deltaRho,xDataLeft,zDataLeft,xDataRight,
                     zDataRight,sFinal,numberPoints=1000):
    """
    Calculates the sum of residual error squared between x data and theorectical points
    through a comparison of z coordinates.
    
    params   = float - fitting parameters for optimization rountine
    deltaRho = float - density differential between fluids
    xData    = float - x coordinates of droplet 
    zData    = float - z coordinates of droplet
    """
    #building relationship from params to bond number
    gamma = params[0]
    apexRadius = params[1]
    bond = deltaRho*9.81*apexRadius**2/gamma
    
    #throwing bond number into curve/coordinate generator
    xModel,zModel = young_laplace(bond,numberPoints,sFinal)
    
    #x and z coordinates with arc length
    xModel = xModel*apexRadius
    zModel = zModel*apexRadius
        
    #creates tiling matrices for subsequent indexing of data to model comparison
    zDatagridLeft,zModelgridLeft,zDatagridRight,zModelgridRight = tiling_matrix(zDataLeft,zDataRight,zModel)
    
    #indexes closest value from model to data    
    indexLeft,indexRight = indexing(zDatagridLeft,zDatagridRight,zModelgridLeft,zModelgridRight)
    
    #building r squared term
    rxLeft=xModel[indexLeft]+xDataLeft
    rxRight=xModel[indexRight]-xDataRight
    
    #returning residual sum of squares
    rsq=np.sum(rxLeft**2)+np.sum(rxRight**2)

    return rsq

def bond_calc(xActual,zActual):
    """
    Spits out an initial guess for bond Number based on droplet profile
    """
    #splitting data at apex into left and right side
    xDataLeft,zDataLeft,xDataRight,zDataRight = split_data(xActual,zActual)

    #looking for Xe    
    xeLeft = max(abs(xDataLeft))
    xeRight = max(abs(xDataRight))

    #looking for Xs    
    indicesLeft = np.argwhere(zDataLeft == 2*xeLeft)    
    indicesRight = np.argwhere(xDataRight == 2*xeRight)
        
    indexLeft = round(np.average(indicesLeft))
    indexRight = round(np.average(indicesRight))

    xsLeft = zDataLeft[indexLeft]
    xsRight = zDataRight[indexRight]

    #averaging left and right values
    sLeft = xsLeft/xeLeft
    sRight = xsRight/xeRight
    sAvg = (sLeft+sRight)/2        
    
    return sAvg
    
def s_interp(sAvg):
    """
    Searches for value to interpolate for s vs 1/H
    """
    #create table as listed in Bidwell et. al, 1964 
    #table needs to be extended to 1.1

    
    #need to fix
    bondGuess = 1    
    return bondGuess

  
def get_data_arc_len(xActualLeft,zActualLeft,xActualRight,zActualRight,r0Guess):
    """
    Computes the arc length of data points assuming that 
    Computes the arc length of data points assuming straight line distances
    xData = float - x coordinates of droplet
    zData = float - z coordinates of droplet 
    """

    #add data point at true apex of (0,0)
    xActualLeft = np.append(xActualLeft,0)
    xActualRight = np.append(0,xActualRight)
    zActualLeft = np.append(zActualLeft,0)
    zActualRight = np.append(0,zActualRight)

    # calculate the straight line distance between each point
    arcDistLeft = ((abs(xActualLeft[1:])-abs(xActualLeft[:-1]))**2
                    + (abs(zActualLeft[1:])-abs(zActualLeft[:-1]))**2)**0.5
    
     
    arcDistRight = ((abs(xActualRight[1:])-abs(xActualRight[:-1]))**2
                    + (abs(zActualRight[1:])-abs(zActualRight[:-1]))**2)**0.5
    
    # creating a summated arclength vector 
    sumArcLeft = np.sum(arcDistLeft)/r0Guess
    sumArcRight = np.sum(arcDistRight)/r0Guess
     
    # return largest s value
    print(sumArcLeft)
    if sumArcLeft > sumArcRight:            
        return sumArcLeft
    else:
        return sumArcRight
  
def final_script(xActual,zActual,sigmaGuess,deltaRho,nReload,
                 bondGuess=0.2,nPoints=2000):
    
    #splitting data at apex into left and right side
    xDataLeft,zDataLeft,xDataRight,zDataRight = split_data(xActual,zActual)
    
    #determine necessary range of integration 
        
    
    # initial guesses to start rountine 
    r0Guess = (bondGuess*sigmaGuess/(deltaRho*9.81))**0.5
    initGuess=[sigmaGuess,r0Guess]

    intRange = get_data_arc_len(xDataLeft,zDataLeft,xDataRight,zDataRight,r0Guess)
    # calling out optimization routine with reload
    for i in range(nReload):
        r=optimize.minimize(objective_fun_v2,initGuess,args=(deltaRho,
                            xDataLeft,zDataLeft,xDataRight,zDataRight,intRange),
                            method='Nelder-Mead',tol=1e-9)
        initGuess=[r.x[0],r.x[1]]
        sigmaFinal=r.x[0]
        r0Final=r.x[1]
        bondFinal=deltaRho*9.81*r0Final**2/sigmaFinal
        
        intRangeFinal = get_data_arc_len(xDataLeft,zDataLeft,xDataRight,zDataRight,r0Final)
        xFit,zFit=young_laplace(bondFinal,nPoints,intRangeFinal)
        
        #xFit = xOffset + xFit
        #zFit = zOffset + zFit
        
        # plot values with fitted bond number and radius of curvature at apex            
        xCurveFit=xFit*r0Final
        zCurveFit=zFit*r0Final
        xCurveFit_App=np.append(list(reversed(-xCurveFit)),xCurveFit[1:])
        zCurveFit_App=np.append(list(reversed(zCurveFit)),zCurveFit[1:])       
     
    
        plt.figure()
        plt.plot(xActual,zActual,'ro')
        plt.axis('equal')
        plt.plot(xCurveFit_App,zCurveFit_App,'b')
        plt.pause(1)

    return sigmaFinal,r0Final,bondFinal




    
#t1 = time.time()


    
if __name__ == "__main__":
    
    plt.close('all')
      
    # fitting based on z coordinates
    testObjFunV2 = False
    # importing test data for analysis
    testData = True
    # vizualization of objective function as a surface plot
    viewObjFunSurf = False
    #summation arc lengh
    testArcSum = True
    
    if testObjFunV2 or testData or testArcSum:
        # Generate test data for objective functions
        sigma = 0.06
        r0 = .0015
        deltaRho = 998
        L = 3.5
        nPoints = 2000
        Bond_actual = deltaRho*9.81*r0**2/sigma
        xActual,zActual = get_test_data(sigma,r0,deltaRho,nPoints,L)
         
        #splitting data at apex into left and right side
        xDataLeft,zDataLeft,xDataRight,zDataRight = split_data(xActual,zActual)
        
        plt.figure()
        plt.plot(xActual,zActual,'x')
        plt.axis('equal')
    

    if testObjFunV2:       
              
        # initial guesses to start rountine 
        nReload = 3
        sigmaGuess = 0.03
        R0Guess = 0.001
    
        initGuess=[sigmaGuess,R0Guess]
 
        # calling out optimization routine with reload
        for i in range(nReload):
            r=optimize.minimize(objective_fun_v2,initGuess,args=(deltaRho,
                                xDataLeft,zDataLeft,xDataRight,zDataRight),
                                method='Nelder-Mead',tol=1e-9)
            initGuess=[r.x[0],r.x[1]]
            sigmaFinal=r.x[0]
            r0Final=r.x[1]
            bondFinal=deltaRho*9.81*r0Final**2/sigmaFinal

            xFit,zFit=young_laplace(bondFinal,nPoints,L)
            
            # plot values with fitted bond number and radius of curvature at apex            
            xCurveFit=xFit*r0Final
            zCurveFit=zFit*r0Final
            xCurveFit_App=np.append(list(reversed(-xCurveFit)),xCurveFit[1:])
            zCurveFit_App=np.append(list(reversed(zCurveFit)),zCurveFit[1:])       
         
        
            plt.figure()
            plt.plot(xActual,zActual,'ro')
            plt.axis('equal')
            plt.plot(xCurveFit_App,zCurveFit_App,'b')
            plt.pause(1)

    t1 = time.time()
     
    print "Running Time:",t1-t0,"seconds"
    
    if testArcSum:
        intRange = get_data_arc_len(xDataLeft,zDataLeft,xDataRight,zDataRight,r0)
        



#    if viewObjFunSurf:
#        #Create Test Data
#        sigma=0.05
#        r0=.005
#        deltaRho=900
#        x,z = get_test_data(sigma,r0,deltaRho)
#        
#        X,Y,Z = get_response_surf([.02,.1],[.001,.01],objective_fun,x,z,
#                           deltaRho,N=10)
#                           
#        ax = Axes3D(plt.figure())
#        ax.plot_surface(X,Y,np.log10(Z),linewidth=0,antialiased=False)
