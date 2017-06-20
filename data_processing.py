#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 18:26:21 2017
@author: vanessaebogan
"""

import numpy as np

from numpy import loadtxt
from scipy import optimize
from scipy.integrate import ode
import matplotlib.pyplot as plt


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
    
    # Repeatedly call the `integrate` method to advance the
    # solution to time t[k], and save the solution in sol[k].
    k = 1
    while solver.successful() and solver.t < s1:
        solver.integrate(s[k])
        sol[k] = solver.y
        k += 1
        
    r = sol[:,1]
    z = sol[:,2]
        
    return s,r,z


def objective_fun_v1(params,deltaRho,xData,zData,sData,numberPoints=700):
    """
    Calculates the sum of residual error squared between data and fitted curve
    with respect to s: x(s) + z(s) leveraging arc length
    params   = float - fitting parameters for optimization rountine
    deltaRho = float - density differential between fluids
    xData    = float - x coordinates of droplet 
    zData    = float - z coordinates of droplet
    sData    = float - arc length of droplet
    """
    #building relationship from params to bond number
    gamma = params[0]
    apexRadius = params[1]
    bond = deltaRho*9.81*apexRadius**2/gamma
    sFinal = 2*sData[-1]
    
    #throwing bond number into curve/coordinate generator
    sModel,xModel,zModel = young_laplace(bond,numberPoints,sFinal)
    
    #x and z coordinates with arc length
    xModel *= apexRadius
    zModel *= apexRadius
    sModel *= apexRadius
    
    #building matrices for indexing based on arclength comparison
    sDatagrid=np.array([sData,]*len(sModel))
    sModelgrid=np.array([sModel,]*len(sData)).transpose()
       
    #indexing location of closest value
    index=np.argmin(abs((sModelgrid-sDatagrid)),axis=0)

    #building r squared term
    rx=xModel[index]-xData
    rz=zModel[index]-zData

    #returning square root of residual sum of squares
    return np.sum((rx**2+rz**2)**0.5)

def objective_fun_v2(params,deltaRho,xData,zData,numberPoints=700,sFinal=2):
    """
    Calculates the sum of residual error squared between data and fitted curve
    with respect to z: x(z) leveraging z coordinates
    params   = float - fitting parameters for optimization rountine
    deltaRho = float - density differential between fluids
    xData    = float - x coordinates of droplet 
    zData    = float - z coordinates of droplet
    """
    #building relationship from params to bond number
    gamma=params[0]
    apexRadius=params[1]
    bond=deltaRho*9.81*apexRadius**2/gamma
    
    #throwing bond number into curve/coordinate generator
    sModel,xModel,zModel = young_laplace(bond,numberPoints,sFinal)
    
    #scaled s (normalized), x and z coordinates
    xModel *= apexRadius
    zModel *= apexRadius
    
    zDatagrid=np.array([zData,]*len(zModel))
    zModelgrid=np.array([zModel,]*len(zData)).transpose()
       
    #indexing location of closest normalized value
    index=np.argmin(abs((zModelgrid-zDatagrid)),axis=0)

    #building r squared term
    rx=xModel[index]-xData

    #returning square root of residual sum of squares
    return np.sum((rx**2)**0.5)
    
    
def get_response_surf(p1Range,p2Range,obj_fun,xData,zData,deltaRho,N=100):
    """
    Plot the error surface for an objective function for a 2D optimization 
    problem.
    p1Range = List [minVal,maxval]
    p2Range = List [minVal,maxval]
    
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
    
    
def get_test_data(sigma,r0,deltaRho,N=50,L=1):
    
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
    sData,xData,zData = young_laplace(bond,N,L)
    xData *= r0
    zData *= r0
    
    xData = np.append(list(reversed(-xData)),xData[1:])
    zData = np.append(list(reversed(zData)),zData[1:])
    
    return xData, zData
    
def get_data_apex():
    """
    
    """
    
    return
    
def get_data_arc_len(xData,zData):
    """
    Computes the arc length of data points assuming that 
    """


if __name__ == "__main__":
    
    plt.close('all')
      
    # fitting based on arc length
    testObjFunV1 = False
    # fitting based on z coordinates
    testObjFunV2 = True
    realDataPoints = False
    viewObjFunSurf = False
    
    if testObjFunV1 or testObjFunV2:
        # Generate test data for objective functions
        sigma = 0.05
        r0 = .005
        deltaRho = 900
        L = 1
        nPoints = 150
        Bond_actual = deltaRho*9.81*r0**2/sigma
        xActual,zActual = get_test_data(sigma,r0,deltaRho,N=50,L=1)
    
    if testObjFunV1:
        
        # initial guesses to start routine
        nReload = 3
        sigmaGuess = 3*sigma
        R0Guess = 3*r0
    
        initGuess=[sigmaGuess,R0Guess]
        
        # Compute arc length of data points
        
        # calling out optimization routine with reload
        print('First Objective Function')
        for i in range(nReload):
            r = optimize.minimize(objective_fun_v1,initGuess,args=(deltaRho,xActual,
                                  zActual,sActual),method='Nelder-Mead')
            initGuess=[r.x[0],r.x[1]]
            sigmaFinal=r.x[0]
            r0Final=r.x[1]
            bondFinal=deltaRho*9.81*r0Final**2/sigmaFinal
            print(bondFinal)
            
            # plot values with fitted bond number and radius of curvature at apex
            fitted=young_laplace(bondFinal,nPoints,L)
            
            xCurveFit=fitted[:,1]*r0Final
            zCurveFit=fitted[:,2]*r0Final
            xCurveFit_App=np.append(list(reversed(-xCurveFit)),xCurveFit[1:])
            zCurveFit_App=np.append(list(reversed(zCurveFit)),zCurveFit[1:])       
         
        
            plt.plot(xActual,zActual,'ro')
            plt.axis('equal')
            plt.plot(xCurveFit_App,zCurveFit_App,'b')   


    if testObjFunV2:       
         
        
        # initial guesses to start rountine 
        nReload = 3
        sigmaGuess=3*sigma
        R0Guess=3*r0
    
        initGuess=[sigmaGuess,R0Guess]
 
        # calling out optimization routine with reload
        print('Second Objective Function')
        for i in range(nReload):
            r=optimize.minimize(objective_fun_v2,initGuess,args=(deltaRho,xActual,
                                  zActual),method='Nelder-Mead')
            initGuess=[r.x[0],r.x[1]]
            sigmaFinal=r.x[0]
            r0Final=r.x[1]
            bondFinal=deltaRho*9.81*r0Final**2/sigmaFinal
            print(bondFinal)

            # plot values with fitted bond number and radius of curvature at apex            
            fitted=young_laplace(bondFinal,nPoints,L)
            
            xCurveFit=fitted[:,1]*r0Final
            zCurveFit=fitted[:,2]*r0Final
            xCurveFit_App=np.append(list(reversed(-xCurveFit)),xCurveFit[1:])
            zCurveFit_App=np.append(list(reversed(zCurveFit)),zCurveFit[1:])       
         
        
            plt.plot(xActual,zActual,'ro')
            plt.axis('equal')
            plt.plot(xCurveFit_App,zCurveFit_App,'b')   
            
    if realDataPoints:
        # pulling the data from .txt file
        data = loadtxt("testfile.txt",delimiter=",")
        # data not arranged in proper order (even vs. odd rows)
        xData = data[:,0]
        zData = data[:,1]        
        
        # need to re-order coordinate arrangement to line up with s-vector
        # seperate left and right side of droplet by even and odd rows
        for i in range(len(data[:,0])):
            if i == 0:
                xDataRight = xData[i]
                zDataRight = zData[i]
                xDataLeft = ()
                zDataLeft = ()
            elif i % 2 == 0:
                xDataRight = np.append(xDataRight,xData[i])
                zDataRight = np.append(zDataRight,zData[i])
            else:
                xDataLeft = np.append(xDataLeft,xData[i])
                zDataLeft = np.append(zDataLeft,zData[i])
        
        magRatio = 0.006
        
        # averaging the right and left side droplet coords for quicker calcs
        index = len(xDataRight)-len(xDataLeft)

        if index < 0:
            xCoords = ((xDataRight-xDataLeft[:-index])/2)*magRatio
            zCoords = ((zDataRight+zDataLeft[:-index])/2)*magRatio
        elif index > 0:
            xCoords = ((xDataRight[:-index]-xDataLeft)/2)*magRatio
            zCoords = ((zDataRight[:-index]+zDataLeft)/2)*magRatio
        else:
            xCoords = ((xDataRight-xDataLeft)/2)*magRatio
            zCoords = ((zDataRight+zDataLeft)/2)*magRatio

        xActual = xCoords[::-1]
        zActual = zCoords[::-1]
        
        xActual_App=np.append(list(reversed(-xActual)),xActual[1:])
        zActual_App=np.append(list(reversed(zActual)),zActual[1:])
        
        # calculate the straight line distance between each point
        Distances = (((xActual[1:]+xActual[:-1])**2+(zActual[1:]-zActual[:-1])**2)
        **0.5)
        
        # creating a summated arclength vector 
        for i in range(len(Distances)):
            if i==0:
                sActual = np.append(0,Distances[i])
            else:
                sActual = np.append(sActual,sum(Distances[:i+1]))
                
        # initial guesses for radius at apex and surface tension
        r0Guess = np.max(abs(xActual))
        deltaRho = 900
        sigmaGuess = 0.04

        initGuess=[r0Guess,sigmaGuess]        
        
        # calling out optimization routine with reload
        for i in range(1):
            r=optimize.minimize(objective_fun_v1,initGuess,args=(deltaRho,xActual,
                                  zActual,sActual),method='Nelder-Mead')
            initGuess=[r.x[0],r.x[1]]
            sigmaFinal=r.x[0]
            r0Final=r.x[1]
            Bond_final=deltaRho*9.81*r0Final**2/sigmaFinal
            
            nPoints=1000
            L = sActual[len(sActual)-1]

            # plot values with fitted bond number and radius of curvature at apex            
            fitted=young_laplace(Bond_final,nPoints,L)
            
            xCurveFit=fitted[:,1]*r0Final
            zCurveFit=fitted[:,2]*r0Final
            xCurveFit_App=np.append(list(reversed(-xCurveFit)),xCurveFit[1:])
            zCurveFit_App=np.append(list(reversed(zCurveFit)),zCurveFit[1:])       
         
        
            plt.plot(xActual_App,zActual_App,'ro')
            plt.axis('equal')
            plt.plot(xCurveFit_App,zCurveFit_App,'b')
    
    if viewObjFunSurf:
        #Create Test Data
        sigma=0.05
        r0=.005
        deltaRho=900
        x,z = get_test_data(sigma,r0,deltaRho)
        
        X,Y,Z = get_response_surf([.02,.1],[.001,.01],objective_fun,x,z,
                           deltaRho,N=10)
                           
        ax = Axes3D(plt.figure())
        ax.plot_surface(X,Y,np.log10(Z),linewidth=0,antialiased=False)
