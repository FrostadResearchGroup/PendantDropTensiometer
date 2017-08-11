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

def ode_system(t,y,bond):
    """
    Outputs system of ordinary differential equations with non-dimensionalized
    coordinates --> f = [d(fi)/ds , d(X)/ds , d(Z)/ds].
    
    t    = float - range of integration vector
    y    = float - desired solution to ODE system [fiVec, xVec, zVec]
    bond = float - Bond number
    """
    phi, r, z = y
    f = [2-bond*(z)-np.sin(phi)/(r), np.cos(phi), np.sin(phi)]
    return f
    
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


def get_response_surf(sigma,r0,theta,deltaRho,xActual,zActual,objective_fun_v2,N=100):
    """
    Plot the error surface for an objective function for a 2D optimization 
    problem.
    
    sigma            = float - surface tension (N/m)
    r0               = float - radius of curvature at apex (m)
    theta            = float - rotation angle
    deltaRho         = float - density difference between two fluids (kg/m^3)
    xActual          = float - x coordinates of droplet 
    zActual          = float - z coordinates of droplet
    objective_fun_v2 = function - objective function, comparison of x values
    """
    
    #split data and get arc length
    xDataLeft,zDataLeft,xDataRight,zDataRight = split_data(xActual,zActual)    
    sFinal = get_data_arc_len(xDataLeft,zDataLeft,xDataRight,zDataRight,r0)
    
    #create range to plot for surface tension and apex radius
    p1Range = [0,2*sigma]
    p2Range = [0,2*r0]
    
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
            Z[i,j] = objective_fun_v2([p1,p2,theta],deltaRho,xDataLeft,zDataLeft,xDataRight,
                                 zDataRight,sFinal,numberPoints=10)
                
    return X,Y,Z
    
    
def get_test_data(sigma,r0,deltaRho,N=100,L=30):   
    """
    Generates drop profile data for testing purposes (can comment out pixelation section).
    
    sigma    = float - surface tension in N/m
    r0       = float - radius of curvature at apex 
    deltaRho = float - density difference between two fluids
    N        = int - number of data points on one side of droplet (give 2N-1 points)
    L        = float - integration range 
    """    
        
    #define Bond Number and solve Young Laplace Eqn.
    bond = deltaRho*9.81*r0**2/sigma
    
    xData,zData = young_laplace(bond,N,L)
    xData = xData*r0
    zData = zData*r0
    
    xData = np.append(list(reversed(-xData)),xData[1:])
    zData = np.append(list(reversed(zData)),zData[1:])
    
#    #convert to arrays and artificially pixelate
#    xData = np.array(xData)
#    zData = np.array(zData)
#    xData = np.int64(xData*100000)/100000.0
#    zData = np.int64(zData*100000)/100000.0
       
    return xData, zData
    
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
    
    xActual = float - x coordinates of droplet
    zActual = float - z coordinates of droplet   
    """

   #find apex of droplet and the corresponding index (or indices) and values
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


def rotate_data(xActualLeft,zActualLeft,xActualRight,zActualRight,thetaVal):
    """
    Rotates data for optimization of theta parameter
    """
    #rotate data points with theta parameter
    xActualLeft = xActualLeft*np.cos(thetaVal) - zActualLeft*np.sin(thetaVal)
    zActualLeft = xActualLeft*np.sin(thetaVal) + zActualLeft*np.cos(thetaVal)
    
    xActualRight = xActualRight*np.cos(thetaVal) - zActualRight*np.sin(thetaVal)
    zActualRight = xActualRight*np.sin(thetaVal) + zActualRight*np.cos(thetaVal)
    
    return xActualLeft,zActualLeft,xActualRight,zActualRight

def objective_fun_v2(params,deltaRho,xDataLeft,zDataLeft,xDataRight,
                     zDataRight,sFinal,numberPoints=1000):
    """
    Calculates the sum of residual error squared between x data and 
    theorectical points through a comparison of z coordinates.
    
    params   = float - fitting parameters for optimization rountine
    deltaRho = float - density differential between fluids
    xData    = float - x coordinates of droplet 
    zData    = float - z coordinates of droplet
    """
    #building relationship from params to bond number
    gamma = params[0]
    apexRadius = params[1]
    theta = params[2]
    bond = deltaRho*9.81*apexRadius**2/gamma
    
    #throwing bond number into curve/coordinate generator
    xModel,zModel = young_laplace(bond,numberPoints,sFinal)


    #x and z coordinates with arc length
    xModel = xModel*apexRadius
    zModel = zModel*apexRadius
    
    #rotate data points with theta parameter
    xDataLeft,zDataLeft,xDataRight,zDataRight = rotate_data(xDataLeft,zDataLeft,xDataRight,zDataRight,theta)
    
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
    Finds s, xe and r0 parameters for initial guess of bond Number based on 
    droplet profile.
    
    xActual = float - x coordinates of droplet 
    zActual = float - z coordinates of droplet 
    """
    #splitting data at apex into left and right side
    xDataLeft,zDataLeft,xDataRight,zDataRight = split_data(xActual,zActual)

    #looking for Xe    
    xeLeft = max(abs(xDataLeft))
    xeRight = max(abs(xDataRight))
    xeAvg = np.average((xeLeft+xeRight))/2
    
    r0Guess = abs(xeRight)
    xeScal = xeAvg/r0Guess
    
    #looking for Xs    
    indicesLeft = np.argmin(abs(zDataLeft-2*xeLeft))    
    indicesRight = np.argmin(abs(zDataRight-2*xeRight))
        
    indexLeft = int(np.average(indicesLeft))
    indexRight = int(np.average(indicesRight))

    xsLeft = zDataLeft[indexLeft]
    xsRight = zDataRight[indexRight]

    #averaging left and right values
    sLeft = xsLeft/xeLeft
    sRight = xsRight/xeRight
    sAvg = (sLeft+sRight)/2        
    
    return sAvg,xeScal,r0Guess
    
def s_interp(sAvg,xeAvg):
    """
    Searches for value to interpolate for s vs 1/H, relationships as described
    by Ambwain and Fort Jr., 1979.
    
    sAvg  = float - average s value over two halves of droplet
    xeAvg = float - average xe value over two halves of droplet
    """
    
    if sAvg >= .9:
        hInv = (.30715/sAvg**2.84636) + (-.69116*sAvg**3)-(-1.08315*sAvg**2)+ \
        (-.18341*sAvg)-(.20970)
    elif sAvg >= .68:
        hInv = (.31345/sAvg**2.64267) - (.09155*sAvg**2)+(.14701**sAvg)-(.05877)
    elif sAvg >= .59:
        hInv = (.31522/sAvg^2.62435) - (.11714*sAvg^2)+(.15756*sAvg)-(.05285)
    elif sAvg >= .46:
        hInv = (.31968/sAvg^2.59725) - (.46898*sAvg^2)+(.50059*sAvg)-(.13261);
    elif sAvg >= .401:
        hInv = (.32720/sAvg^2.56651) - (.97553*sAvg^2)+(.84059*sAvg)-(.18069);
    else:
        print('Shape is too spherical');
        #Use formula for S > 0.401 even though it is wrong
        hInv = (.32720/sAvg**2.56651) - (.97553*sAvg**2)+(.84059*sAvg)-(.18069);
        
    bondGuess = -1/(4*hInv*(xeAvg)**2);

    return bondGuess

  
def get_data_arc_len(xActualLeft,zActualLeft,xActualRight,zActualRight,r0Guess):
    """
    Computes the arc length of data points through discretization using 
    straight line distances.
    
    xData = float - x coordinates of droplet
    zData = float - z coordinates of droplet 
    """

    #add data point at true apex of (0,0)
    xActualLeft = np.append(xActualLeft,0)
    xActualRight = np.append(0,xActualRight)
    zActualLeft = np.append(zActualLeft,0)
    zActualRight = np.append(0,zActualRight)

    #calculate the straight line distance between each point
    arcDistLeft = ((abs(xActualLeft[1:])-abs(xActualLeft[:-1]))**2
                    + (abs(zActualLeft[1:])-abs(zActualLeft[:-1]))**2)**0.5
    
     
    arcDistRight = ((abs(xActualRight[1:])-abs(xActualRight[:-1]))**2
                    + (abs(zActualRight[1:])-abs(zActualRight[:-1]))**2)**0.5
    
    # creating a summated arclength vector 
    sumArcLeft = np.sum(arcDistLeft)/r0Guess
    sumArcRight = np.sum(arcDistRight)/r0Guess
     
    # return largest s value

    if sumArcLeft > sumArcRight:            
        return sumArcLeft
    else:
        return sumArcRight
  
def optimize_params(xActual,zActual,bondGuess,r0Guess,deltaRho,nReload,
                 nPoints=1000,thetaGuess=0):
    """
    Optimizes bondnumber, apex radius of curvature, and rotational angle to fit
    curve (as described through system of ODEs) to data points. Outputs fitted
    parameters and resultant surface tension.
    
    xActual    = float - x-coordinate values of droplet data points
    zActual    = float - z-coordinate values of droplet data points
    bondGuess  = float - inital guess for bond number
    r0Guess    = float - inital guess for radius of curvature at apex
    deltaRho   = float - density difference between two fluids (user input)
    nReload    = int - number of reloads for optimization rountine (user input)
    nPoints    = int - number of points on theorectical curve
    thetaGuess = float - inital guess for rotational angle (radians)
    """
    
    #splitting data at apex into left and right side
    xDataLeft,zDataLeft,xDataRight,zDataRight = split_data(xActual,zActual)
    
    # initial guesses to start rountine
    sigmaGuess = deltaRho*9.81*r0Guess**2/bondGuess
    r0Guess = (bondGuess*sigmaGuess/(deltaRho*9.81))**0.5
    initGuess = [sigmaGuess,r0Guess,thetaGuess]


    intRange = get_data_arc_len(xDataLeft,zDataLeft,xDataRight,zDataRight,r0Guess)
    
    # calling out optimization routine with reload
    for i in range(nReload):
        r=optimize.minimize(objective_fun_v2,initGuess,args=(deltaRho,
                            xDataLeft,zDataLeft,xDataRight,zDataRight,intRange),
                            method='Nelder-Mead',tol=1e-9)
        initGuess = [r.x[0],r.x[1],r.x[2]]
        sigmaFinal = r.x[0]
        thetaFinal = r.x[2]
        r0Final = r.x[1] * np.cos(thetaFinal)
        bondFinal=deltaRho*9.81*r0Final**2/sigmaFinal
        
        intRangeFinal = get_data_arc_len(xDataLeft,zDataLeft,xDataRight,zDataRight,r0Final)
        xFit,zFit=young_laplace(bondFinal,nPoints,intRangeFinal)
        
        
        xActual = xActual*np.cos(thetaFinal) - zActual*np.sin(thetaFinal)
        zActual = xActual*np.sin(thetaFinal) + zActual*np.cos(thetaFinal)    
        
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

    return sigmaFinal,r0Final,thetaFinal,bondFinal








######################## For Testing Purposes #################################   
   
if __name__ == "__main__":
    
    plt.close('all')
      
    # fitting based on z coordinates
    testObjFunV2 = False
    # importing test data for analysis
    testData = False
    # vizualization of objective function as a surface plot
    viewObjFunSurf = False
    #summation arc lengh
    testArcSum = False
    #test initial Bond finder
    testInitBond = True
    
    if testObjFunV2 or testData or testArcSum or testInitBond:
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
        nReload = 1
        sigmaGuess = 0.05
        thetaGuess = 0
        s,xe = bond_calc(xActual,zActual)
        bondGuess = s_interp(s,xe)  

        r0Guess = (bondGuess*sigmaGuess/(deltaRho*9.81))**0.5
        initGuess = [sigmaGuess,r0Guess,thetaGuess]
        
        intRange = get_data_arc_len(xDataLeft,zDataLeft,xDataRight,zDataRight,r0Guess)
        
        # calling out optimization routine with reload
        for i in range(nReload):
            r=optimize.minimize(objective_fun_v2,initGuess,args=(deltaRho,
                                xDataLeft,zDataLeft,xDataRight,zDataRight,intRange),
                                method='Nelder-Mead',tol=1e-9)
            initGuess = [r.x[0],r.x[1],r.x[2]]
            sigmaFinal = r.x[0]            
            thetaFinal = r.x[2]
            r0Final = r.x[1] * np.cos(thetaFinal)
            bondFinal=deltaRho*9.81*r0Final**2/sigmaFinal
            
            intRangeFinal = get_data_arc_len(xDataLeft,zDataLeft,xDataRight,
                                             zDataRight,r0Final)
            xFit,zFit=young_laplace(bondFinal,nPoints,intRangeFinal)
            
            xFit = xFit*np.cos(-thetaFinal) - zFit*np.sin(-thetaFinal)
            zFit = xFit*np.sin(-thetaFinal) + zFit*np.cos(-thetaFinal)
        
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
     
        print "Surface tension = ",sigmaFinal,"N/m"
    
    if testArcSum:
        intRange = get_data_arc_len(xDataLeft,zDataLeft,xDataRight,zDataRight,r0)
    
    if testInitBond:
        s,xe = bond_calc(xActual,zActual)
        initBondGuess = s_interp(s,xe)

    #update this section with new script        
    if viewObjFunSurf:
        #Create Test Data
        sigma=0.05
        r0=.005
        deltaRho=900
        x,z = get_test_data(sigma,r0,deltaRho)
        
        X,Y,Z = get_response_surf([.02,.1],[.001,.01],objective_fun_v2,x,z,
                           deltaRho,N=10)
                           
        ax = Axes3D(plt.figure())
        ax.plot_surface(X,Y,np.log10(Z),linewidth=0,antialiased=False)
