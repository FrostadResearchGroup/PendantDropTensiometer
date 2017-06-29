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
    
    #find apex of droplet and the corresponding index (or indices) and values
    xApex,zApex,apexIndex,indices = get_data_apex(xActual,zActual)
        
    if len(indices) % 2 == 0:
        #add interpolated point if there does not exist one center apex point
        xData,zData,newApexIndex = add_interp_point(xActual,zActual,xApex,zApex,apexIndex)
        #parsing into left and right sections of data
        xDataLeft = xData[:newApexIndex+1][::-1]
        zDataLeft = zData[:newApexIndex+1][::-1]
        xDataRight = xData[newApexIndex:]
        zDataRight = zData[newApexIndex:]
    else:
        xDataLeft = xActual[:apexIndex+1][::-1]
        zDataLeft = zActual[:apexIndex+1][::-1]
        xDataRight = xActual[apexIndex:]
        zDataRight = zActual[apexIndex:]
        
    return xDataLeft,zDataLeft,xDataRight,zDataRight



def objective_fun_v2(params,deltaRho,xDataLeft,zDataLeft,xDataRight,
                     zDataRight,xShift,zShift,numberPoints=700,sFinal=100):
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
    xModel = xModel + xShift
    zModel = zModel + zShift
    
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
    s = np.linspace(0.2,.94,741)
    hInverse = np.array([19.35325, 19.15633, 18.96089, 18.76692, 18.57443,
                         18.38343, 18.19390, 18.00584, 17.81927, 17.63418,
                         17.45056, 17.26842, 17.08776, 16.90858, 16.73087,
                         16.55465, 16.37990, 16,20663, 16.03484, 15.86452,
                         15.69569, 15.52833, 15.36245, 15.19805, 15.03513,
                         14.87369, 14.71372, 14.55524, 14.39823, 14.24270,
                         14.08864, 13.93607, 13.78497, 13.63535, 13.48721,
                         13.34055, 13.19537, 13.05167, 12.90944, 12.76869,
                         12.62942, 12.49163, 12.35531, 12.22048, 12.08712,
                         11.95524, 11.82484, 11.69592, 11.56848, 11.44251,
                         11.31802, 11.19501, 11.07348, 10.95343, 10.83485,
                         10.71776, 10.60214, 10.48800, 10.37534, 10.26415,
                         10.15445, 10.04622,  9.93947,  9.83420,  9.73041,
                          9.62809,  9.52726,  9.42790,  9.33002,  9.23362,
                          9.13870,  9.04525,  8.95328,  8.86280,  8.77379,
                          8.68625,  8.60020,  8.51562,  8.43253,  8.35091,
                          8.27077,  8.19211,  8.11492,  8.03921,  7.96499,
                          7.89224,  8.05228,  7.98182,  7.91201,  7.84283,
                          7.77429,  7.70640,  7.63914,  7.57252,  7.50655,
                          7.44121,  7.37651,  7.31245,  7.24903,  7.18625,
                          6.53790,  6.48279,  6.42833,  6.37450,  6.32132,
                          6.26878,  6.21687,  6.16560,  6.11498,  6.06499,
                          6.01565,  5.96694,  5.91887,  5.87144,  5.82466,
                          5.77851,  5.73300,  5.68813,  5.64390,  5.64907,
                          5.60579,  5.56287,  5.52033,  5.47815,  5.43635,
                          5.39491,  5.35385,  5.31316,  5.27284,  5.23289,
                          5.19331,  5.15410,  5.11526,  5.07679,  5.03869,
                          5.00096,  4.96361,  4.92662,  4.89000,  4.85376,
                          4.47952,  4.44772,  4.41629,  4.38523,  4.37119,
                          4.34076,  4.31058,  4.28065,  4.25096,  4.22152,
                          4.19232,  4.16337,  4.13466,  4.10620,  4.07798,
                          4.05001,  4.02229,  3.99481,  3.96758,  3.94059,
                          3.91358,  3.88735,  3.86110,  3.83509,  3.80933,
                          3.78382,  3.75855,  3.73353,  3.70875,  3.68422,
                          3.65993,  3.63589,  3.61210,  3.58855,  3.56524,
                          3.54941,  3.52652,  3.50381,  3.48127,  3.45891,
                          3.43672,  3.41471,  3.39288,  3.37123,  3.34975,
                          3.32844,  3.30732,  3.28636,  3.26559,  3.24499,
                          3.22457,  3.20432,  3.18425,  3.16436,  3.14464,
                          3.12510,  3.10573,  3.08654,  3.06753,  3.04870,
                          3.03003,  3.01155,  2.99691,  2.97875,  2.96073,
                          2.94284,  2.92508,  2.90745,  2.88996,  2.87261,
                          2.85538,  2.83829,  2.82133,  2.80451,  2.78782,
                          2.77126,  2.75484,  2.73855,  2.72239,  2.70636,
                          2.69047,  2.67471,  2.65909,  2.64360,  2.62824,
                          2.61301,  2.59792,  2.58502,  2.57018,  2.55545,
                          2.54081,  2.52629,  2.51187,  2.49755,  2.48334,
                          2.46923,  2.45523,  2.44133,  2.42754,  2.41385,
                          2.40026,  2.38678,  2.37341,  2.36014,  2.34697,
                          2.33391,  2.32096,  2.30811,  2.29536,  2.28272,
                          2.27018,  2.25899,  2.24664,  2.23437,  2.22218,
                          2.21008,  2.19806,  2.18613,  2.17428,  2.16252,
                          2.15084,  2.13924,  2.12773,  2.11631,  2.10497,
                          2.09371,  2.08254,  2.07145,  2.06045,  2.04953,
                          2.03870,  2.02795,  2.01809,  2.00750,  1.99698,
                          1.98653,  1.97614,  1.96583,  1.95559,  1.94542,
                          1.93532,  1.92529,  1.91532,  1.90543,  1.89561,
                          1.88586,  1.87617,  1.86656,  1.85702,  1.84755,
                          1.83814,  1.82881,  1.81955,  1.81089,  1.80175,
                          1.79267,  1.78365,  1.77468,  1.76578,  1.75693,
                          1.74815,  1.73942,  1.73075,  1.72214,  1.71358,
                          1.70509,  1.69666,  1.68828,  1.67996,  1.67170,
                          1.66351,  1.65536,  1.64766,  1.63963,  1.63164,
                          1.62371,  1.61583,  1.60799,  1.60021,  1.59248,
                          1.58479,  1.57716,  1.56958,  1.56204,  1.55456,
                          1.54713,  1.53975,  1.53241,  1.52513,  1.51790,
                          1.44156,  1.43486,  1.42820,  1.42158,  1.41500,
                          1.40847,  1.40198,  1.39554,  1.38934,  1.38297,
                          1.37665,  1.37036,  1.36410,  1.35789,  1.35171,
                          1.34557,  1.33947,  1.33341,  1.32728,  1.32139,
                          1.31544,  1.30953,  1.30365,  1.29781,  1.29201,
                          1.28641,  1.28067,  1.27498,  1.26931,  1.26368,
                          1.25808,  1.25252,  1.24699,  1.24149,  1.23602,
                          1.23059,  1.22519,  1.21983,  1.21449,  1.20920,
                          1.15304,  1.14810,  1.14319,  1.13831,  1.13346,
                          1.12864,  1.12385,  1.11918,  1.11444,  1.10973,
                          1.10505,  1.10039,  1.09576,  1.09115,  1.08657,
                          1.08202,  1.07749,  1.07299,  1.06851,  1.06407,
                          1.05964,  1.05525,  1.05095,  1.04661,  1.04228,
                          1.03798,  1.03371,  1.02946,  1.02523,  1.02102,
                          1.01684,  1.01268,  1.00855,  1.00444,  1.00035,
                           .99629,   .99225,   .98829,   .98430,   .98032,
                           .97637,   .97244,   .96853,   .96464,   .96077,
                           .95692,   .95310,   .94929,   .94551,   .94175,
                           .93801,   .93434,   .93064,   .92696,   .92330,
                           .91965,   .91603,   .91243,   .90884,   .90528,
                           .90173,   .89821,   .89470,   .89121,   .88774,
                           .88434,   .88091,   .87749,   .87410,   .87072,
                           .86736,   .83157,   .82842,   .82528,   .82216,
                           .81905,   .81596,   .81288,   .80982,   .80678,
                           .80375,   .80074,   .79775,   .79480,   .79183,
                           .78888,   .78595,   .78303,   .78012,   .77723,
                           .77435,   .77149,   .76864,   .76581,   .76299,
                           .76019,   .75743,   .75465,   .75189,   .74914,
                           .74098,   .73828,   .74098,   .73828,   .73560,
                           .73293,   .73028,   .72764,   .72501,   .72242,
                           .71982,   .71723,   .71466,   .71209,   .70954,
                           .70700,   .70447,   .70196,   .69946,   .69697,
                           .69449,   .69203,   .68959,   .68715,   .68472,
                           .68231,   .67990,   .67750,   .67512,   .67275,
                           .67039,   .66804,   .66570,   .66338,   .66108,
                           .65877,   .65648,   .65420,   .65193,   .64966,
                           .64741,   .64517,   .64294,   .64073,   .63852,
                           .63632,   .63413,   .63197,   .62980,   .62765,
                           .62550,   .62336,   .62124,   .61912,   .61701,
                           .61491,   .61283,   .61075,   .60868,   .60663,
                           .60459,   .60255,   .60051,   .59849,   .59648,
                           .59448,   .59248,   .59050,   .58852,   .58656,
                           .58460,   .58266,   .58072,   .57879,   .57687,
                           .57495,   .57305,   .57115,   .56926,   .56738,
                           .56551,   .56365,   .56179,   .55995,   .55812,
                           .55629,   .55446,   .55265,   .55084,   .54904,
                           .54725,   .54547,   .54370,   .54193,   .54017,
                           .53843,   .53669,   .53495,   .53322,   .53150,
                           .52979,   .52808,   .52638,   .52469,   .52300,
                           .52133,   .51966,   .51800,   .51635,   .51470,
                           .51306,   .51142,   .50979,   .50817,   .50656,
                           .50495,   .50335,   .50176,   .50017,   .49860,
                           .49703,   .49546,   .49390,   .49234,   .49080,
                           .48926,   .48772,   .48619,   .48467,   .48315,
                           .48165,   .48015,   .47865,   .47716,   .47568,
                           .47420,   .47272,   .47126,   .46980,   .46834,
                           .46689,   .46545,   .46402,   .46259,   .46116,
                           .45974,   .45833,   .45692,   .45551,   .45412,
                           .45272,   .45134,   .44995,   .44858,   .44722,
                           .44585,   .44449,   .44314,   .44179,   .44044,
                           .42600,   .42472,   .42344,   .42216,   .42089,
                           .41963,   .41837,   .41712,   .41587,   .41462,
                           .41338,   .41214,   .41091,   .40968,   .40846,
                           .40724,   .40602,   .40481,   .40361,   .40241,
                           .40121,   .40002,   .39883,   .39764,   .39646,
                           .38374,   .38261,   .38147,   .38035,   .37922,
                           .37811,   .37700,   .37588,   .37477,   .37367,
                           .37257,   .37147,   .37037,   .36928,   .36819])
    
    #need to fix
    bondGuess = 1    
    return bondGuess

    
def final_script(xActual,zActual,sigmaGuess,deltaRho,nReload,
                 bondGuess=0.05,nPoints=700,L=100):
    
    #re-shift coordinates for true apex
    
    xOffset = -(min(xActual) + max(xActual))
    zOffset = -(min(zActual))
    
    xActual = xOffset + xActual
    zActual = zOffset + zActual
              
    #splitting data at apex into left and right side
    xDataLeft,zDataLeft,xDataRight,zDataRight = split_data(xActual,zActual)
    # initial guesses to start rountine 
    r0Guess = (bondGuess*sigmaGuess/(deltaRho*9.81))**0.5
    initGuess=[sigmaGuess,r0Guess]

    # calling out optimization routine with reload
    for i in range(nReload):
        r=optimize.minimize(objective_fun_v2,initGuess,args=(deltaRho,
                            xDataLeft,zDataLeft,xDataRight,zDataRight,xOffset,
                            zOffset),method='Nelder-Mead',tol=1e-9)
        initGuess=[r.x[0],r.x[1]]
        sigmaFinal=r.x[0]
        r0Final=r.x[1]
        bondFinal=deltaRho*9.81*r0Final**2/sigmaFinal

        xFit,zFit=young_laplace(bondFinal,nPoints,L)
        
        xFit = xOffset + xFit
        zFit = zOffset + zFit
        
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
    testObjFunV2 = True
    # importing test data for analysis
    testData = True
    # vizualization of objective function as a surface plot
    viewObjFunSurf = False
    
    if testObjFunV2 or testData:
        # Generate test data for objective functions
        sigma = 0.06
        r0 = .0015
        deltaRho = 998
        L = 3.5
        nPoints = 1000
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
