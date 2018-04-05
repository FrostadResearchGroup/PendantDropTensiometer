#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 18:26:21 2017
@author: vanessaebogan
"""

# Custom modules
import image_processing as ip
import image_extraction as ie

# Python modules
import numpy as np
from scipy import optimize
from scipy.integrate import ode
from scipy.integrate import odeint

import matplotlib.pyplot as plt
import time

def ode_system(y,t,bond):
    """
    Outputs system of ordinary differential equations with non-dimensionalized
    coordinates --> f = [d(fi)/ds , d(X)/ds , d(Z)/ds].
    
    t    = float - range of integration vector
    y    = float - desired solution to ODE system [fiVec, xVec, zVec]
    bond = float - Bond number
    """
    phi, r, z = y
    f = [2-bond*(z)-np.sin(phi)/(r), np.cos(phi), np.sin(phi),np.pi*r**2*np.sin(phi)]
    return f

def young_laplace(Bo,nPoints,L):
    """
    Runs through YL equation.
    
    Bo      = float - Bond number
    nPoints = int - number of integration points desired
    L       = float - final arc length for range of integration
    """
    
    #integration range and number of integration points
    s1=L
    N=nPoints 
    
    #set initial values
    s0 = 0
    y0 = [0.00001,0.00001,0.00001]
    
    sVec = np.linspace(s0,s1,N) 
    bond=Bo
   
    sol = odeint(ode_system,y0,sVec,args=(bond,))

    r = sol[:,1]
    z = sol[:,2]
    fi = sol[:,0]
        
    return r,z,fi
    
    
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
    
    Rotates data for optimization of theta parameter.

    xActualLeft =  float - x coordinates of droplet (left side)
    zActualLeft =  float - z coordinates of droplet (left side)
    xActualRight = float - x coordinates of droplet (right side)
    zActualLeft =  float - z coordinates of droplet (left side)
    
    """
    
    #rotate data points with theta parameter
    xActualLeft = xActualLeft*np.cos(thetaVal) - zActualLeft*np.sin(thetaVal)
    zActualLeft = xActualLeft*np.sin(thetaVal) + zActualLeft*np.cos(thetaVal)
    
    xActualRight = xActualRight*np.cos(thetaVal) - zActualRight*np.sin(thetaVal)
    zActualRight = xActualRight*np.sin(thetaVal) + zActualRight*np.cos(thetaVal)
    
    return xActualLeft,zActualLeft,xActualRight,zActualRight

def objective_fun_v2(params,deltaRho,xDataLeft,zDataLeft,xDataRight,
                     zDataRight,sFinal,trueRotation,numberPoints=1000):
    """
    Calculates the sum of residual error squared between x data and 
    theorectical points through a comparison of z coordinates.
    
    params       = float - fitting parameters for optimization rountine
    deltaRho     = float - density differential between fluids
    xDataLeft    = float - x coordinates of droplet (left side)
    zDataLeft    = float - z coordinates of droplet (left side)
    xDataRight   = float - x coordinates of droplet (right side)
    zDataLeft    = float - z coordinates of droplet (left side)
    sFinal       = float - integration range for ODE solver
    trueRotation = 
    
    """
    #building relationship from params to bond number
    gamma = params[0]
    apexRadius = params[1]
    bond = deltaRho*9.81*apexRadius**2/gamma
    
    #throwing bond number into curve/coordinate generator
    xModel,zModel,fiModel = young_laplace(bond,numberPoints,sFinal)


    #x and z coordinates with arc length
    xModel = xModel*apexRadius
    zModel = zModel*apexRadius
    
    #rotate data points with theta parameter
    xDataLeft,zDataLeft,xDataRight,zDataRight = rotate_data(xDataLeft,zDataLeft,xDataRight,zDataRight,trueRotation)
    
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
    xeAvg = (xeLeft+xeRight)/2
    
    r0Guess = abs(xeRight)
    xeScal = xeAvg
    
    #looking for Xs    
    indicesLeft = np.argmin(abs(zDataLeft-2*xeLeft))    
    indicesRight = np.argmin(abs(zDataRight-2*xeRight))
        
    indexLeft = int(np.average(indicesLeft))
    indexRight = int(np.average(indicesRight))

    xsLeft = abs(xDataLeft[indexLeft])
    xsRight = abs(xDataRight[indexRight])
    
    #averaging left and right values
    sLeft = xsLeft/xeLeft
    sRight = xsRight/xeRight
    sAvg = (sLeft+sRight)/2        

    return sAvg,xeScal,r0Guess
    
def s_interp(sAvg,xeAvg,deltaP):
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
        hInv = (.31345/sAvg**2.64267) - (.09155*sAvg**2)+(.14701*sAvg)-(.05877)
    elif sAvg >= .59:
        hInv = (.31522/sAvg**2.62435) - (.11714*sAvg**2)+(.15756*sAvg)-(.05285)
    elif sAvg >= .46:
        hInv = (.31968/sAvg**2.59725) - (.46898*sAvg**2)+(.50059*sAvg)-(.13261);
    elif sAvg >= .401:
        hInv = (.32720/sAvg**2.56651) - (.97553*sAvg**2)+(.84059*sAvg)-(.18069);
    else:
        print('Shape is too spherical');
        #Use formula for S > 0.401 even though it is wrong
        hInv = (.32720/sAvg**2.56651) - (.97553*sAvg**2)+(.84059*sAvg)-(.18069);
        
    surfTenGuess = deltaP*9.81*(2*xeAvg)**2*hInv;

    return surfTenGuess

  
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
                    + (abs(zActualLeft[1:])-abs(zActualLeft[:-1]))**2)**0.5/r0Guess
    
     
    arcDistRight = ((abs(xActualRight[1:])-abs(xActualRight[:-1]))**2
                    + (abs(zActualRight[1:])-abs(zActualRight[:-1]))**2)**0.5/r0Guess
    
    # creating a summated arclength vector 
    sumArcLeft = 1.5*np.sum(arcDistLeft)/r0Guess
    sumArcRight = 1.5*np.sum(arcDistRight)/r0Guess
     
    # return largest s value
    if sumArcLeft > sumArcRight:            
        return sumArcLeft,arcDistLeft
    else:
        return sumArcRight,arcDistRight
    

  
def optimize_params(xActual,zActual,sigmaGuess,r0Guess,deltaRho,nReload,trueRotation,
                 nPoints=1000):
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
    initGuess = [sigmaGuess,r0Guess]

    intRange,arcLength = get_data_arc_len(xDataLeft,zDataLeft,xDataRight,zDataRight,r0Guess)
    
    # calling out optimization routine with reload
    for i in range(nReload):
        r=optimize.minimize(objective_fun_v2,initGuess,args=(deltaRho,
                            xDataLeft,zDataLeft,xDataRight,zDataRight,intRange,trueRotation),
                            method='Nelder-Mead',tol=1e-9)
        initGuess = [r.x[0],r.x[1]]
        sigmaFinal = r.x[0]
        r0Final = r.x[1] * np.cos(trueRotation)
        bondFinal=deltaRho*9.81*r0Final**2/sigmaFinal
        
        intRangeFinal,arcLength = get_data_arc_len(xDataLeft,zDataLeft,xDataRight,zDataRight,r0Final)
        xFit,zFit,fiFit = young_laplace(bondFinal,nPoints,intRangeFinal)
        
        xActual = xActual*np.cos(trueRotation) - zActual*np.sin(trueRotation)
        zActual = xActual*np.sin(trueRotation) + zActual*np.cos(trueRotation)    
        

    return sigmaFinal,r0Final,bondFinal
    
    
def get_drop_volume(xActual,zActual,r0):
    """
    Determines volume of droplet.
    """
    volVec = np.abs(np.pi*xActual[1:]**2*(zActual[1:]-zActual[:-1]))/2

    dropletVolume = np.sum(volVec)

    return dropletVolume
    
def get_drop_SA(xActual,zActual,r0):
    """
    Determines surface area of droplet.
    """
    
    surfAreaVec = np.abs(2*np.pi*xActual[1:]*(zActual[1:]-zActual[:-1]))/2
    
    surfArea = np.sum(surfAreaVec)
    
    return surfArea

def get_volume_error(dropVolume,coeffThermalExpansion,
                                               magnificationRatio,deltaT):
    """
    Creates an error measurement of volume based off resolution uncertainty 
    and thermal expansion.
    """
    
    #define uncertainty associated with pixelation and error from temp. fluctuations
    resUncert = (magnificationRatio/2)
    tempFluct = coeffThermalExpansion*deltaT*dropVolume*10**9
    
    #define total error
    totalError = resUncert**3+tempFluct

    return totalError   

def get_droplet_profile(image,capillaryImage,capillaryDiameter):
    """
    Maps out droplet profile.
    """
   
    
    #binarize image
    binarizedImage = ip.binarize_image(image)

    #get interface coordinates
    interfaceCoordinates = ip.get_interface_coordinates(binarizedImage)

    interfaceCoordinates = np.array(interfaceCoordinates)
    #flip the coordinates vertically
    interfaceCoordinates *= [1,-1]

    #offset vertical points     
    zOffset = -min(interfaceCoordinates[:,1])
    interfaceCoordinates = interfaceCoordinates + [0,zOffset]
#    plt.plot(interfaceCoordinates[:,0],interfaceCoordinates[:,1])
    # Process capillary image    
    capillaryRotation,zCoords,pixelsConv = ip.get_capillary_rotation(capillaryImage,zOffset)        
      
    #isolate drop
    xCoords = [min(interfaceCoordinates[:,0]),max(interfaceCoordinates[:,0])] 
    dropCoords = ip.isolate_drop(xCoords,zCoords,interfaceCoordinates)
#    plt.plot(dropCoords[:,0],dropCoords[:,1])
  
    #get magnification ratio
    magRatio = ip.get_magnification_ratio(pixelsConv, capillaryDiameter,
                                          capillaryRotation)  
    
    #shift coordinates so apex is at 0,0
    newCenter = [0,0]
    shiftedCoords = ip.shift_coords(dropCoords[:,0],dropCoords[:,1],newCenter)

    #scale drop
    scaledCoords = ip.scale_drop(shiftedCoords,magRatio)
    
    #reorder data points and estimate surface tension using 5 point method
    xData,zData = ip.reorder_data(scaledCoords)
    
    dropProfile = np.vstack((xData,zData))
    
    return dropProfile

def get_surf_tension(dropImage, capillaryImage, deltaRho, capillaryDiameter, 
                     numMethod, trueSyringeRotation, reloads, tempFluct, 
                     thermalExpCoeff):
    """
    
    Get the surface tension from the droplet and capillary images and all of 
    the user inputs.
    
    """

    #binarize image
    binarizedImage = ip.binarize_image(dropImage)
    #return nans if black image
    if np.all(binarizedImage == 255):
        surfTen  = np.nan
        dropVol  = np.nan
        volError = np.nan
        bondNumber = np.nan
        worthNum = np.nan
        surfArea = np.nan

    else:
        #get interface coordinates
        interfaceCoordinates = ip.get_interface_coordinates(binarizedImage)
    
        interfaceCoordinates = np.array(interfaceCoordinates)
        #flip the coordinates vertically
        interfaceCoordinates *= [1,-1]
    
        #offset vertical points     
        zOffset = -min(interfaceCoordinates[:,1])
        interfaceCoordinates = interfaceCoordinates + [0,zOffset]
#        plt.plot(interfaceCoordinates[:,0],interfaceCoordinates[:,1])
        # Process capillary image    
        capillaryRotation,zCoords,pixelsConv = ip.get_capillary_rotation(capillaryImage,zOffset)        
          
        #isolate drop
        xCoords = [min(interfaceCoordinates[:,0]),max(interfaceCoordinates[:,0])] 
        dropCoords = ip.isolate_drop(xCoords,zCoords,interfaceCoordinates)
#        plt.plot(dropCoords[:,0],dropCoords[:,1])
      
        #get magnification ratio
        magRatio = ip.get_magnification_ratio(pixelsConv, capillaryDiameter,
                                              capillaryRotation)  
        
        #shift coordinates so apex is at 0,0
        newCenter = [0,0]
        shiftedCoords = ip.shift_coords(dropCoords[:,0],dropCoords[:,1],newCenter)
    
        #scale drop
        scaledCoords = ip.scale_drop(shiftedCoords,magRatio)
        
        #reorder data points and estimate surface tension using 5 point method
        xData,zData = ip.reorder_data(scaledCoords)
        s,xe,apexRadiusGuess = bond_calc(xData,zData)
        surfTen = s_interp(s,xe,deltaRho)
        bondNumber = deltaRho*9.81*apexRadiusGuess**2/surfTen
        
        dropVol = get_drop_volume(xData,zData,apexRadiusGuess)
        surfArea = get_drop_SA(xData,zData,apexRadiusGuess)
        
        volError = get_volume_error(dropVol,thermalExpCoeff,magRatio,tempFluct)        
        
        if numMethod == 2: # use all points
            #run through optimization routine
            surfTen,apexRadius,bondNumber = optimize_params(xData,zData,
                                                            surfTen,
                                                            apexRadiusGuess,
                                                            deltaRho,
                                                            reloads,
                                                            trueSyringeRotation)
        
        worthNum = deltaRho*9.81*dropVol/(np.pi*surfTen*capillaryDiameter*10**-3)
        
    return surfTen,apexRadius
 
    
##############################################################################

def ode_system_elastic(y,t,pressure,elastMod,poissRatio,deltaRho,hoopStrech,
                       merStrech,surfTenRef):
    
    """
    
    Outputs system of ordinary differential equations with non-dimensionalized
    coordinates --> f = [d(fi)/ds , d(X)/ds , d(Z)/ds, d(TauMer)/ds].
    
    """
    
    phi,r,z,tauMer = y
    
    tauHoop = elastMod*((hoopStrech-1+poissRatio*(merStrech-1)))/(merStrech*(1-poissRatio))+surfTenRef
    
    f = [merStrech*(pressure-deltaRho*9.81*z-tauHoop*np.sin(phi)/r)/tauMer,
         merStrech*np.cos(phi),
         merStrech*np.sin(phi),
         -merStrech*np.cos(phi)*(tauMer-tauHoop)/r]
    
    return f

def elastic_model(elastMod,poissRatio,pressure,nPoints,L,deltaRho,hoopStrech,merStrech,surfTenRef):
    """
    
    Runs through elastic model equation based off variation of YL equation.
    
    """

    #integration range and number of integration points
    s1=L
    N=nPoints 
    
    #set initial values
    s0 = 0
    y0 = [0.00001,0.00001,0.00001,0.000001]
    
    sVec = np.linspace(s0,s1,N) 
   
    sol = odeint(ode_system_elastic,y0,sVec,args=(pressure,elastMod,
                        poissRatio,deltaRho,hoopStrech,
                        merStrech,surfTenRef,),mxstep=500000)

    r = sol[:,1]
    z = sol[:,2]
    fi = sol[:,0]
    tauMer = sol[:,3]
        
    return r,z,fi,tauMer



def get_midpoint(coords):
    """
    
    Returns z-value midpoint of droplet shape.
    
    """
    
    zMin = np.min(coords[:,1])
    zMax = np.max(coords[:,1])
    
    zAvg = np.average([zMin,zMax])
    
    midPointIndex = np.argmin(np.abs(coords[:,1]-zAvg))
    
    return zAvg,midPointIndex


def get_streches(refShapeCoords,defShapeCoords,apexRad):
    """
    Returns meridional and hoop streches at droplet midpoint with respect 
    to the reference shape.
    """
    
    #get the z value midpoint and index  
    zAvgRef,refMidpointIndex = get_midpoint(refShapeCoords)
    zAvgDef,defMidpointIndex = get_midpoint(defShapeCoords)
    
    #split data for feeding into get_data_arc_len function
    xRefLeft,xRefRight,zRefLeft,zRefRight = split_data(refShapeCoords[:,0],refShapeCoords[:,1])
    xDefLeft,xDefRight,zDefLeft,zDefRight = split_data(defShapeCoords[:,0],defShapeCoords[:,1])
    
    #get the arc length at midpoint (scaling factor is trivial)    
    sumArcLenRef,refArcLenMP = get_data_arc_len(xRefLeft,xRefRight,zRefLeft,zRefRight,apexRad)
    sumArcLenDef,defArcLenMP = get_data_arc_len(xDefLeft,xDefRight,zDefLeft,zDefRight,apexRad)
    
    #determine arc length values at droplet midpoint
    defArcLenMP = np.sum(defShapeCoords[:defMidpointIndex])
    refArcLenMP = np.sum(refShapeCoords[:refMidpointIndex])
    
    #determine r values at droplet midpoint
    rValRef = refShapeCoords[:,0][refMidpointIndex]
    rValDef = defShapeCoords[:,0][defMidpointIndex]
    
    #calculate strech values
    merStrech  = defArcLenMP/refArcLenMP
    hoopStrech = rValDef/rValRef
    
    return merStrech,hoopStrech

    
def objective_fun_v2_elastic(params,deltaRho,hoopStrech,merStrech,surfTenRef,
                     xDataLeft,zDataLeft,xDataRight,zDataRight,sFinal,
                     trueRotation,apexRadius,numberPoints=1000):
    """
    Calculates the sum of residual error squared between x data and 
    theorectical points through a comparison of z coordinates.
    
    params   = float - fitting parameters for optimization rountine
    deltaRho = float - density differential between fluids
    xData    = float - x coordinates of droplet 
    zData    = float - z coordinates of droplet
    """
    #building relationship from params to bond number
    elastMod = params[0]
    poissRatio = params[1]
    pressure = params[2]
    
    #throwing bond number into curve/coordinate generator
    xModel,zModel,fiModel,tauMerModel = elastic_model(elastMod,poissRatio,pressure,
                                                      numberPoints,sFinal,deltaRho,
                                                      hoopStrech,merStrech,surfTenRef)

    #x and z coordinates with arc length
    xModel = xModel*apexRadius
    zModel = zModel*apexRadius
    
    #rotate data points with theta parameter
    xDataLeft,zDataLeft,xDataRight,zDataRight = rotate_data(xDataLeft,zDataLeft,xDataRight,zDataRight,trueRotation)
    
    #creates tiling matrices for subsequent indexing of data to model comparison
    zDatagridLeft,zModelgridLeft,zDatagridRight,zModelgridRight = tiling_matrix(zDataLeft,zDataRight,zModel)
    
    #indexes closest value from model to data    
    indexLeft,indexRight = indexing(zDatagridLeft,zDatagridRight,zModelgridLeft,zModelgridRight)
    
    #building r squared term
    rxLeft=xModel[indexLeft]+xDataLeft
    rxRight=xModel[indexRight]-xDataRight
    
    #returning residual sum of squares
    rsq=np.sum(rxLeft**2)+np.sum(rxRight**2)
    print(rsq)
    return rsq    
    
def optimize_params_elastic(defCoords,elastModGuess,poissRatioGuess,pressGuess,r0,
                            surfTenRef,hoopStrech,merStrech,deltaRho,nReload,
                            trueRotation,nPoints=1000):
    
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
    xDataLeft,zDataLeft,xDataRight,zDataRight = split_data(defCoords[:,0],defCoords[:,1])
    
    # initial guesses to start rountine
    initGuess = [elastModGuess,poissRatioGuess,pressGuess]

    intRange,arcLength = get_data_arc_len(xDataLeft,zDataLeft,xDataRight,zDataRight,r0)
    
    # calling out optimization routine with reload
    for i in range(nReload):
        r=optimize.minimize(objective_fun_v2_elastic,initGuess,args=(deltaRho,
                            hoopStrech,merStrech,surfTenRef,xDataLeft,zDataLeft,
                            xDataRight,zDataRight,intRange,trueRotation,r0),
                            method='Nelder-Mead',tol=1e-9)
    
        initGuess = [r.x[0],r.x[1],r.x[2]]
        y2DFinal = r.x[0]
        v2DFinal = r.x[1] 
        pressureFinal = r.x[2]
        
        intRangeFinal,arcLength = get_data_arc_len(xDataLeft,zDataLeft,xDataRight,zDataRight,r0)
        xModel,zModel,fiModel,tauMerModel = elastic_model(y2DFinal,v2DFinal,pressureFinal,
                                                      nPoints,intRangeFinal,deltaRho,
                                                      hoopStrech,merStrech,surfTenRef)
           
        

    return xModel,zModel,y2DFinal,v2DFinal,pressureFinal


def get_elastic_properties(refImage,defImage,capillaryImage, deltaRho, 
                           capillaryDiameter, numMethod, trueSyringeRotation, 
                           reloads):
    """
    
    Determines the elastic properties of membrane through shape analysis.
    
    """
    
    #obtain droplet profile of reference shape
    refProfile = get_droplet_profile(refImage,capillaryImage,capillaryDiameter)
    
    #obtain droplet profile of deformed shape
    defProfile = get_droplet_profile(defImage,capillaryImage,capillaryDiameter)
    
#    plt.plot(defProfile[:,0],defProfile[:,1])
    #obtain surface tension of reference shape
    refSurfTen,r0 = get_surf_tension(refImage, capillaryImage, deltaRho, capillaryDiameter, 
                     numMethod, trueSyringeRotation, reloads, tempFluct, 
                     thermalExpCoeff)
    

    #obtain strech values
    strechMer,strechHoop = get_streches(refProfile,defProfile,r0)
    
    #initial guesses
    y2DGuess   = 0.05 #mN
    v2DGuess   = 0.5  
    pressGuess = 101*10**3 #pascals
    
    #throw into optimize_params_elastic
    xFit,zFit,y2D,v2D,pFinal = optimize_params_elastic(defProfile,y2DGuess,
                                        v2DGuess,pressGuess,r0,refSurfTen,
                                        deltaRho,strechHoop,strechMer,deltaRho,
                                        reloads,trueSyringeRotation)
    
    
        
    return xFit,zFit,y2D,v2D,pFinal
    
######################## For Testing Purposes #################################   
   
if __name__ == "__main__":

    #working directory name
    dirName = 'C:/Research/Pendant Drop/Data/Biofilm Testing/Positive Contol Test (PA 14)/48 Hour Tests/Test 4/Droplet Compression/'

    #reference pictures from Hegemann et. al 
    referenceImage = dirName + 'Reference_Image.jpg'
    deformedImage  = dirName + 'Compressed_Image_1.jpg'
    capillaryImage = dirName + 'Capillary_Image.jpg'
    
    #load images with image extraction script
    capillaryImage = ie.load_image_file(capillaryImage)    
    referenceImage = ie.load_image_file(referenceImage)
    deformedImage = ie.load_image_file(deformedImage)
    
    #geometric and fluid properties 
    deltaRho = 998 #kg/m3
    capDiameter = 2.10 #m
    tempFluct = 2 #K
    thermalExpCoeff = 0.0002 #1/K
    
    numMethod = 2
    syrRotation = 0
    reloads = 1
    
    
    xFit,zFit,y2D,v2D,pressure = get_elastic_properties(referenceImage,deformedImage,
                                         capillaryImage,deltaRho,capDiameter,
                                         numMethod,syrRotation,reloads)


    # Define plotting text features 
    titleFont = {'family': 'serif',
    'color':  'darkred',
    'weight': 'bold',
    'size': 15,
    }
    
    axesFont = {'weight': 'bold'}
     
    fig,ax1 = plt.subplots()
    
    plt1 = ax1.plot(xFit,zFit,'bo')
    
    ax1.set_title('Compressed Droplet Profile',titleFont)
    ax1.set_xlabel('length scale (mm)',fontdict=axesFont)
    ax1.set_ylabel('length scale (mm)',fontdict=axesFont)
