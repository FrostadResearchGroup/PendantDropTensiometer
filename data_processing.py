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


#set equations
def ode_system(t,y,bond):
    phi, r, z = y
    f = [2-bond*(z)-np.sin(phi)/(r), np.cos(phi), np.sin(phi)]
    return f
    
#define ode solver function
def young_laplace(Bo,nPoints,L):
    """
    Bo = float - Bond number
    nPoints = int - number of integration points desired
    L = float - final arc length for range of integration
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
    s0 = -L
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
        
    return sol

def objective_fun(params,deltaRho,xCoords,zCoords):
    gamma = params[0]
    R0 = params[1]
    #takes in initial bond number and optimizes through objective function as 
    #defined by N. Alvarez et. al
    bond= deltaRho*9.81*R0**2/gamma
    # print outputs of each section
    model=young_laplace(bond,400,2)
    
    xModel=model[:,1]*R0
    zModel=model[:,2]*R0
    xModel_App=np.append(list(reversed(-xModel)),xModel[1:])
    zModel_App=np.append(list(reversed(zModel)),zModel[1:])
     
    xModelFit=xModel_App[:-1]
    zModelFit=zModel_App[:-1]
    xModelFit_plus=xModel_App[1:]
    zModelFit_plus=zModel_App[1:]
    
    xData=xCoords
    zData=zCoords
    
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
    
    output=np.power(((a-b)/np.power((c+d),0.5)),2)
    
    rsq=np.min(output,axis=0)
    
    return np.sum(rsq)
    
def line(x,m,b):
    y = m*x + b
    return y
    
def test_obj_fun(params,x_data,y_data):
    m = params[0]
    b = params[1]
    y_fit = line(x_data,m,b)
    error = np.sum((y_fit-y_data)**2)
    return error

if __name__ == "__main__":
    
    plt.close('all')
    
    testYLP = False
    testLine = True
    
    if testLine:
        # Generate test data
        N = 100
        m_data = -2.321
        b_data = 4.567
        guess = [0,0]
        x_data = np.linspace(0,10,N)
        y_data = line(x_data,m_data,b_data)
        
        output = optimize.minimize(test_obj_fun,guess,args=(x_data,y_data),
                                   method='Nelder-Mead')
        m_fit = output.x[0]
        b_fit = output.x[1]
        
        print m_fit, b_fit
    
    if testYLP:
    #### Acutal Data Points 
        sigmaActual=0.073
        r0_actual=.003
        deltaRho=900
        Bond_actual=deltaRho*9.81*r0_actual**2/sigmaActual
        
        temp = young_laplace(Bond_actual,50,1)
        xActual = temp[:,1]*r0_actual
        zActual = temp[:,2]*r0_actual
        
        xActual_App=np.append(list(reversed(-xActual)),xActual[1:])
        zActual_App=np.append(list(reversed(zActual)),zActual[1:])
        ################################################################  
        
        sigmaGuess=.05
        R0Guess=.002
        
        initGuess=[sigmaGuess,R0Guess]
        
        for i in range(3):
            r=optimize.minimize(objective_fun,initGuess,args=(deltaRho,xActual_App,
                                  zActual_App),method='Nelder-Mead')
            initGuess=[r.x[0],r.x[1]]
            print(initGuess)
        
            sigmaFinal=r.x[0]
            r0Final=r.x[1]
            Bond_final=deltaRho*9.81*r0Final**2/sigmaFinal
            
            fitted=young_laplace(Bond_final,50,1)
            
            xCurveFit=fitted[:,1]*r0Final
            zCurveFit=fitted[:,2]*r0Final
            xCurveFit_App=np.append(list(reversed(-xCurveFit)),xCurveFit[1:])
            zCurveFit_App=np.append(list(reversed(zCurveFit)),zCurveFit[1:])
        
        
            plt.plot(xActual_App,zActual_App,'ro')
            plt.axis('equal')
            plt.plot(xCurveFit_App,zCurveFit_App,'b')
            
