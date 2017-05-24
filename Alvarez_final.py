#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 18:26:21 2017

@author: vanessaebogan
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from numpy import loadtxt
from scipy.integrate import ode
from numpy import matlib



#set equations
def func(t,y,bond,a):
    phi, r, z = y
    f = [2-bond*(z/a)-np.sin(phi)/(r/a), np.cos(phi), np.sin(phi)]
    return f
    
#define ode solver function
def odesolve(params):
        
    #set solver
    solver = ode(func)
    solver.set_integrator('dopri5')

    current_s=10
    N=50

    bond=params[0]
    a=params[1]

    #sets bond number to value inputted when ylp_func is called by solver
    solver.set_f_params(bond,a)

    #set initial values
    s0 = -np.pi
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

def workfile(params):
   
    # print outputs of each section
    model=odesolve(params)
    
    xm=model[:,1]
    zm=model[:,2]
    xm_2=np.append(xm,-xm[1:])
    zm_2=np.append(zm,zm[1:])
     
    xmod=xm_2[:-1]
    zmod=zm_2[:-1]
    xmod_plus=xm_2[1:]
    zmod_plus=zm_2[1:]
    
    xdata=np.tile(xdatavec,(len(xmod),1))
    zdata=np.tile(zdatavec,(len(zmod),1))
    
    xmodel=np.tile(xmod,(len(xdatavec),1)).transpose()
    zmodel=np.tile(zmod,(len(zdatavec),1)).transpose()
    xmodel_plus=np.tile(xmod_plus,(len(xdatavec),1)).transpose()
    zmodel_plus=np.tile(zmod_plus,(len(zdatavec),1)).transpose()
    
    a=(zmodel-zdata)*(xmodel-xmodel_plus)
    b=(xmodel-xdata)*(zmodel-zmodel_plus)
    c=np.power((xmodel-xmodel_plus),2)
    d=np.power((zmodel-zmodel_plus),2)
    
    output=np.power(((a+b)/np.power((c+d),0.5)),2)
    
    rsq=np.min(output,axis=0)
    
    return np.sum(rsq)

    
if __name__ == "__main__":
    
    data=loadtxt("testfile_2.txt",delimiter=",")
    xvec=data[:,0]
    zvec=data[:,1]
    
    mag_ratio=0.0197
    
    xdatavec=mag_ratio*xvec
    zdatavec=mag_ratio*zvec
    
    params=[.5,.5]
    r=scipy.optimize.minimize(workfile,params,method='Nelder-Mead')
    Bond=r.x[0]
    R_0=r.x[1]
    
    model=odesolve([Bond,R_0])
    
    xm=model[:,1]
    zm=model[:,2]
    xm_2=np.append(xm,-xm[1:])
    zm_2=np.append(zm,zm[1:])
    
    plt.plot(xdatavec,zdatavec,'ro')
    plt.plot(xm_2,zm_2,'bo')
    
    
    