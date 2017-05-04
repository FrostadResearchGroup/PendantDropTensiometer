# -*- coding: utf-8 -*-
"""
Created on Mon Jan 02 15:29:44 2017

@author: Yohan
"""

import numpy as np
from scipy.integrate import ode

#set equations
def func(t,y,bond):
    phi, r, z = y
    f = [2-bond*z-np.sin(phi)/r, np.cos(phi), np.sin(phi)]
    return f
    
#define ode solver function
def odesolve(current_s,bond,N):
        
    #set solver
    solver = ode(func)
    solver.set_integrator('dopri5')

    #sets bond number to value inputted when ylp_func is called by solver
    solver.set_f_params(bond)

    #set initial values
    s0 = 0.0
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
         
