# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 12:52:54 2020

@author: Trevor
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, leastsq

def remove_first_orbit(lc, t, lc_err, time):
    for i in range(len(t)):
        delta=t[i+1]-t[i]
        if delta > 2.* (t[1]-t[0]):
            break
    i+=1
    lc=lc[i:]
    lc_err=lc_err[i:]
    t=t[i:]
    time=time[i:]

    return lc, t, lc_err, time

def remove_first_exposures(lc, t, lc_err, time):
    firsts = [0]
    
    for i in range(len(t)-1):
        delta=t[i+1]-t[i]
        if delta > 2.* (t[1]-t[0]):
            firsts = np.append(firsts, i+1)
    print('First exposures =', firsts)
    lc = np.delete(lc, firsts)
    lc_err = np.delete(lc_err, firsts)
    t = np.delete(t, firsts)
    time = np.delete(time, firsts)

    return lc, t, lc_err, time

def lin_trend(t, m, b):
    return m*t + b

def guess_lin_coeffs(t, flux):
    lin_params, params_covariance = curve_fit(lin_trend, t, flux)
    plt.figure()
    plt.plot(t, flux, 'bo')
    plt.plot(t, lin_trend(t, lin_params[0], lin_params[1]), 'r-', 
             label='fitted function')
    plt.xlabel('time since start of obs (days)')
    plt.ylabel('Normalized Flux')
    plt.legend()
    plt.show()
    
    m = lin_params[0]
    b = lin_params[1]
    print('m =', m)
    print('b =', b)
    
    return m, b    

def hook_trend(HSTphase, S, phi, C):
    return 1.0 - np.exp(-S*(HSTphase+phi)) + C*HSTphase

def guess_hook_coeffs(HSTphase, flux):

    hook_params, h_params_covariance = curve_fit(hook_trend, HSTphase, flux)

    plt.figure()
    plt.plot(HSTphase, flux, 'bo')
    plt.plot(HSTphase, hook_trend(HSTphase, hook_params[0], hook_params[1], 
                                  hook_params[2]), 'ro', 
        label='fitted function')
    plt.xlabel('HST Phase')
    plt.ylabel('Normalized Flux')
    plt.legend()
    plt.show()
    
    S = hook_params[0]
    phi = hook_params[1]
    C = hook_params[2]
    
    print('S =', S)
    print('phi =', phi)
    print('C =', C)
    
    return S, phi, C
    

'''
def hook_trend(HSTphase, A, S, phi, C):
    return 1.0 - A*np.exp(-S*(HSTphase+phi)) + C*HSTphase

def guess_hook_coeffs(HSTphase, flux):

    hook_params, h_params_covariance = curve_fit(hook_trend, HSTphase, flux)

    plt.figure()
    plt.plot(HSTphase, flux, 'bo')
    plt.plot(HSTphase, hook_trend(HSTphase, hook_params[0], hook_params[1], 
                                  hook_params[2], hook_params[3]), 'ro', 
        label='fitted function')
    plt.xlabel('HST Phase')
    plt.ylabel('Flux (e$^-$)')
    plt.legend()
    plt.show()
    
    A = hook_params[0]
    S = hook_params[1]
    phi = hook_params[2]
    C = hook_params[3]
    
    print('A =', A)
    print('S =', S)
    print('phi =', phi)
    print('C =', C)
    
    return A, S, phi, C
'''