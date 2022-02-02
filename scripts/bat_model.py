# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 08:11:37 2020

@author: Trevor
"""

import numpy as np
import matplotlib.pyplot as plt
import batman
from scipy.optimize import leastsq


#intialize a transit model
def initialize_model(t, t0, per, rp, a, inc, ecc, w, u, limb_dark, 
                     transit_type, fp, t_secondary):
    
    params = batman.TransitParams()
    params.t0 = t0
    params.per = per
    params.rp = rp
    params.a = a
    params.inc = inc
    params.ecc = ecc
    params.w = w
    params.u = u
    params.limb_dark = limb_dark
    if transit_type == "secondary":
        params.fp = fp
        params.t_secondary = t_secondary

    init_bat_model = batman.TransitModel(params, t, transittype = transit_type)

    return params, init_bat_model		#return parameters and model objects 

def modify_model(theta, init_bat_model, params, t, HSTphase, t_secondary, 
                 fixed_t_sec):
    
    m=theta[0]
    b=theta[1]
    S=theta[2]
    phi=theta[3]
    C =theta[4]
    params.fp = theta[5]
    if fixed_t_sec == True:
        params.t_secondary = t_secondary
    else:
        params.t_secondary = theta[6]
    
    model_lc = init_bat_model.light_curve(params)
    hookfit = 1.0 - np.exp(-S*(HSTphase+phi)) + C*HSTphase
    linearfit = m*t + b
    
    full_bat_model = model_lc*linearfit*hookfit
    
    return full_bat_model, model_lc, linearfit, hookfit

def residuals(theta, init_bat_model, params, t, HSTphase, flux, flux_err,
             t_secondary, fixed_t_sec):
    
    full_bat_model, model_lc, linearfit, hookfit = modify_model(theta, 
                                        init_bat_model, params, t, HSTphase,
                                        t_secondary, fixed_t_sec)
    residuals = (flux-full_bat_model)/flux_err
    
    return residuals

def fit_model(theta_guess, init_bat_model, params, t, HSTphase, flux, 
               flux_err, t_secondary, fixed_t_sec):
    
    fit_theta, pcov, infodict, errmsg, success =leastsq(residuals, theta_guess, 
                                                    args=(init_bat_model, 
                                                          params, t, 
                                                          HSTphase, flux, 
                                                          flux_err, 
                                                          t_secondary, 
                                                          fixed_t_sec),
                                                          full_output=1)

    rchi2 = (residuals(fit_theta, init_bat_model, params, t, HSTphase, 
                      flux, flux_err, t_secondary, fixed_t_sec)**2
                        ).sum()/(len(flux)-len(theta_guess))
    
    '''
    pcov = pcov * rchi2
    
    error = [] 
    for i in range(len(fit_theta)):
        try:
          error.append(np.absolute(pcov[i][i])**0.5)
        except:
          error.append( 0.00 )
    fit_err = np.array(error)
    '''
    
    print('Guess', theta_guess)
    print('Fitted', fit_theta)
    print('Diff', abs(theta_guess-fit_theta))
    #print('Fit Error', fit_err)
    
    fit_corr_bat_lc, fit_bat_lc, linearfit, hookfit = modify_model(fit_theta, 
                                            init_bat_model, params, t, 
                                                         HSTphase, t_secondary,
                                                         fixed_t_sec)
    
    res = residuals(fit_theta, init_bat_model, params, t, HSTphase, flux, 
                    flux_err, t_secondary, fixed_t_sec)
    
    plt.figure()
    plt.scatter(t, res)
    plt.xlabel('time since start of obs (days)')
    plt.ylabel('Residual')
    plt.show()
    
    chi2 = sum(res*res)
    print('Chi-square=', chi2)
    
    dof = len(flux)-len(theta_guess)
    print('Deg of freedom=', dof)
    
    print('Reduced Chi-square=', rchi2)
           
    plt.figure()
    plt.errorbar(t, flux, flux_err, fmt='o', color='b', label='obs')
    plt.plot(t, fit_corr_bat_lc, 'ro',
             label='fitted function')
    plt.xlabel('time since start of obs (days)')
    plt.ylabel('Flux')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()

    return fit_theta, fit_corr_bat_lc, fit_bat_lc, linearfit, hookfit