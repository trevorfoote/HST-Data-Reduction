# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 10:03:32 2021

@author: Trevor
"""

import pandas as pd 
import numpy as np 
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
import pysynphot.binning as astrobin
import pickle
import warnings as warn
import astropy.units as u

def binning(x, y,  dy=None, binwidth=None, r=None,newx= None, log = False, nan=False):
    fpath='/Users/Trevor/Pythoncode/HST/WASP79/'
    
    if len(x) != len(y):
        raise Exception('X and Y are not the same length')

    #check that either newx or binwidth are specified 

    if newx is None and binwidth is None and r is None:
        raise Exception('Need to either supply new x axis, resolution, or a binwidth')
    if (binwidth is None) and (log): 
        raise Exception("Cannot do logarithmic binning without a binwidth")

    if newx is not None: 
        bin_x = newx
        bin_x, bin_y, bin_dy, bin_n = uniform_tophat_mean(bin_x,x, y, dy=dy,nan=nan)
        bin_edge = astrobin.calculate_bin_edges(bin_x)
        
    elif r is not None:
        bin_x = bin_wave_to_R(x, r)
        bin_x, bin_y, bin_dy, bin_n = uniform_tophat_mean(bin_x,x, y, dy=dy,nan=nan)
        bin_edge = astrobin.calculate_bin_edges(bin_x)

    elif binwidth is not None: 
        if (binwidth < 0) and (log):
            warn.warn(UserWarning("Negative binwidth specified. Assuming this is log10(binwidth)"))
            binwidth = 10**binwidth
        if log:
            bin_x = np.arange(np.log10(min(x)),np.log10(max(x)),np.log10(binwidth))
            bin_x = 10**bin_x
        elif not log:
            bin_x = np.arange(min(x),max(x),binwidth)
        bin_x, bin_y, bin_dy, bin_n = uniform_tophat_mean(bin_x,x, y, dy=dy,nan=nan)
        bin_edge = astrobin.calculate_bin_edges(bin_x)

    outpath = fpath+'/W79_data/'
    fileObject = open(outpath+'bin_extract_out', 'wb')
    pickle.dump([bin_y, bin_x, bin_edge, bin_dy, bin_n], fileObject)
    fileObject.close()
    return {'bin_y':bin_y, 'bin_x':bin_x, 'bin_edge':bin_edge, 'bin_dy':bin_dy, 'bin_n':bin_n} 
    
def uniform_tophat_mean(newx,x, y, dy=None,nan=False):
    newx = np.array(newx)
    szmod=newx.shape[0]
    delta=np.zeros(szmod)
    ynew=np.zeros(szmod)
    bin_dy =np.zeros(szmod)
    bin_n =np.zeros(szmod)
        
    delta[0:-1]=newx[1:]-newx[:-1]  
    delta[szmod-1]=delta[szmod-2] 
    
    for i in range(szmod-1):
        i=i+1
        loc=np.where((x >= newx[i]-0.5*delta[i-1]) & (x < newx[i]+0.5*delta[i]))
        loc=np.array(loc)
        #make sure there are values within the slice 
        if len(loc[0]) > 0:
            #ynew[i]=np.mean(y[loc])
            ynew[i]=np.mean(np.array(y)[loc.astype(int)])
            if dy is not None: 
                bin_dy[i] = np.sqrt(np.sum(dy[loc]**2.0))/len(y[loc])
            bin_n[i] = len(np.array(y)[loc.astype(int)])
        #if not give empty slice a nan
        elif len(loc[0]) is 0 : 
            warn.warn(UserWarning("Empty slice exists within specified new x, replacing value with nan"))
            ynew[i]=np.nan
            bin_n[i] = np.nan 

    #fill in zeroth entry
    loc=np.where((x > newx[0]-0.5*delta[0]) & (x < newx[0]+0.5*delta[0]))
    loc=np.array(loc)
    
    if len(loc[0]) > 0: 
        #ynew[0]=np.mean(y[loc])
        ynew[0]=np.mean(np.array(y)[loc.astype(int)])
        bin_n[0] = len(np.array(y)[loc.astype(int)])
        if dy is not None: 
            bin_dy[0] = np.sqrt(np.sum(dy[loc]**2.0))/len(y[loc])
    elif len(loc[0]) is 0 : 
        ynew[0]=np.nan
        bin_n[0] = np.nan
        if dy is not None:
            bin_dy[0] = np.nan 

    #remove nans if requested
    out = pd.DataFrame({'bin_y':ynew, 'bin_x':newx, 'bin_dy':bin_dy, 'bin_n':bin_n})
    if not nan:
        out = out.dropna()
        
    return out['bin_x'].values,out['bin_y'].values, out['bin_dy'].values, out['bin_n'].values