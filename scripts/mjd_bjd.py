# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 11:49:59 2020

@author: Trevor
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
from skyfield.api import load
from astropy.time import Time
from barycorrpy import utc_tdb

###########   Working Directory   #####################   


def mjd_bjd(MJD_UTC, flux, flux_err, raDeg, decDeg):
    #fpath='/Users/Trevor/Pythoncode/HST/WASP79/W79_data/'
    fpath = '/home/trevor/OneDrive/Coding/HST/WASP79/W79_data/'
    fileObject = open(fpath+'wlc_extract_out', 'rb')
    MJD_UTC, flux, flux_err, raDeg, decDeg, scidata_0, scidata, xybox = pickle.load(fileObject)
    fileObject.close()
    
    stations_url = 'http://celestrak.com/NORAD/elements/science.txt'
    satellites = load.tle_file(stations_url)
    by_name = {sat.name: sat for sat in satellites}
    satellite = by_name['HST']
    print(satellite)
    
    # MJDUTC in correct format for datetime64
    MJDUTC_dt = Time(MJD_UTC, format='mjd', scale='utc')
    yrs =[]
    mths = []
    days = []
    hours = []
    mins = []
    secs = []
    
    for i in np.arange(len(MJDUTC_dt)):
        tt = MJDUTC_dt[i].datetime64
        yrss = tt.astype('datetime64[Y]')
        MM = tt.astype('datetime64[M]')
        mthss = MM-yrss
        dd = tt.astype('datetime64[D]')
        dayss = dd-MM
        hh = tt.astype('datetime64[h]')
        hourss = hh-dd
        mm = tt.astype('datetime64[m]')
        minss = mm-hh
        ss = tt.astype('datetime64[ns]')
        secss = ss-mm
    
        yrs.append(yrss.astype(int)+1970)
        mths.append(mthss.astype(int)+1)
        days.append(dayss.astype(int)+1)
        hours.append(hourss.astype(int))
        mins.append(minss.astype(int))
        secs.append(secss.astype('float')*1e-9)
    
    ts = load.timescale(builtin=True)
    
    tmin = ts.utc(yrs,mths,days,hours,mins,secs) 
    geocentric = satellite.at(tmin)
    subpoint = geocentric.subpoint()
    lati = subpoint.latitude.degrees
    longit = subpoint.longitude.degrees
    elevation = subpoint.elevation.m
    
    for i in np.arange(len(elevation)):
        elevation[i] = int(elevation[i])
        
    JDUTC = MJD_UTC + 2400000.5
    results=[]
    for i in np.arange(len(JDUTC)):
        results.append(utc_tdb.JDUTC_to_BJDTDB(JDUTC[i], ra= raDeg, dec = decDeg, 
                                               lat=lati[i], longi=longit[i], 
                                               alt=elevation[i]))
        if i % 12 == 0:
                print(i)
    
    BJD_TDB=[]
    for i in np.arange(len(results)):
        BJD_TDB.append(float(results[i][0]))
    BJD_TDB = np.array(BJD_TDB)
    
    plt.figure()
    plt.errorbar(BJD_TDB, flux, flux_err, fmt='o', color='k')
    plt.xlabel('Time ($BJD_{TDB}$)')
    plt.ylabel('Flux (e$^-$)')
    plt.show()

    outpath = fpath
    fileObject = open(outpath+'BJD_TDB', 'wb')
    pickle.dump([BJD_TDB, flux, flux_err], fileObject)
    fileObject.close()    
    
    return BJD_TDB

