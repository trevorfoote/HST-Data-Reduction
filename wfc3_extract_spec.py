# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 09:44:11 2020

@author: Trevor
"""
import numpy as np
import scipy as sp
import glob
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy import optimize, signal
import pickle
import warnings
warnings.filterwarnings("ignore")

def wfc3_extract(num_bins=3):

###########   Working Directory   #####################
    fpath='/Users/Trevor/Pythoncode/HST/WASP79/'
    filenames=glob.glob(fpath +'exposures/'+ '*ima.fits')
    flatimage=glob.glob(fpath + '*G141.flat*')
    hdulist = fits.open(flatimage[0])
    flat=hdulist[0].data
    flat=flat[245:245+522, 245:245+522]
    hdulist.close()

###########   LOOP OVER EACH FITS FILE   #####################
    for i in range(0, len(filenames)):
        hdulist = fits.open(filenames[i])

########  make the arrays we will need  #######################
        if i == 0:
            subarray=len(hdulist[1].data)
            nsamp=hdulist[0].header['NSAMP']
            print (str(len(filenames))+'files and'+str(nsamp)+'samples each')
            nsamp=int(nsamp)
            images=np.zeros(shape=(subarray, subarray, nsamp, len(filenames)))
            bin_flux = np.zeros(shape=(len(filenames), num_bins))
            bin_flux_err = np.zeros(shape=(len(filenames), num_bins))

        if i % 25 == 0:
            print(i)
        ###########   LOOP OVER EACH NDR   #####################
        for j in range(0,nsamp):
            scidata=hdulist[j*5+1].data
            err=hdulist[j*5+2].data

            if j == 0:
                if i ==0:
                    xybox=getbox(scidata) #THIS GETS A BOX THE SIZE OF THE SPEC
                    flux = np.zeros(shape=(len(filenames), nsamp, xybox[1]-xybox[0]+1))
                    error = np.zeros(shape=(len(filenames), nsamp, xybox[1]-xybox[0]+1))
                    xrng = np.arange(xybox[0], xybox[1]+1)
                    yref = (xybox[3]+xybox[2])/2
                    xref = xybox[1]+50
                    
                    DLDP_A_0 = [8.95431E+03,    9.35925E-02,    0.0,
                                0.0,            0.0,            0.0]
                    DLDP_A_1 = [4.51423E+01,    3.17239E-04,    2.17055E-03,
                                -7.42504E-07,   3.48639E-07,    3.09213E-07]

                    DLDP_0 = DLDP_A_0[0] + DLDP_A_0[1]*xref + DLDP_A_0[2]*yref
                    DLDP_1 = DLDP_A_1[0] + DLDP_A_1[1]*xref + DLDP_A_1[2]*yref + \
                             DLDP_A_1[3]*xref**2 + DLDP_A_1[4]*xref*yref + \
                             DLDP_A_1[5]*yref**2
                    wl_y = DLDP_0 + DLDP_1*(xrng-xref) + yref
                    obs_wl = (1.-0.02)+wl_y/10000
                    wl_len = len(obs_wl)
                    print('wl_len', wl_len)
                    wl_width = wl_len/num_bins

                    bin_wls = (np.mean(obs_wl[0:int(wl_width)]),
                               np.mean(obs_wl[int(wl_width):2*int(wl_width)]),
                               np.mean(obs_wl[2*int(wl_width):3*int(wl_width)]))
                    print('Wl1', obs_wl[0], obs_wl[int(wl_width)])
                    print('Wl2', obs_wl[int(wl_width)], obs_wl[2*int(wl_width)])
                    print('Wl3', obs_wl[2*int(wl_width)], obs_wl[3*int(wl_width)-1])

         #############  FLAT FIELD AND BACKGROUND ###############
            scidata, images=background_and_flat(scidata, images, flat, j, i)
            
            for k in np.arange(xybox[0], xybox[1]+1):
                flux[i,j,k-xybox[0]] = np.sum(scidata[xybox[2]:xybox[3]+1, k])
                error[i,j,k-xybox[0]] = np.sum(err[xybox[2]:xybox[3]+1, k]**2.)

            
##############  BINNING  #######################
    ff = np.zeros(shape=(len(filenames),xybox[1]-xybox[0]+1))
    er = np.zeros(shape=(len(filenames),xybox[1]-xybox[0]+1))
    for ii in np.arange(len(filenames)):
        for jj in np.arange(xybox[1]-xybox[0]+1):
            er[ii,jj] = np.sum(error[ii,:,jj])
            for kk in np.arange(nsamp):
                if kk==0:
                    sumdiff = 0
                else:
                    sumdiff += flux[ii,kk-1,jj] - flux[ii,kk,jj]
            ff[ii,jj] = sumdiff

    for iii in np.arange(len(filenames)):
        bin_flux[iii,0] = np.sum(ff[iii, 0:int(wl_width)])
        bin_flux[iii,1] = np.sum(ff[iii, int(wl_width):2*int(wl_width)])
        bin_flux[iii,2] = np.sum(ff[iii, 2*int(wl_width):3*int(wl_width)])

        bin_flux_err[iii,0] = (np.sum(er[iii, 0:int(wl_width)]))**0.5
        bin_flux_err[iii,1] = (np.sum(er[iii, int(wl_width):2*int(wl_width)]))**0.5
        bin_flux_err[iii,2] = (np.sum(er[iii, 2*int(wl_width):3*int(wl_width)]))**0.5
    
    #outpath = fpath+'/W79_data/'
    #fileObject = open(outpath+'spec_extract_out', 'wb')
    #pickle.dump([bin_wls, bin_flux, bin_flux_err], fileObject)
    #fileObject.close()
    
    return bin_wls, bin_flux, bin_flux_err, obs_wl


def background_and_flat(scidata, images, flat, j, i):
    scidata=scidata/flat
    cols1=np.arange(0,15)
    cols2=np.arange(500,522)
    edges=np.append(cols1,cols2)
    m = np.zeros_like(scidata)
    m[:,edges] = 1
    m[edges, :] = 1
    scidata=np.ma.masked_array(scidata, m)
    scidata = sigma_clip(scidata, sigma=7)
    #backbox=scidata[xybox[2]-100:xybox[2]-50, :]
    backbox=scidata[xybox[3]+50:xybox[3]+100, :]
    bkgd=backbox.mean(axis=0)
    #print('background',bkgd)
    bkgd = sp.signal.medfilt(bkgd,31)
    bkgd=np.array([bkgd,]*522)

    scidata=scidata-bkgd
    scidata = sigma_clip(scidata, sigma=5)
    images[:,:,j,i]=scidata

    return scidata, images


### Finds 1st order
def getbox(scidata):
    holdy=np.zeros(10)
    holdx=np.zeros(10)
    for xx in range(80,180,10):
        for yy in range(0,250):
            ybot=yy
            if scidata[yy,xx] > 2*np.mean(scidata):
                break
        holdy[int((xx-80)/10-1)]=ybot
    ybot=int(np.median(holdy))

    for xx in range(80,180,10):
        for yy in range(450,0, -1):
            ytop=yy
            if scidata[yy,xx] > 2*np.mean(scidata):
                break
        holdy[int((xx-80)/10-1)]=ytop
    ytop=int(np.median(holdy))

    for yy in range(ybot,ytop, (ytop-ybot)//6):
        for xx in range(0,350):
            xleft=xx
            if scidata[yy,xx] > 2*np.mean(scidata):
                break
        holdx[int((yy-ybot)/((ytop-ybot)//6)-1)]=xleft
    xleft=int(np.median(holdx))

    for yy in range(ybot,ytop, (ytop-ybot)//6):
        for xx in range(250,0, -1):
            xright=xx
            if scidata[yy,xx] > 2*np.mean(scidata):
                break
        holdx[int((yy-ybot)/((ytop-ybot)//6)-1)]=xright
    xright=int(np.median(holdx))

    global xybox
    xybox=np.array([xleft, xright, ybot, ytop])
    print('xybox(xleft, xright, ybot, ytop)=', xybox)

    return xybox

#bin_wls, bin_flux, bin_flux_err, obs_wl = wfc3_extract(num_bins=3)