# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 09:44:11 2020

@author: Trevor
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import glob
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy import optimize, signal
import pickle
import warnings
warnings.filterwarnings("ignore")

def wfc3_extract(plnm='W79', numxaps=5, numyaps=10):

###########   Working Directory   #####################   
    #fpath='/Users/Trevor/Pythoncode/HST/WASP79/'
    fpath = '/home/trevor/OneDrive/Coding/HST/WASP79/' #CHANGE to your directory
    filenames = glob.glob(fpath +'exposures/'+ '*ima.fits') #CHANGE if storing 
                               #your exposures in a different folder structure
    filenames = sorted(filenames)
    directimage = glob.glob(fpath +'*ima.fits')
    flatimage = glob.glob(fpath + '*G141.flat*')
    hdulist = fits.open(flatimage[0])
    flat = hdulist[0].data
    flat = flat[245:245+522, 245:245+522] #CHANGE Will need to look in header 
            #information or look at image using DS9 to find ACTUAL pixel 
            #numbers on detector that correspond to origin in image.
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
            temp=np.zeros(shape=(subarray, subarray, nsamp))
            images=np.zeros(shape=(subarray, subarray, nsamp, len(filenames)))
            time=np.zeros(len(filenames))
            raDeg=np.zeros(len(filenames))
            decDeg=np.zeros(len(filenames))
            wlc=np.zeros(shape=(len(filenames), numyaps, numxaps))
            scan_ang=np.zeros(len(filenames))
            ndrwhite=np.zeros(shape=(nsamp, numyaps, numxaps))  
            flux=np.zeros(shape=(nsamp, numyaps, numxaps))
            error=np.zeros(shape=(nsamp, numyaps, numxaps))
            diff=np.zeros(shape=(numyaps, numxaps))
            wlc_error=np.zeros(shape=(len(filenames), numyaps, numxaps))
            
        if i % 25 == 0:
            print(i)        
        ###########   LOOP OVER EACH NDR   #####################  
        for j in range(0,nsamp):
            scidata=hdulist[j*5+1].data
            err=hdulist[j*5+2].data

            if j == 0:
                scidata_0 = scidata
                expstart=hdulist[0].header['EXPSTART']
                expend=hdulist[0].header['EXPEND']
                exptime=hdulist[0].header['EXPTIME']
                scan_ang[i]=hdulist[0].header['SCAN_ANG']
                scanlen=hdulist[0].header[ 'SCAN_LEN']
                raDeg = hdulist[0].header['RA_TARG']
                decDeg = hdulist[0].header['DEC_TARG']
                time[i]=0.5*(expend+expstart)
                if i ==0:
                    xybox=getbox(scidata) #THIS GETS A BOX THE SIZE OF THE SPEC
                    x_range=xybox[1]+1-xybox[0]
                    y_range=xybox[3]+1-xybox[2]
                    x_cen=np.floor((xybox[1]+1-xybox[0])/2.)
                    y_cen=np.floor((xybox[3]+1-xybox[2])/2.)

         #############  FLAT FIELD AND BACKGROUND ###############
            scidata, images=background_and_flat(scidata, images, flat, j, i) 

            for aprx in range(0,numxaps):
                xwidth=x_range+aprx
                
                for apry in range(0,numyaps):
                    ff=np.sum(scidata[xybox[2]-apry:xybox[3]+1+apry, xybox[0]-
                                      aprx:xybox[1]+1+aprx])
                    er=np.sum(err[xybox[2]-apry:xybox[3]+1+apry, xybox[0]-
                                      aprx:xybox[1]+1+aprx]**2.)
                    flux[j,apry, aprx]=ff
                    error[j,apry, aprx]=er

            wlc_error[i,:,:]=np.sum(error, axis=0)**0.5
            
        for jj in range(0,nsamp):
            if jj==0: 
                diff=np.zeros(shape=(numyaps, numxaps))
            else:  
                diff+=(flux[jj-1, :, :]- flux[jj]) 

        wlc[i, :, :]=diff
    print('wlc',wlc)
    plt.figure()
    plt.errorbar(time,  wlc[:,0,0], yerr=wlc_error[:,0,0], fmt='o', color='k')
    plt.xlabel('Time ($MJD_{UTC}$)')
    plt.ylabel('Flux (e$^-$)')
    plt.title('WFC3 Raw Light curve')
    plt.show()
       
    ###These two can be commented out but have to figure out what the 
    ###other data in that packet are still
    wlc = wlc[:,0,0]
    wlc_error = wlc_error[:,0,0]

    
    outpath = fpath+'W79_data/'
    fileObject = open(outpath+'wlc_extract_out', 'wb')
    pickle.dump([time, wlc, wlc_error, raDeg, decDeg, scidata_0, scidata, 
                 xybox], fileObject)
    fileObject.close()
    
    return time, wlc, wlc_error, raDeg, decDeg, scidata_0, scidata, xybox


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
    backbox=scidata[xybox[3]+50:xybox[3]+100, :] #Change range of backbox to 
                                                #be where you want the background
                                                #pulled from in form 
                                                #scidata[ymin, ymax, xmin, xmax]
    bkgd = backbox.mean(axis=0)
    #print('background',bkgd)
    bkgd = sp.signal.medfilt(bkgd,31)
    bkgd = np.array([bkgd,]*522)
    
    scidata = scidata-bkgd
    scidata = sigma_clip(scidata, sigma=5)
    images[:,:,j,i]=scidata


    return scidata, images

### Finds 1st order 
def getbox(scidata):
    holdy=np.zeros(10)
    holdx=np.zeros(10)
    for xx in range(80,180,10): #CHANGE the 80 & 180 to span pixels that are on 
                                #either side of the left edge of your spectra 
                                #without encompassing contamination sources
        for yy in range(0,250): #CHANGE the 0 & 250 to span pixels that are on
                                #either side of the bottom edge of your spectra
                                #without encompassing contamination sources
            ybot=yy
            if scidata[yy,xx] > 2*np.mean(scidata):
                break
        holdy[int((xx-80)/10-1)]=ybot #CHANGE 80 to min value chosen for xx
    ybot=int(np.median(holdy))
    
    for xx in range(80,180,10): #CHANGE the 80 & 180 to span pixels that are on 
                                #either side of the left edge of your spectra 
                                #without encompassing contamination sources
        for yy in range(450,0, -1): #CHANGE the 450 & 0 to span pixels that are 
                                #on either side of the top edge of your spectra
                                #without encompassing contamination sources
            ytop=yy
            if scidata[yy,xx] > 2*np.mean(scidata):
                break
        holdy[int((xx-80)/10-1)]=ytop #CHANGE 80 to min value chosen for xx
    ytop=int(np.median(holdy))

    for yy in range(ybot,ytop, (ytop-ybot)//6):
        for xx in range(0,350): #CHANGE the 0 & 350 to span pixels that are on 
                                #either side of the left edge of your spectra 
                                #without encompassing contamination sources
            xleft=xx
            if scidata[yy,xx] > 2*np.mean(scidata):
                break
        holdx[int((yy-ybot)/((ytop-ybot)//6)-1)]=xleft
    xleft=int(np.median(holdx))
    for yy in range(ybot,ytop, (ytop-ybot)//6):
        for xx in range(250,0, -1): #CHANGE the 250 & 0 to span pixels that are on 
                                #either side of the right edge of your spectra 
                                #without encompassing contamination sources
            xright=xx
            if scidata[yy,xx] > 2*np.mean(scidata):
                break
        holdx[int((yy-ybot)/((ytop-ybot)//6)-1)]=xright
    xright=int(np.median(holdx))
    global xybox
    xybox=np.array([xleft, xright, ybot, ytop])

    print('xybox(xleft, xright, ybot, ytop)=', xybox)
    return xybox

#time, wlc, wlc_error, raDeg, decDeg, scidata_0, scidata, xybox = wfc3_extract(plnm='W79', numxaps=5, numyaps=10)