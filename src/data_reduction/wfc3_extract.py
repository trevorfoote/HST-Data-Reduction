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
from scipy import signal
import pandas as pd
# import pickle
import warnings
warnings.filterwarnings("ignore")

from mjd_bjd import mjd_bjd


def wfc3_extract(
    fdir: str,
    scan: str,
    bkgd_dim: list,
    srch_box: list,
    x_means: float,
    numxaps=1,
    numyaps=1,
    flat_correct=False,
):
    """Extract white light curve from HST WFC3 data

    Parameters
    ----------
    fdir:           string
                    directory of target
    scan:           string
                    scanning direction for HST (forward or reverse)
    bkgd_dim:       list
                    Specify where to pull background from on image
                    in form [ymin, ymax, xmin, xmax]
    srch_box:       list
                    Specify where to search for signal to extract
                    in form [ymin, ymax, xmin, xmax]
    x_means:        float
                    how many means of image is xybox criteria boundary
    numxaps:        int
                    Aperature size in x direction
    numyaps:        int
                    Aperature size in y direction
    flat_correct:   bool
                    Specify whether to flatfield correct or not
    Returns
    -------
    wlc_data
        pandas data frame containing timestamps in MJD_UTC and BJD_TDB, flux, and flux error
    """

    ### Read in list of all exposure filenames
    filenames = sorted(glob.glob(f"{fdir}/exposures/{scan}/*ima.fits"))

    ### Read in direct image
    # directimage = glob.glob(f"{fdir}/direct image/*ima.fits")

    ### Loop over each exposure FITS file
    for i in range(0, len(filenames)):
        hdulist = fits.open(filenames[i])

        ### Initialize the arrays we will need
        if i == 0:
            nsamp = int(hdulist[0].header["NSAMP"])
            xaxis_len = hdulist[1].header["NAXIS1"]
            yaxis_len = hdulist[1].header["NAXIS2"]
            xoffset = int(hdulist[1].header['LTV1'] * -1)
            yoffset = int(hdulist[1].header['LTV2'] * -1)
            subarray_dim = [xoffset, yoffset, xaxis_len, yaxis_len]
            print(len(filenames), "files and", nsamp, "samples each")
            MJD_UTC = np.zeros(len(filenames))
            flux = np.zeros(shape=(nsamp, numyaps, numxaps))
            error = np.zeros(shape=(nsamp, numyaps, numxaps))
            wlc = np.zeros(shape=(len(filenames), numyaps, numxaps))
            wlc_error = np.zeros(shape=(len(filenames), numyaps, numxaps))
        if i % 10 == 0:
            print(i)

        ### Loop over each NDR
        # diff_image = np.zeros(shape=(nsamp - 1, xaxis_len, yaxis_len))
        for j in range(0, nsamp):
            # scidata = hdulist[j * 5 + 1].data
            err = hdulist[j * 5 + 2].data

            if j == 0:
                raDeg = hdulist[0].header["RA_TARG"]
                decDeg = hdulist[0].header["DEC_TARG"]
                expstart = hdulist[0].header["EXPSTART"]
                expend = hdulist[0].header["EXPEND"]
                MJD_UTC[i] = 0.5 * (expend + expstart)

            else:
                ### Take differential image between j and j-1
                scidata = hdulist[(j - 1) * 5 + 1].data - hdulist[j * 5 + 1].data
                scidata = clean(scidata, sigma=10)
                scidata, bkgd = background(fdir, scidata, bkgd_dim, subarray_dim, flat_correct)
                xybox = getbox(scidata, srch_box, x_means)
                # scidata = background_and_flat(fdir, diff_image[j - 1], bkgd_dim, subarray_dim, flat_correct)

                for aprx in range(0, numxaps):
                    for apry in range(0, numyaps):
                        ff = np.sum(
                            scidata[
                                xybox[2] - apry : xybox[3] + 1 + apry,
                                xybox[0] - aprx : xybox[1] + 1 + aprx,
                            ]
                        )
                        er = np.sum(
                            err[
                                xybox[2] - apry : xybox[3] + 1 + apry,
                                xybox[0] - aprx : xybox[1] + 1 + aprx,
                            ]
                            ** 2.0
                        )
                        flux[j, apry, aprx] = ff
                        error[j, apry, aprx] = er

                wlc_error[i, :, :] = np.sum(error, axis=0) ** 0.5
                wlc[i, :, :] = np.sum(flux)

    ### Convert MJD_UTC times to BJD_TDB
    BJD_TDB = mjd_bjd(MJD_UTC, raDeg, decDeg, "HST")

    ### Save extracted data

    wlc_data = np.vstack((MJD_UTC, BJD_TDB, wlc[:, 0, 0], wlc_error[:, 0, 0]))
    wlc_data = wlc_data.T.reshape(-1,4)
    wlc_data = pd.DataFrame(wlc_data, columns = ['Time (MJD_UTC)','Time (BJD_TDB)', 'Flux (e-)', 'Flux Error (e-)'])
    wlc_data.to_csv((f"{fdir}/output_data/{scan}_wlc.csv"), sep=',', index=False)

    ### Keep for future if we use aperature feature
    # fileObject = open(f"{fdir}/output_data/{scan}_wlc_extract_out", "wb")
    # pickle.dump(
    #     [MJD_UTC, BJD_TDB, wlc, wlc_error, raDeg, decDeg, scidata_0, xybox], fileObject
    # )
    # fileObject.close()

    return wlc_data

def clean(scidata, sigma):
    ### Find outlier data and replace with median values
    medfilt = signal.medfilt2d(scidata, (31,1))
    diff = scidata - medfilt
    temp = sigma_clip(diff, sigma=sigma, axis=0)
    mask = temp.mask
    int_mask = mask.astype(float) * medfilt
    test = (~mask).astype(float)
    clean_scidata = (scidata*test) + int_mask
    
    return clean_scidata

def flat_correct(fdir, scidata, subarray_dim,):
    ### Correct for flat field
    if flat_correct == True:
        flatimage = glob.glob(f"{fdir}/flatfield/*G141.flat*")
        hdulist = fits.open(flatimage[0])
        flat = hdulist[0].data
        sub_xstart, sub_ystart, sub_xwidth, sub_ywidth = subarray_dim
        flat = flat[sub_xstart : sub_xstart + sub_xwidth, sub_ystart : sub_ystart + sub_ywidth]
        hdulist.close()
        scidata = scidata / flat

def background(scidata, bkgd_dim):
    ### Background subtract
    ### Mask a 15 pixel border around scidata
    # xcols1 = np.arange(0, 15)
    # xcols2 = np.arange(sub_xwidth-15, sub_xwidth)
    # ycols1 = np.arange(0, 15)
    # ycols2 = np.arange(sub_ywidth-15, sub_ywidth)    
    # xedges = np.append(xcols1, xcols2)
    # yedges = np.append(ycols1, ycols2)
    # m = np.zeros_like(scidata)
    # m[:, xedges] = 1
    # m[yedges, :] = 1
    # scidata = np.ma.masked_array(scidata, m)

    ### Calculate median background and subtract from data
    backbox = scidata[bkgd_dim[0] : bkgd_dim[1], bkgd_dim[2] : bkgd_dim[3]]

#     bkgd = backbox.mean(axis=0)
#     bkgd = signal.medfilt(bkgd, 31)
#     bkgd = np.median(bkgd)
#     bkgd = np.ones((522,522))*bkgd

    bkgd = np.ones(np.shape(scidata))*np.mean(backbox)
    scidata = scidata - bkgd

    return scidata

def background_and_flat(fdir, scidata, bkgd_dim, subarray_dim, flat_correct):
    ### Correct for flat field
    if flat_correct == True:
        flatimage = glob.glob(f"{fdir}/flatfield/*G141.flat*")
        hdulist = fits.open(flatimage[0])
        flat = hdulist[0].data
        sub_xstart, sub_ystart, sub_xwidth, sub_ywidth = subarray_dim
        flat = flat[
            sub_xstart : sub_xstart + sub_xwidth, sub_ystart : sub_ystart + sub_ywidth
        ]
        hdulist.close()
        scidata = scidata / flat

    ### Mask a 15 pixel border around scidata
    cols1 = np.arange(0, 15)
    cols2 = np.arange(507, 522)
    edges = np.append(cols1, cols2)
    m = np.zeros_like(scidata)
    m[:, edges] = 1
    m[edges, :] = 1
    scidata = np.ma.masked_array(scidata, m)

    ### Clip any data in scidata that's greater than +/- 10 sigma
    scidata = sigma_clip(scidata, sigma=10)

    ### Calculate median background and subtract from data
    backbox = scidata[bkgd_dim[0] : bkgd_dim[1], bkgd_dim[2] : bkgd_dim[3]]
    bkgd = backbox.mean(axis=0)
    bkgd = signal.medfilt(bkgd, 31)
    bkgd = np.array(
        [
            bkgd,
        ]
        * 522
    )
    scidata = scidata - bkgd

    ### Clip any background corrected data that's greater than +/- 5 sigma
    #     scidata = sigma_clip(scidata, sigma=5)

    return scidata



def getbox(scidata, srch_box, x_means):
    scidata[scidata < 0] = 0

    bot_guess, top_guess, left_guess, right_guess = srch_box
    xx_step = (right_guess - left_guess) // 10
    holdy_top = []
    holdy_bot = []
    holdx_left = []
    holdx_right = []
    
    ### Find upper and lower edges
    for xx in range(left_guess, right_guess, xx_step):
        for yy in range(bot_guess, top_guess):
            ybot = yy
            if scidata[yy, xx] > x_means * np.mean(scidata):
                holdy_bot.append(ybot)
                break

        for yy in range(top_guess, bot_guess, -1):
            ytop = yy
            if scidata[yy, xx] > x_means * np.mean(scidata):
                holdy_top.append(ytop)
                break
                
    ybot = int(np.median(holdy_bot))
    ytop = int(np.median(holdy_top))

    ### Find left and right edges
    for yy in range(ybot, ytop, (ytop - ybot) // 9):
        for xx in range(left_guess, right_guess):
            xleft = xx
            if scidata[yy, xx] > x_means * np.mean(scidata):
                holdx_left.append(xleft)
                break
        
        for xx in range(right_guess, left_guess, -1):
            xright = xx
            if scidata[yy, xx] > x_means * np.mean(scidata):
                holdx_right.append(xright)
                break
        
    xleft = int(np.median(holdx_left))
    xright = int(np.median(holdx_right))

    xybox = np.array([xleft, xright, ybot, ytop])
    
    return xybox


if __name__ == "__main__":
    fdir = "/home/trevor/OneDrive/Coding/HST/HST-Data-Reduction/src/data_reduction/data/HD189733/Eclipse/Obs_date_062413"
    scan = "forward"
    subarray_dim = [245, 245, 522, 522]
    bkgd_dim = [425, 475, 0, 522]
    srch_box = [250, 500, 50, 250]
    time, wlc, wlc_error = wfc3_extract(fdir, scan, subarray_dim, bkgd_dim, srch_box)
