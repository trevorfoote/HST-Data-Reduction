import numpy as np
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def main():
    fpath='/Users/Trevor/Pythoncode/HST/WASP79/'
########### GET DIRECT IMAGE AND DETERMINE CENTROID ################

    from photutils import centroid_com as cen
    directimage=glob.glob(fpath +'*ima.fits')
    hdulist = fits.open(directimage[0])
    a = hdulist[1].data
    idx = np.argmax(a, axis=None)
    multi_idx = np.unravel_index(idx, a.shape)
    #cr=int(multi_idx[0])
    #cc=int(multi_idx[1])
    cr = 282 # estimate of centriod y pos from viewing direct image
    cc = 254 # estimate of centriod x pos from viewing direct image
    a = np.ma.array(a, mask=True)
    a.mask[cr-10:cr+10,cc-10:cc+10] = False # mask everything outside 10x10 pixel window to block other light sources (e.g. cosmic rays)
    x0, y0 = cen(a)+246
    centroid=[y0,x0]
    print('Centriod (y,x) =', centroid)
    pix=np.linspace(0,521, 522)
    pixp=pix+246

    wl = calibrateLambda(pixp, centroid, 'G141')
    plt.plot(pixp,wl)
    plt.xlabel('column number')
    plt.ylabel('wavelength')
    plt.show()
    
    trace = calcTrace(pixp, centroid, 'G141')
    #plt.plot(pixp, trace)
    #plt.show()
    
    return pixp, wl, trace
   

def calcTrace(x, centroid, grism):
    '''
    Calculates the WFC3 trace given the position of the direct image in physical pixels.
    
    Parameters
    ----------
    x           : physical pixel values along dispersion direction over which the trace is calculated
    centroid    : [y,x] pair describing the centroid of the direct image
    
    Returns
    -------
    y           : computed trace
    
    History
    -------
    Initial version by LK
    Modified by Kevin Stevenson     November 2012
    '''
    yref, xref = centroid
    print(yref)
    if isinstance(yref, float) == False:
        yref    = yref[:,np.newaxis]
        x       = x[np.newaxis]
    
    if grism == 'G141':
        #WFC3-2009-17.pdf
        #Table 1: Field dependent trace descriptions for G141.
        #Term       a0              a1(X)           a2(Y)           a3(X^2)         a4(X*Y)         a5(Y^2)
        DYDX_A_0 = [1.96882E+00,    9.09159E-05,    -1.93260E-03] 
        DYDX_A_1 = [1.04275E-02,    -7.96978E-06,   -2.49607E-06,   1.45963E-09,    1.39757E-08,    4.84940E-10]
    elif grism == 'G102':
        #WFC3-2009-18.pdf
        #Table 1: Field dependent trace descriptions for G102.
        #Term       a0              a1(X)           a2(Y)           a3(X^2)         a4(X*Y)         a5(Y^2)
        DYDX_A_0 = [-3.55018E-01,    3.28722E-05,   -1.44571E-03] 
        DYDX_A_1 = [ 1.42852E-02,   -7.20713E-06,   -2.42542E-06,   1.18294E-09,    1.19634E-08,    6.17274E-10
]
    else:
        print("Unknown filter/grism: " + grism)
        return 0
    
    DYDX_0 = DYDX_A_0[0] + DYDX_A_0[1]*xref + DYDX_A_0[2]*yref
    DYDX_1 = DYDX_A_1[0] + DYDX_A_1[1]*xref + DYDX_A_1[2]*yref + \
             DYDX_A_1[3]*xref**2 + DYDX_A_1[4]*xref*yref + DYDX_A_1[5]*yref**2
    
    y      = DYDX_0 + DYDX_1*(x-xref) + yref
    
    return y

def calibrateLambda(x, centroid, grism):
    '''
    Calculates coefficients for the dispersion solution
    
    Parameters
    ----------
    x           : physical pixel values along dispersion direction over which the wavelength is calculated
    centroid    : [y,x] pair describing the centroid of the direct image
    
    Returns
    -------
    y           : computed wavelength values
    
    History
    -------
    Initial version by LK
    Modified by Kevin Stevenson     November 2012
    '''
    yref, xref = centroid
    
    if isinstance(yref, float) == False:
        yref    = yref[:,np.newaxis]
        x       = x[np.newaxis]
    
    if grism == 'G141':
        #WFC3-2009-17.pdf
        #Table 5: Field dependent wavelength solution for G141.
        #Term       a0              a1(X)           a2(Y)           a3(X^2)         a4(X*Y)         a5(Y^2)
        DLDP_A_0 = [8.95431E+03,    9.35925E-02,            0.0,             0.0,           0.0,            0.0] 
        DLDP_A_1 = [4.51423E+01,    3.17239E-04,    2.17055E-03,    -7.42504E-07,   3.48639E-07,    3.09213E-07]
    elif grism == 'G102':
        #WFC3-2009-18.pdf
        #Table 5: Field dependent wavelength solution for G102.
        #FINDME: y^2 term not given in Table 5, assuming 0.
        #Term       a0              a1(X)           a2(Y)           a3(X^2)         a4(X*Y)         a5(Y^2)
        DLDP_A_0 = [6.38738E+03,    4.55507E-02,            0.0] 
        DLDP_A_1 = [2.35716E+01,    3.60396E-04,    1.58739E-03,    -4.25234E-07,  -6.53726E-08,            0.0]
    else:
        print("Unknown filter/grism: " + grism)
        return 0
    
    DLDP_0 = DLDP_A_0[0] + DLDP_A_0[1]*xref + DLDP_A_0[2]*yref
    DLDP_1 = DLDP_A_1[0] + DLDP_A_1[1]*xref + DLDP_A_1[2]*yref + \
             DLDP_A_1[3]*xref**2 + DLDP_A_1[4]*xref*yref + DLDP_A_1[5]*yref**2
    
    y      = DLDP_0 + DLDP_1*(x-xref) + yref
    
    return y

'''
def calcTrace(x, centroid, grism):
    
    Calculates the WFC3 trace given the position of the direct image in physical pixels.
    
    Parameters
    ----------
    x           : physical pixel values along dispersion direction over which the trace is calculated
    centroid    : [y,x] pair describing the centroid of the direct image
    
    Returns
    -------
    y           : computed trace
    
    History
    -------
    Initial version by LK
    Modified by Kevin Stevenson     November 2012
    Modified by Trevor Foote        October 2020
    
    yref, xref = centroid
    #print(yref)
    if isinstance(yref, float) == False:
        yref    = yref[:,np.newaxis]
        x       = x[np.newaxis]
    
    if grism == 'G141':
        #WFC3-2016-15.pdf
        #Table 1: Field dependent trace descriptions for G141.
        #Term       a0                      a1(X)                       a2(Y)                   a3(X^2)                 a4(X*Y)                 a5(Y^2)
        DYDX_A_0 = [2.08196481352,          -0.00019752130624389416,    -0.002202066565067532,  3.143514082596283e-8,   4.3212786880932414e-7,  1.210435999122636e-7] 
        DYDX_A_1 = [0.010205281672977665,   -6.06056923866002e-6,       -3.2485600412356953e-6, 4.2363866304617406e-10, 1.230956851333159e-8,   1.6123073931033502e-9]
    elif grism == 'G102':
        #WFC3-2016-15.pdf
        #Table 1: Field dependent trace descriptions for G102.
        #Term       a0                      a1(X)                       a2(Y)                   a3(X^2)                 a4(X*Y)                 a5(Y^2)
        DYDX_A_0 = [-0.16294444053626034,   -0.0002663379534341786,     -0.0018501621696214792, 4.877038129656633e-10,  5.408087165623451e-7,   1.5530956524130845e-7] 
        DYDX_A_1 = [  0.01409870961049523,  -6.379312386311464e-6,      -1.7291364097521173e-6, -6.706042298906293e-10, 1.2760204375540773e-8,  -1.4759256948695718e-10]
    else:
        print("Unknown filter/grism: " + grism)
        return 0
    
    DYDX_0 = DYDX_A_0[0] + DYDX_A_0[1]*xref + DYDX_A_0[2]*yref + \
             DYDX_A_0[3]*xref**2 + DYDX_A_0[4]*xref*yref + DYDX_A_0[5]*yref**2
    DYDX_1 = DYDX_A_1[0] + DYDX_A_1[1]*xref + DYDX_A_1[2]*yref + \
             DYDX_A_1[3]*xref**2 + DYDX_A_1[4]*xref*yref + DYDX_A_1[5]*yref**2
    
    y      = DYDX_0 + DYDX_1*(x-xref) + yref
    
    return y

def calibrateLambda(x, centroid, grism):
    
    Calculates coefficients for the dispersion solution
    
    Parameters
    ----------
    x           : physical pixel values along dispersion direction over which the wavelength is calculated
    centroid    : [y,x] pair describing the centroid of the direct image
    
    Returns
    -------
    y           : computed wavelength values
    
    History
    -------
    Initial version by LK
    Modified by Kevin Stevenson     November 2012
    Modified by Trevor Foote        October 2020
    
    yref, xref = centroid
    
    if isinstance(yref, float) == False:
        yref    = yref[:,np.newaxis]
        x       = x[np.newaxis]
    
    if grism == 'G141':
        #WFC3-2016-15.pdf
        #Table 5: Field dependent wavelength solution for G141.
        #Term       a0                  a1(X)                   a2(Y)                   a3(X^2)                     a4(X*Y)                     a5(Y^2)
        DLDP_A_0 = [8951.38620572,      0.08044032819916265,    -0.009279698766495334,  0.000021856641668116504,    -0.000011048008881387708,   0.00003352712538187608] 
        DLDP_A_1 = [44.97227893276267,  0.0004927891511929662,  0.0035782416625653765,  -9.175233345083485e-7,      2.2355060371418054e-7,      -9.258690000316504e-7]
    elif grism == 'G102':
        #WFC3-2016-15.pdf
        #Table 5: Field dependent wavelength solution for G102.
        #FINDME: y^2 term not given in Table 5, assuming 0.
        #Term       a0                  a1(X)                   a2(Y)                   a3(X^2)                     a4(X*Y)                     a5(Y^2)
        DLDP_A_0 = [6344.08102248,      0.201430850028975,      0.0802131361796817,     -0.00019613135070868445,    0.00003013960034834457,     -0.00008431572555355592] 
        DLDP_A_1 = [24.001233940762805, -0.0007160621018940599, 0.0008411542615870384,  8.977548140491455e-7,       -3.160441003220574e-7,      7.140436248957638e-7]
    else:
        print("Unknown filter/grism: " + grism)
        return 0
    
    DLDP_0 = DLDP_A_0[0] + DLDP_A_0[1]*xref + DLDP_A_0[2]*yref + \
             DLDP_A_0[3]*xref**2 + DLDP_A_0[4]*xref*yref + DLDP_A_0[5]*yref**2
    DLDP_1 = DLDP_A_1[0] + DLDP_A_1[1]*xref + DLDP_A_1[2]*yref + \
             DLDP_A_1[3]*xref**2 + DLDP_A_1[4]*xref*yref + DLDP_A_1[5]*yref**2
    
    y      = DLDP_0 + DLDP_1*(x-xref) + yref
    
    return y
'''

#main()