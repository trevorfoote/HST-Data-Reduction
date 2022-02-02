# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 14:36:27 2020

@author: Trevor
"""
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import bat_model

def mcmc(theta, BJD_TDB, flux, flux_err, model, params, HSTphase, t_secondary, 
         fixed_t_sec, G_t, G_err, SDNR, P):
    ndim, nwalkers, nsteps = len(theta), 50, 10000 #6*len(theta) #3*len(theta) 
    sampler=emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(params, model,
                                        BJD_TDB, flux, flux_err, HSTphase,
                                        t_secondary, fixed_t_sec, G_t, G_err, P))
    #pos=[theta*(1.+1.e-2*np.random.randn(ndim)) for i in range(nwalkers)]
    pos=[theta*(1.+SDNR*np.random.randn(ndim)) for i in range(nwalkers)]
    if fixed_t_sec == False:
        for i in np.arange(len(pos)):
            pos[i][6] = theta[6] + (0.005*np.random.randn())
    sampler.run_mcmc(pos, nsteps, progress=True);

    #Burn in removal
    samples=sampler.chain[:,int(0.2*nsteps):,:].reshape((-1,ndim))
    #samples=sampler.chain[:,:,:].reshape((-1,ndim))

    if fixed_t_sec == False:
        m_mcmc, b_mcmc, S_mcmc, phi_mcmc, C_mcmc, fp_mcmc, t0_mcmc=map(
                lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
                zip(*np.percentile(samples, [16, 50, 84], axis=0)))
        fig = corner.corner(samples, labels = ['m', 'b', 'S', 'phi', 'C', 
                                               'fp', 't_secondary'])
        mcmc_params = [m_mcmc, b_mcmc, S_mcmc, phi_mcmc, C_mcmc, fp_mcmc, 
                       t0_mcmc]
    else:
        m_mcmc, b_mcmc, S_mcmc, phi_mcmc, C_mcmc, fp_mcmc =map(
                lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
                zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    
        fig = corner.corner(samples, labels = ['m', 'b', 'S', 'phi', 'C', 
                                               'fp'])
        mcmc_params = [m_mcmc, b_mcmc, S_mcmc, phi_mcmc, C_mcmc, fp_mcmc]
    plt.show() 
    
    print('parameter', )
    print('m', m_mcmc)
    print('b', b_mcmc)
    print('S', S_mcmc)
    print('phi', phi_mcmc)
    print('C', C_mcmc)
    print('fp', fp_mcmc)
    if fixed_t_sec == False:
        print('t_sec', t0_mcmc)
        
    
    
    return sampler, samples, pos, mcmc_params

#posterior probability
def lnprob(theta, params, model, BJD_TDB, flux, flux_err, HSTphase, 
           t_secondary, fixed_t_sec, G_t, G_err, P):
    lp = lnprior(theta, G_t, G_err, fixed_t_sec, P)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, params, model, BJD_TDB, flux, flux_err, HSTphase,
                       t_secondary, fixed_t_sec)

#prior
def lnprior(theta, G_t, G_err, fixed_t_sec, P):
    if fixed_t_sec == False:
        m, b, S, phi, C, fp, t_secondary = theta
        if fp>1 or fp<0:
        #if fp>1 or fp<0 or t_secondary>G_t+P/3 or t_secondary<G_t-P/3:
            return -np.inf
        else:
            #return 0.0
            mu = G_t
            sigma = G_err
            return np.log((1.0/(np.sqrt(2*np.pi)*sigma**2))*np.exp(-0.5*(t_secondary-mu)**2/sigma**2))
            
    else:
        m, b, S, phi, C, fp = theta
        if fp>1 or fp<0:
            return -np.inf
        else:
            return 0.0

#likelihood function 
def lnlike(theta, params, model, BJD_TDB, flux, flux_err, HSTphase, 
           t_secondary, fixed_t_sec):
    res = bat_model.residuals(theta, model, params, BJD_TDB, HSTphase, 
                              flux, flux_err, t_secondary, fixed_t_sec)
    
    ln_likelihood = -0.5*(np.sum((res)**2 + np.log(2.0*np.pi*(flux_err)**2)))

    return ln_likelihood
