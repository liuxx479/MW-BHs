from astropy.io import fits
from scipy import *
import numpy as np
from pylab import *
#from astropy.coordinates import SkyCoord
import astropy.units as units
#from IPython.display import Image
#from astropy.coordinates import FK5, ICRS
from astropy import units as u
#from astropy.coordinates import match_coordinates_sky
#import WLanalysis
#from scipy.interpolate import interp2d,NearestNDInterpolator
from scipy.interpolate import interp1d
import os, sys
#from apogee.spec import continuum
from emcee.utils import MPIPool 

apodir = '/work/02977/jialiu/ApogeeLine/'

os.chdir(apodir+'binspec')

import utils
import spectral_model
import fitting

# read in all individual neural networks we'll need. 
NN_coeffs_norm = utils.read_in_neural_network(name = 'normalized_spectra')
NN_coeffs_flux = utils.read_in_neural_network(name = 'unnormalized_spectra')
NN_coeffs_R = utils.read_in_neural_network(name = 'radius')
NN_coeffs_Teff2_logg2 = utils.read_in_neural_network(name = 'Teff2_logg2')
wavelength = utils.load_wavelength_array()


def read_spec(fitsfile, get_vhelio=0):    
    ihdulist = fits.open(fitsfile)
    idate = ihdulist[0].header['DATE-OBS'][:10]
    ispec = ihdulist[1].data
    ierr = ihdulist[2].data
    imask = ihdulist[3].data
    ipass = ones(shape=imask.shape)
    for ibit in (0,1,2,3,4,5,6,7,12,13,14):
        ipass[~logical_not(imask & 2**ibit)]=0
    ilambda = ihdulist[4].data
    iidx=ilambda.flatten()
    if get_vhelio:
        vhelio = ihdulist[0].header['VHELIO']
        return ilambda, ispec, ierr, ipass, idate, vhelio
    else:
        return ilambda, ispec, ierr, ipass, idate


def prep_normed_spec (ifitsfn):
    ilambda, ispec, ierr, ipass, idate, ivhelio = read_spec(ifitsfn, get_vhelio=1)
    ierr[where(ipass==0)]*=100
    data_spec = interp1d(ilambda[where(ipass)].flatten(), ispec[where(ipass)].flatten(), fill_value="extrapolate")(wavelength)
    specerr = interp1d(ilambda.flatten(), ierr.flatten(), fill_value="extrapolate")(wavelength)
    icont = utils.get_apogee_continuum(wavelength, data_spec, spec_err = specerr, cont_pixels = None)   
    data_spec/=icont
    specerr/=icont
    return data_spec, specerr, ivhelio, idate
    
def prep_visit_spec(iapoid):
    fitsfn_arr = [apodir+'specs_visit/MS/%s/%s'%(iapoid, ifn) 
                  for ifn in os.listdir(apodir+'specs_visit/MS/%s'%(iapoid))]
    out_arr = map(prep_normed_spec, fitsfn_arr)
    data_spec_arr,data_err_arr,vhelio_arr,date_arr  = [[ivisit[i] for ivisit in out_arr] for i in range(4)]
    return data_spec_arr, data_err_arr, vhelio_arr, date_arr
    #return out_arr

def fit_visits (iapoid, N=3):
    data_spec_arr, data_err_arr, vhelio_arr,date_arr = prep_visit_spec(iapoid)
    
    p0, pcov, test1_spec = fitting.fit_normalized_spectrum_single_star_model(norm_spec=data_spec_arr[0], 
                    spec_err=data_err_arr[0], NN_coeffs_norm = NN_coeffs_norm, NN_coeffs_flux = NN_coeffs_flux, 
                    p0 = None, num_p0 = 1)
    
    popt_single, pcov, single_spec = fitting.fit_visit_spectra_single_star_model(norm_spectra=data_spec_arr, 
                    spec_errs=data_err_arr, NN_coeffs_norm = NN_coeffs_norm, NN_coeffs_flux = NN_coeffs_flux, 
                    v_helios=vhelio_arr, p0 = p0, num_p0 = 1)
    
    out = fitting.fit_visit_spectra_N(norm_spectra=data_spec_arr, spec_errs=data_err_arr, NN_coeffs_norm = NN_coeffs_norm, 
                          NN_coeffs_flux = NN_coeffs_flux, NN_coeffs_R = NN_coeffs_R, NN_coeffs_Teff2_logg2 = NN_coeffs_Teff2_logg2, 
                          v_helios=vhelio_arr, p0_single=popt_single,N=N)
    popt, pcov, model_specs = out
    return data_spec_arr, data_err_arr, single_spec, model_specs, vhelio_arr, date_arr, popt_single, popt, pcov

def plot_visit_fits (iapoid, out3, out10, ishow=0):
    data_spec_arr, data_err_arr, single_spec, model_specs3, vhelio_arr, date_arr, popt_single, popt3, pcov = out3
    data_spec_arr, data_err_arr, single_spec, model_specs10, vhelio_arr, date_arr, popt_single, popt10, pcov = out10
    istep=0.3
    ledges = [[15140, 15810], [15850, 16435], [16470,16955]]
    dof = [len(array(data_spec_arr).flatten())+ ix for ix in (len(popt_single), len(popt3), len(popt10))]
    
    chi1, chi3, chi10 = [sum((array(([single_spec, model_specs3, model_specs10][i])-array(data_spec_arr))/array(data_err_arr))**2)/dof[i] for i in range(3)]
    #print chi1, chi3, chi10
        
    f, axes=subplots(3,1,figsize=(12,8))
    for j in range(len(date_arr)):        
        data_spec, specerr, N1_spec, N3_spec, N10_spec = (data_spec_arr[j], 
                      data_err_arr[j], single_spec[j], model_specs3[j], model_specs10[j])
        
        for i in range(len(ledges)):
            m = (wavelength > ledges[i][0]) & (wavelength < ledges[i][1]) 
            axes[i].fill_between(wavelength[m], j*istep+data_spec[m]-specerr[m], j*istep+data_spec[m]+specerr[m], color='k',alpha=0.1)
            axes[i].plot(wavelength[m], j*istep+data_spec[m], color='k', lw=0.5, label = '%s'%(date_arr[j]))
            if j==len(spec_fitted)-1:
                axes[i].plot(wavelength[m], j*istep+N1_spec[m], color='b', lw=0.5, label = 'N=1(%.2f)'%(chi1))
                axes[i].plot(wavelength[m], j*istep+N3_spec[m], color='r', lw=0.5, label = 'N=3(%.2f)'%(chi3))
                axes[i].plot(wavelength[m], j*istep+N10_spec[m], color='g', lw=0.5, label = 'N=10(%.2f)'%(chi10))
            else:
                axes[i].plot(wavelength[m], j*istep+N1_spec[m], color='b', lw=0.5)
                axes[i].plot(wavelength[m], j*istep+N3_spec[m], color='r', lw=0.5)
                axes[i].plot(wavelength[m], j*istep+N10_spec[m], color='g', lw=0.5)

            axes[i].set_xlim(ledges[i])
            axes[i].set_ylim(0.7, len(spec_fitted)*istep+1)
    axes[0].legend(loc = 'best', frameon = 1, fontsize= 6, ncol=len(date_arr)/10+1)
    axes[1].set_ylabel('Normalized Flux')
    axes[-1].set_xlabel('Wavelenght A')
    axes[0].set_title('Visit Spec Fit %s'%(iapoid))
    if ishow: 
        show()
    else:
        savefig(apodir+'specs_fit_plot/MS_visit_joint/%s_fit.jpg'%(iapoid))
        savefig(apodir+'specs_fit_plot/MS_visit_joint_pdf/%s_fit.pdf'%(iapoid))
        close()


    f, axes=subplots(3,1,figsize=(12,8))
    for j in range(len(date_arr)):
        data_spec, specerr, N1_spec, N3_spec, N10_spec = (data_spec_arr[j], 
                      data_err_arr[j], single_spec[j], model_specs3[j], model_specs10[j])
        for i in range(len(ledges)):
            m = (wavelength > ledges[i][0]) & (wavelength < ledges[i][1]) 
            axes[i].fill_between(wavelength[m], j*istep+1-specerr[m], j*istep+1+specerr[m], color='k',alpha=0.2)
            axes[i].plot(wavelength[m], j*istep+ones(sum(m)), color='k', lw=1, label = '%s'%(date_arr[j]))
            if j==len(spec_fitted)-1:  
                axes[i].plot(wavelength[m], j*istep+N1_spec[m]/data_spec[m], color='b', lw=0.3, label = 'N=1(%.2f)'%(chi1))
                axes[i].plot(wavelength[m], j*istep+N3_spec[m]/data_spec[m], color='r', lw=0.3, label = 'N=3(%.2f)'%(chi3))
                axes[i].plot(wavelength[m], j*istep+N10_spec[m]/data_spec[m], color='g', lw=0.3, label = 'N=10(%.2f)'%(chi10))
            else:
                axes[i].plot(wavelength[m], j*istep+N1_spec[m]/data_spec[m], color='b', lw=0.3)
                axes[i].plot(wavelength[m], j*istep+N3_spec[m]/data_spec[m], color='r', lw=0.3)
                axes[i].plot(wavelength[m], j*istep+N10_spec[m]/data_spec[m], color='g', lw=0.3)

            axes[i].set_xlim(ledges[i])
            axes[i].set_ylim(0.7, len(spec_fitted)*istep+1)
    axes[0].legend(loc = 'best', frameon = 1, fontsize= 6,ncol=len(date_arr)/10+1)
    axes[1].set_ylabel('Flux_model / Flux_data')
    axes[-1].set_xlabel('Wavelenght A')
    axes[0].set_title('Visit Spec Fit Ratio %s'%(iapoid))
    if ishow: 
        show()
    else:
        savefig(apodir+'specs_fit_plot/MS_visit_joint/%s_diff.jpg'%(iapoid))
        savefig(apodir+'specs_fit_plot/MS_visit_joint_pdf/%s_diff.pdf'%(iapoid))
        close()
        
def process_MS_visit_fits(iapoid):
    'process all the MS visit spec, takes a loooong time'
    out_arr = []
    for iN in (2,3,5,10):
        print iapoid, iN
        out = fit_visits(iapoid, N=iN)
        out_arr.append(out)
        ### save to files
        data_spec_arr, data_err_arr, single_spec, model_specs, vhelio_arr, date_arr, popt_single, popt, pcov = out
        save(apodir+'specs_fit/%s_N%i_specs.npy'%(iapoid, iN), 
             [data_spec_arr, data_err_arr, single_spec, model_specs])
        save(apodir+'specs_fit/%s_N%i_params.npy'%(iapoid, iN), popt)
        save(apodir+'specs_fit/%s_N%i_cov.npy'%(iapoid, iN), pcov)
        if iN ==2:
            save(apodir+'specs_fit/%s_vhelio.npy'%(iapoid), vhelio_arr)        
            save(apodir+'specs_fit/%s_date.npy'%(iapoid), date_arr)
            save(apodir+'specs_fit/%s_N1_params.npy'%(iapoid), popt_single)
    plot_visit_fits (iapoid, out_arr[1], out_arr[-1], ishow=0)
    
pool=MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

apoid_candidates = os.listdir(apodir+'specs_visit/MS')    
pool.map(process_MS_visit_fits, apoid_candidates)
pool.close()

print 'done-done-done'
