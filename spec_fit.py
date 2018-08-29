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
import matplotlib
matplotlib.use('Agg')

apodir = '/work/02977/jialiu/ApogeeLine/'
os.chdir(apodir+'binspec')
list_components=[2,]#3,5] #[2,3,5,10]

import utils
import spectral_model
import fitting

############ new addition after Yuansen created new NN for giants
batch = str(sys.argv[1])
print batch

#if batch == 'lachlan':
    #apoid_candidates = load(apodir+'overly_bright_ids_filt.npy').T[0][::-1]
    #batchname = 'dwarfs_lachlan_kareemNN'
#elif batch == 'test':
    #seed(10)
    #all_giants = load(apodir+'APOGEE_ID_giants_goodpara.npy')
    ##apoid_candidates = all_giants[2000:2100]
    #apoid_candidates = all_giants[choice(len(all_giants), 1000, replace=0)]
    #batchname = 'giants_'+batch
#elif batch == 'badpara':
    #all_giants = load(apodir+'APOGEE_ID_giants_badpara.npy')
    ##apoid_candidates = all_giants[2000:2100]
    #apoid_candidates = all_giants#[choice(len(all_giants), 1000, replace=0)]
    #batchname = 'giants_'+batch
#else: ## batch = 0,1,2,3,..9, chop up the data into 10 chunks for analysis
    #all_giants = load(apodir+'APOGEE_ID_giants_goodpara.npy')
    #Nchunk = len(all_giants)/10+1
    #apoid_candidates = all_giants[Nchunk*int(batch):Nchunk*(int(batch)+1)]
    #batchname = 'giants_'+batch



#hdulist_visit = fits.open(apodir+'allVisit-l31c.2.fits')
#out = [hdulist_visit[1].data[x] for x in ['APOGEE_ID','PLATE','MJD','FILE']]
#save('ID_PLATE_MJD_FILE.npy',array(out).T)
APOGEE_ID, PLATE, MJD, FILE = load(apodir+'ID_PLATE_MJD_FILE.npy').T

apoid_unique = unique(APOGEE_ID)
Nchunk = len(apoid_unique)/10+1
apoid_candidates = apoid_unique[Nchunk*int(batch):Nchunk*(int(batch)+1)]
batchname = 'batch_'+batch
print batchname, 'total candidates: %s'%(len(apoid_candidates))


fitparams_dir = apodir+'specs_fit_params/%s/'%(batchname)
fitspecs_dir = apodir+'specs_fit_specs/%s/'%(batchname)

os.system('mkdir -pv '+fitparams_dir)
os.system('mkdir -pv '+fitspecs_dir)


def specfn(params):
    iplate,imjd,ifn = params
    out = '/scratch/02977/jialiu/ApogeeLine/apo25m/{0}/{1}/{2}'.format(iplate,imjd,ifn) 
    out=out.replace(" ", "")
    #out.replace("\t", "")
    return  out

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
        try:
            vhelio = ihdulist[0].header['VHELIO']
        except Exception:
            vhelio = 0.0
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
    '''
    For apogee objects with APOGEE_ID=iapoid, get the list of visit spectrum
    '''
    #idx_visit = where(hdulist_visit[1].data['APOGEE_ID']==iapoid)[0]
    #fitsfnparams_arr = [[hdulist_visit[1].data[x][iidx] for x in ['PLATE','MJD','FILE']] for iidx in idx_visit]
    idx_visit = where(APOGEE_ID==iapoid)[0]
    fitsfnparams_arr = [[PLATE[iidx],MJD[iidx],FILE[iidx]] for iidx in idx_visit ]
    fitsfn_arr = map(specfn, fitsfnparams_arr)
    out_arr = map(prep_normed_spec, fitsfn_arr)
    data_spec_arr,data_err_arr,vhelio_arr,date_arr  = [[ivisit[i] for ivisit in out_arr] for i in range(4)]
    return data_spec_arr, data_err_arr, vhelio_arr, date_arr
    #return out_arr

def fit_visits (iapoid, N=3):
    data_spec_arr, data_err_arr, vhelio_arr,date_arr = prep_visit_spec(iapoid)
    
    p0, pcov, test1_spec = fitting.fit_normalized_spectrum_single_star_model(norm_spec=data_spec_arr[0], 
                    spec_err=data_err_arr[0], NN_coeffs_norm = NN_coeffs_norm, NN_coeffs_flux = NN_coeffs_flux, 
                    p0 = None, num_p0 = 2)
    
    popt_single, pcov, single_spec = fitting.fit_visit_spectra_sb1_model(norm_spectra=data_spec_arr, 
                    spec_errs=data_err_arr, NN_coeffs_norm = NN_coeffs_norm, NN_coeffs_flux = NN_coeffs_flux, 
                    v_helios=vhelio_arr, p0 = p0, num_p0 = 5)
    
    out = fitting.fit_visit_spectra_N(norm_spectra=data_spec_arr, spec_errs=data_err_arr, NN_coeffs_norm = NN_coeffs_norm, 
                          NN_coeffs_flux = NN_coeffs_flux, NN_coeffs_R = NN_coeffs_R, NN_coeffs_Teff2_logg2 = NN_coeffs_Teff2_logg2, 
                          v_helios=vhelio_arr, p0_single=popt_single[:6],N=N)
    popt, pcov, model_specs = out
    ############## add Teff and logg for secondaries
    q2arr=zeros(shape=(N,4))
    Teff1, logg1, feh, alphafe, vmacro1, dv1 = popt[:6]
    q2arr[0] = [1, Teff1, logg1, vmacro1]#
    for n in range(N-1): ## compute the Teff and logg for additional componenets
        q, vmacro2, dv2 = popt[6+n*3:9+n*3]
        Teff2, logg2 = spectral_model.get_Teff2_logg2_NN(labels = [Teff1, logg1, feh, q], 
        NN_coeffs_Teff2_logg2 = NN_coeffs_Teff2_logg2)
        q2arr[n+1] = [q, Teff2, logg2, vmacro2]#
        
    return array(data_spec_arr), array(data_err_arr), array(single_spec), array(model_specs), vhelio_arr, date_arr, popt_single, popt, pcov, q2arr

def plot_visit_fits (iapoid, data_spec_arr, data_err_arr, single_spec, model_specs3, model_specs10, 
                     date_arr, popt_single, popt3, popt10, ishow=0):
    #data_spec_arr, data_err_arr, single_spec, model_specs3, vhelio_arr, date_arr, popt_single, popt3, pcov, q2arr = out3
    #data_spec_arr, data_err_arr, single_spec, model_specs10, vhelio_arr, date_arr, popt_single, popt10, pcov, q2arr = out10
    istep=0.3
    ledges = [[15140, 15810], [15850, 16435], [16470,16955]]
    dof = [len(array(data_spec_arr).flatten())- ix for ix in (len(popt_single), len(popt3), len(popt10))]
    
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
            if j==len(date_arr)-1:
                axes[i].plot(wavelength[m], j*istep+N1_spec[m], color='b', lw=0.5, label = 'N=1(%.2f)'%(chi1))
                axes[i].plot(wavelength[m], j*istep+N3_spec[m], color='r', lw=0.5, label = 'N=3(%.2f)'%(chi3))
                axes[i].plot(wavelength[m], j*istep+N10_spec[m], color='g', lw=0.5, label = 'N=10(%.2f)'%(chi10))
            else:
                axes[i].plot(wavelength[m], j*istep+N1_spec[m], color='b', lw=0.5)
                axes[i].plot(wavelength[m], j*istep+N3_spec[m], color='r', lw=0.5)
                axes[i].plot(wavelength[m], j*istep+N10_spec[m], color='g', lw=0.5)

            axes[i].set_xlim(ledges[i])
            axes[i].set_ylim(0.7, len(date_arr)*istep+1)
    axes[0].legend(loc = 'best', frameon = 1, fontsize= 6, ncol=len(date_arr)/10+1)
    axes[1].set_ylabel('Normalized Flux')
    axes[-1].set_xlabel('Wavelenght A')
    axes[0].set_title('Visit Spec Fit %s'%(iapoid))
    if ishow: 
        show()
    else:
        savefig(apodir+'specs_fit_plot/MS_visit_joint/%s_fit.jpg'%(iapoid))
        #savefig(apodir+'specs_fit_plot/MS_visit_joint_pdf/%s_fit.pdf'%(iapoid))
        close()


    f, axes=subplots(3,1,figsize=(12,8))
    for j in range(len(date_arr)):
        data_spec, specerr, N1_spec, N3_spec, N10_spec = (data_spec_arr[j], 
                      data_err_arr[j], single_spec[j], model_specs3[j], model_specs10[j])
        for i in range(len(ledges)):
            m = (wavelength > ledges[i][0]) & (wavelength < ledges[i][1]) 
            axes[i].fill_between(wavelength[m], j*istep+1-specerr[m], j*istep+1+specerr[m], color='k',alpha=0.2)
            axes[i].plot(wavelength[m], j*istep+ones(sum(m)), color='k', lw=1, label = '%s'%(date_arr[j]))
            if j==len(date_arr)-1:  
                axes[i].plot(wavelength[m], j*istep+N1_spec[m]/data_spec[m], color='b', lw=0.3, label = 'N=1(%.2f)'%(chi1))
                axes[i].plot(wavelength[m], j*istep+N3_spec[m]/data_spec[m], color='r', lw=0.3, label = 'N=3(%.2f)'%(chi3))
                axes[i].plot(wavelength[m], j*istep+N10_spec[m]/data_spec[m], color='g', lw=0.3, label = 'N=10(%.2f)'%(chi10))
            else:
                axes[i].plot(wavelength[m], j*istep+N1_spec[m]/data_spec[m], color='b', lw=0.3)
                axes[i].plot(wavelength[m], j*istep+N3_spec[m]/data_spec[m], color='r', lw=0.3)
                axes[i].plot(wavelength[m], j*istep+N10_spec[m]/data_spec[m], color='g', lw=0.3)

            axes[i].set_xlim(ledges[i])
            axes[i].set_ylim(0.7, len(date_arr)*istep+1)
    axes[0].legend(loc = 'best', frameon = 1, fontsize= 6,ncol=len(date_arr)/10+1)
    axes[1].set_ylabel('Flux_model / Flux_data')
    axes[-1].set_xlabel('Wavelenght A')
    axes[0].set_title('Visit Spec Fit Ratio %s'%(iapoid))
    if ishow: 
        show()
    else:
        savefig(apodir+'specs_fit_plot/MS_visit_joint/%s_diff.jpg'%(iapoid))
        #savefig(apodir+'specs_fit_plot/MS_visit_joint_pdf/%s_diff.pdf'%(iapoid))
        close()

def plot_visit_fits_2comp (iapoid, data_spec_arr, data_err_arr, single_spec, model_specs2, 
                     date_arr, popt_single, popt2, ishow=0):
    fnpath='/scratch/02977/jialiu/ApogeeLine/specs_fit_plot/Giants_MS/'
    istep=0.3
    ledges = [[15140, 15810], [15850, 16435], [16470,16955]]
    dof = [len(array(data_spec_arr).flatten())- ix for ix in (len(popt_single), len(popt2))]
    
    chi1, chi2 = [sum((array(([single_spec, model_specs2][i])-array(data_spec_arr))/array(data_err_arr))**2)/dof[i] for i in range(2)]
    # record Teff vs chi2
    Teff1, logg1, feh, alphafe, vmacro1, dv1 = popt_single[:6]
    os.system('echo %s\t%s\t%s\t%s >> /scratch/02977/jialiu/ApogeeLine/testNN.txt'%(iapoid, 
                                                    Teff1, logg1, chi1))
    
    f, axes=subplots(3,1,figsize=(12,8))
    for j in range(len(date_arr)):        
        data_spec, specerr, N1_spec, N2_spec = (data_spec_arr[j], 
                      data_err_arr[j], single_spec[j], model_specs2[j])
        
        for i in range(len(ledges)):
            m = (wavelength > ledges[i][0]) & (wavelength < ledges[i][1]) 
            axes[i].fill_between(wavelength[m], j*istep+data_spec[m]-specerr[m], j*istep+data_spec[m]+specerr[m], color='k',alpha=0.1)
            axes[i].plot(wavelength[m], j*istep+data_spec[m], color='k', lw=0.5, label = '%s'%(date_arr[j]))
            if j==len(date_arr)-1:
                axes[i].plot(wavelength[m], j*istep+N1_spec[m], color='b', lw=0.5, label = 'N=1(%.2f)'%(chi1))
                axes[i].plot(wavelength[m], j*istep+N2_spec[m], color='r', lw=0.5, label = 'N=2(%.2f)'%(chi2))
            else:
                axes[i].plot(wavelength[m], j*istep+N1_spec[m], color='b', lw=0.5)
                axes[i].plot(wavelength[m], j*istep+N2_spec[m], color='r', lw=0.5)

            axes[i].set_xlim(ledges[i])
            axes[i].set_ylim(0.7, len(date_arr)*istep+1)
    axes[0].legend(loc = 'best', frameon = 1, fontsize= 6, ncol=len(date_arr)/10+1)
    axes[1].set_ylabel('Normalized Flux')
    axes[-1].set_xlabel('Wavelenght A')
    axes[0].set_title('Visit Spec Fit %s'%(iapoid))
    if ishow: 
        show()
    else:
        fnfig1='%s_fit.jpg'%(iapoid)
        savefig(fnpath+fnfig1)
        close()


    f, axes=subplots(3,1,figsize=(12,8))
    for j in range(len(date_arr)):
        data_spec, specerr, N1_spec, N2_spec = (data_spec_arr[j], 
                      data_err_arr[j], single_spec[j], model_specs2[j])
        for i in range(len(ledges)):
            m = (wavelength > ledges[i][0]) & (wavelength < ledges[i][1]) 
            axes[i].fill_between(wavelength[m], j*istep+1-specerr[m], j*istep+1+specerr[m], color='k',alpha=0.2)
            axes[i].plot(wavelength[m], j*istep+ones(sum(m)), color='k', lw=1, label = '%s'%(date_arr[j]))
            if j==len(date_arr)-1:  
                axes[i].plot(wavelength[m], j*istep+N1_spec[m]/data_spec[m], color='b', lw=0.3, label = 'N=1(%.2f)'%(chi1))
                axes[i].plot(wavelength[m], j*istep+N2_spec[m]/data_spec[m], color='r', lw=0.3, label = 'N=2(%.2f)'%(chi2))
            else:
                axes[i].plot(wavelength[m], j*istep+N1_spec[m]/data_spec[m], color='b', lw=0.3)
                axes[i].plot(wavelength[m], j*istep+N2_spec[m]/data_spec[m], color='r', lw=0.3)

            axes[i].set_xlim(ledges[i])
            axes[i].set_ylim(0.7, len(date_arr)*istep+1)
    axes[0].legend(loc = 'best', frameon = 1, fontsize= 6,ncol=len(date_arr)/10+1)
    axes[1].set_ylabel('Flux_model / Flux_data')
    axes[-1].set_xlabel('Wavelenght A')
    axes[0].set_title('Visit Spec Fit Ratio %s'%(iapoid))
    if ishow: 
        show()
    else:
        
        fnfig2='%s_diff.jpg'%(iapoid)
        savefig(fnpath+fnfig2)
        close()

    print 'uploading to dropbox'
    os.system('/work/02977/jialiu/Dropbox-Uploader/dropbox_uploader.sh upload %s %s'%(fnpath+fnfig1,fnfig1))
    os.system('/work/02977/jialiu/Dropbox-Uploader/dropbox_uploader.sh upload %s %s'%(fnpath+fnfig2,fnfig2))
        
        
def process_visit_fits(iapoid):
    'process all the MS visit spec, takes a loooong time'
    if iapoid[0]!='2':
        return 0 ## not a valid file
    #if os.path.isfile(fitparams_dir+'%s/%s_N%i_components.npy'%(iapoid,iapoid, list_components[-1])):
        ########## fit already processed:
        ##data_spec_arr, data_err_arr, single_spec, model_specs3, date_arr, popt_single, popt3, 
        #data_spec_arr, data_err_arr, single_spec, model_specs3 = load(fitspecs_dir+'%s/%s_N3_specs.npy'%(iapoid,iapoid))
        #data_spec_arr, data_err_arr, single_spec, model_specs10 = load(fitspecs_dir+'%s/%s_N10_specs.npy'%(iapoid,iapoid))
        #popt3=load(fitparams_dir+'%s/%s_N%i_params.npy'%(iapoid,iapoid, 3))
        ##popt10=load(fitparams_dir+'%s/%s_N%i_params.npy'%(iapoid,iapoid, 10))
        #popt_single=load(fitparams_dir+'%s/%s_N1_params.npy'%(iapoid,iapoid))
        #date_arr=load(fitparams_dir+'%s/%s_date.npy'%(iapoid,iapoid))
        ##plot_visit_fits (iapoid, data_spec_arr, data_err_arr, single_spec, model_specs3, model_specs10, 
        ##             date_arr, popt_single, popt3, popt10)
    out_arr = []
    os.system('mkdir -pv %s%s'%(fitparams_dir,iapoid))
    os.system('mkdir -pv %s%s'%(fitspecs_dir,iapoid))
    for iN in list_components:
        print iapoid, iN
        fn_components=fitparams_dir+'%s/%s_N%i_components.npy'%(iapoid,iapoid, iN)
        #if os.path.isfile(fn_components):
            #continue
        out = fit_visits(iapoid, N=iN)
        out_arr.append(out)
        ### save to files
        data_spec_arr, data_err_arr, single_spec, model_specs, vhelio_arr, date_arr, popt_single, popt, pcov, q2arr = out
        fimp = sum((abs(single_spec-data_spec_arr) - abs(model_specs-data_err_arr))/data_err_arr)
        fimp /= sum(abs(single_spec-model_specs)/data_err_arr)
        
        ############# 8.28, add chi^2 array
        dof = [len(array(data_spec_arr).flatten())+ ix for ix in (len(popt_single), len(popt))]
        chi1, chi2 = [sum((array(([single_spec, model_specs][i])-array(data_spec_arr))/array(data_err_arr))**2)/dof[i] for i in range(2)]
        Teff1, logg1, feh, alphafe, vmacro1, dv1 = popt_single[:6]
        save(fitspecs_dir+'%s/%s_N%i_specs.npy'%(iapoid,iapoid, iN), 
             [data_spec_arr, data_err_arr, single_spec, model_specs])
        save(fitparams_dir+'%s/%s_N%i_params.npy'%(iapoid,iapoid, iN), popt)
        save(fitparams_dir+'%s/%s_N%i_cov.npy'%(iapoid,iapoid, iN), pcov)
        save(fn_components, q2arr)
        if iN ==2:
            save(fitparams_dir+'%s/%s_vhelio.npy'%(iapoid,iapoid), vhelio_arr)        
            save(fitparams_dir+'%s/%s_date.npy'%(iapoid,iapoid), date_arr)
            save(fitparams_dir+'%s/%s_N1_params.npy'%(iapoid,iapoid), popt_single)
            os.system('echo %s\t%s\t%s\t%s\t%s >> /scratch/02977/jialiu/ApogeeLine/chi2_all.txt'%(iapoid, 
                                                    Teff1, logg1, chi1, chi2, fimp))
            ########## make a plot for likely binary stars
            #if batch == 'test':
            #if chi2/chi1<0.7:
                #plot_visit_fits_2comp (iapoid, data_spec_arr, data_err_arr, single_spec, model_specs, 
                     #date_arr, popt_single, popt, ishow=0)

pool=MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

pool.map(process_visit_fits, apoid_candidates)
pool.close()

print 'done-done-done'
