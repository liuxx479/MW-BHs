import astropy.io.fits as pyfits
from scipy import *
import numpy as np
from pylab import *
import matplotlib as mpl
import matplotlib.pylab as plt
import os

plot_color = 0
plot_highV_spec = 0

hdulist = pyfits.open('allStar-l30e.2.fits')
idx_all=where( (hdulist[1].data['LOCATION_ID']>1))[0]
idx_noTelluric=where( (hdulist[1].data['LOCATION_ID']>1) & logical_not (hdulist[1].data['APOGEE_TARGET2'] & 2**9))[0]
# Number of Telluric standards = 17293
rvCC = hdulist[1].data['RV_CCFWHM']
rvAUTO = hdulist[1].data['RV_AUTOFWHM']

def download(ii, download=1):
    LOCATION_ID = hdulist[1].data['LOCATION_ID'][ii]
    FILE = hdulist[1].data['FILE'][ii]   
    if download and not os.path.isfile('spec/'+FILE):
        bashCommand = 'wget  https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/stars/apo25m/%s/%s; mv %s spec'%(LOCATION_ID, FILE, FILE)#--spider
        os.system(bashCommand)
    return LOCATION_ID, FILE

def read_spec(fitsfile):
    hdulist = pyfits.open('spec/'+fitsfile)
    ispec = hdulist[1].data
    if len(ispec.shape)>1:
        ispec=ispec[0]
    lambda0 = hdulist[1].header['CRVAL1']
    delta_lambda = hdulist[1].header['CDELT1']
    shift = hdulist[0].header['VHELIO']
    
    lambda_arr = 10**(lambda0+arange(len(ispec))*delta_lambda)
    #shift_lambda = shift/3e5*lambda_arr
    #lambda_arr += shift_lambda
    return lambda_arr, ispec

#ibins=linspace(0,200,51)
#f=figure(figsize=(6,6))
#ax=f.add_subplot(111)
#ax.hist2d(rvCC[idx], rvAUTO[idx], bins=ibins,norm=mpl.colors.LogNorm())#cmap=mpl.cm.gray
#ax.set_xlabel('V_FWHM_star (km/s)')
#ax.set_ylabel('V_FWHM_template (km/s)')
#savefig('RV_FWHM_hist2d_DR13_NoTelluric.png');close()

#ibins=linspace(0,600,101)
#f=figure(figsize=(8,6))
#ax=f.add_subplot(111)
#ax.hist(rvCC[idx_noTelluric], bins=ibins,histtype='step',log=1,label='Remove Tellurics')
#ax.hist(rvCC[idx_all], bins=ibins,histtype='step',log=1,lw=3, label='All')

#ax.set_xlabel('V_FWHM_star (km/s)')
#ax.set_ylabel('PDF')
#ax.legend(frameon=0)
#savefig('RV_FWHM_hist1d_DR13_NoTelluric.png');close()

idx_high_noTelluric = where( (rvCC>100) & logical_not (hdulist[1].data['APOGEE_TARGET2'] & 2**9))[0]
seed(10027)
idx_low_noTelluric = where( (rvCC<20) & logical_not (hdulist[1].data['APOGEE_TARGET2'] & 2**9))[0]
seed(10027)

#idx_high = where(rvCC>100)[0]
idx_high_select = idx_high_noTelluric[randint(0,len(idx_high_noTelluric)-1,10)]
loc_file_arr_hi = array([download(ii) for ii in idx_high_select])

#idx_low = where(rvCC<50)[0]
idx_low_select = idx_low_noTelluric[randint(0,len(idx_low_noTelluric)-1,10)]
loc_file_arr_lo = array([download(ii) for ii in idx_low_select])

########## sample spectrum 
#for icut in range(len(lambda_cut)-1):
#f=figure(figsize=(12,12))
#ax1=f.add_subplot(211)
#ax2=f.add_subplot(212)
#for ii in arange(len(loc_file_arr_lo)):
    #for jj in range(2):
        #ax=[ax1,ax2][jj]
        #ifile=[loc_file_arr_hi, loc_file_arr_lo][jj][ii][1]
        #iwave, ispec = read_spec(ifile)
        #ax.plot(iwave,ispec-mean(ispec[(iwave<16000)&(iwave>15900)])+2000*ii,label=ifile)
        #ax.set_xlabel('Wavelenght A')
        #ax.set_ylabel('flux')
        #ax.set_title('V_FWHM %s km/s'%('> 100', '< 20')[jj])
        #ax.set_ylim(0,2.5e4)
##legend(frameon=0,loc=0)
#plt.tight_layout()
#savefig('spec_sample.png');close()
######################

#target_mat = zeros( (len(idx_high_noTelluric), 23))
#target_mat[:,0] =  hdulist[1].data['RV_CCFWHM'][idx_high_noTelluric]
#for itarget in arange(1,23):
    #target_mat[:,itarget] = ((hdulist[1].data['APOGEE_TARGET2'][idx_high_noTelluric] & 2**itarget).astype(bool)).astype(int)

#savetxt('APOGEE_TARGET2_highV.txt', target_mat, delimiter='\t')
    
#check=((hdulist[1].data['APOGEE_TARGET2'][idx_high_noTelluric] & 2**31).astype(bool)).astype(int)
#savetxt('APOGEE_TARGET2Checked_highV.txt', check)

######### plot color ##########
if plot_color:
    MagJ = hdulist[1].data['J'][idx_high_noTelluric]
    MagH = hdulist[1].data['H'][idx_high_noTelluric]
    MagK = hdulist[1].data['K'][idx_high_noTelluric]

    idx_young_cluster = where((hdulist[1].data['APOGEE_TARGET2'][idx_high_noTelluric] & 2**13)!=0)[0]
    idx_not_young_cluster = where((hdulist[1].data['APOGEE_TARGET2'][idx_high_noTelluric] & 2**13)==0)[0]

    hist2d(hdulist[1].data['J']-hdulist[1].data['H'], hdulist[1].data['H']-hdulist[1].data['K'],range=[[-0.3,1],[-0.3,1]],bins=100,cmap='Greys')
    scatter( (MagJ-MagH)[idx_young_cluster], (MagH-MagK)[idx_young_cluster], color='r',s=1,label='young clusters members (%i)'%(len(idx_young_cluster)))
    scatter( (MagJ-MagH)[idx_not_young_cluster], (MagH-MagK)[idx_not_young_cluster], color='b',s=1,label='other large vel. objects (%s)'%(len(idx_not_young_cluster)))
    legend(frameon=0)
    xlabel('J-H')
    ylabel('H-K')
    xlim(-0.2,1)
    ylim(-0.2,1)
    savefig('color_2MASS.png');close()
    
if plot_highV_spec:
    #file_arr=array([download(ii, download=1) for ii in idx_high_noTelluric])
    filename_arr = hdulist[1].data['FILE'][idx_high_noTelluric]
    for ii in arange(930,len(filename_arr)):#arange(5):#
        #print ii
        ifile=filename_arr[ii]
        FWHM = hdulist[1].data['RV_CCFWHM'][idx_high_noTelluric][ii]
        yes_young = (hdulist[1].data['APOGEE_TARGET2'][idx_high_noTelluric][ii] & 2**13)
        figname='highV/YC%i_%05d_%s.png'%(yes_young,FWHM, ifile[:-5])
        if os.path.isfile(figname) or ii in[247,248,668,709,710,711,712,714,793,800,957]:
            continue
        try:
            iwave, ispec = read_spec(ifile)
            idx1=where((iwave>15500)&(iwave<15600))
            idx2=where((iwave>16000)&(iwave<16100))
            f=figure(figsize=(10,6))
            ax1=f.add_subplot(311)
            ax2=f.add_subplot(312)
            ax3=f.add_subplot(313)
            ax1.plot(iwave,ispec)
            ax1.set_ylabel('Flux')        
            ax2.set_ylabel('Flux')
            ax3.set_ylabel('Flux')
            #ax2.set_xlim (15500, 15600)
            ax2.plot(iwave[idx1],ispec[idx1])
            ax3.plot(iwave[idx2],ispec[idx2])        
            ax3.set_xlabel('Wavelenght A')
            ax1.set_title('%s     %.1fkm/s   inYoungCluster=%i'%(ifile[:-5],FWHM,yes_young))
            plt.tight_layout()
            savefig(figname)
            close()
        except Exception:
            print ii
            pass
        
