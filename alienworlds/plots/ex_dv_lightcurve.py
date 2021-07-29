from astropy.io import fits, ascii
import numpy as np
import os
from matplotlib import pyplot as plt

DIR_PATH='/media/ushasi/New Volume/personal documents/alienworlds/alienworlds_data/'
NO_OF_PLOTS=5

i=0
plt.style.use('seaborn-bright')
plt.rcParams['font.family']='serif'
def plot_data(filename):
    hdu = fits.open(DIR_PATH+filename)
    
    #time vs best LC version
    fig2,ax2=plt.subplots(1,1,figsize=(10,6))
    time=hdu[1].data['TIME']
    flux=hdu[1].data['LC_DETREND']
    ax2.set_title('Kepler_'+filename[4:13]+'\nMean zero median unity light curve (Q1-17)')
    ax2.set_xlabel('time')
    ax2.set_ylabel('flux')
    ax2.plot(time,flux,marker='.',ls='None', color='#006B38FF')
    #plt.savefig('kep_'+filename[4:13]+'_time.png')

    #phase vs zoomed transit
    fig3,ax3=plt.subplots(1,1,figsize=(10,6))
    time=hdu[1].data['TIME']
    flux=hdu[1].data['LC_WHITE']
    flux_ex=hdu[1].data['MODEL_WHITE']
    ax3.set_title('Kepler_'+filename[4:13]+'\nSingle transit vs model (Q1-17)')
    ax3.set_xlabel('time')
    ax3.set_ylabel('flux')
    ax3.plot(time,flux,marker='.',ls='None', color='#006B38FF',label='data')
    ax3.plot(time,flux_ex,color='#101820FF',label='model')
    ax3.legend()
    ax3.set_xlim(200,210)
    #plt.savefig('kep_'+filename[4:13]+'_zoomed.png')
    
    #phase vs model
    fig1,ax1=plt.subplots(1,1,figsize=(10,6))
    time=hdu[1].data['PHASE']
    obs=hdu[1].data['LC_WHITE']
    flux=hdu[1].data['MODEL_WHITE']
    ax1.set_title('Kepler_'+filename[4:13]+'\nModel light curve (Q1-17)')
    ax1.set_xlabel('phase')
    ax1.set_ylabel('flux')
    ax1.plot(time,obs,marker='.',ls='None', color='#006B38FF',label='data')
    ax1.plot(time,flux,color='#101820FF',label='model')
    ax1.legend()
    ax1.set_xlim(-1,1)
    #plt.savefig('kep_'+filename[4:13]+'_model_data_phase.png')

entries = os.scandir(DIR_PATH)

for entry in entries:
    i=i+1
    if(i==NO_OF_PLOTS): break
    plot_data(entry.name)

plt.show()
