import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.interpolate import interp1d
from astropy.io import fits,ascii
import tensorflow as tf
import GetLightcurves as gc

#ohkay... me trying out more and more ideas which i dont think will work. Now if we take this as a segmentation 
#problem at face value... its worth a shot. similar to the recinstruction idea but more fancy ig.
FILEPATH_FPS="E:\Masters_Project_Data\\alienworlds_fps\\"
FILEPATH_DATA="E:\Masters_Project_Data\\alienworlds_data\\"
FILEPATH_OTH="E:\Masters_Project_Data\\alienworlds_others\\" 
TRAINING_MODULE="../../processed_directories/"
CATALOG="../../Catalogs/"

def remove_nan(red_flux,bins):
    
    for i in range(0,len(red_flux)):
        if np.isnan(red_flux[i]):
            red_flux[i]=0

def get_instance_maps(pathin, pathout, bins, maxex=2):
    entries=os.listdir(pathin)
    av_entry=ascii.read(CATALOG+'autovetter_label.tab')
    ref_kepid=[('0000'+str(el)[:9])[-9:] for el in av_entry['kepid']]
    ref_label=av_entry['av_training_set']

    tick=0
    lightcurve=[]
    totmask=[]
    for el in entries:
        tick+=1 
        if(tick==5000): break
        hdu = fits.open(pathin+el)
        sz=len(hdu)-2
        
        try: flux = hdu[1].data['LC_DETREND']
        except: continue
        try: residue = hdu[len(hdu)-1].data['RESIDUAL_LC']
        except: continue

        loc=np.where(np.asarray(ref_kepid)==el[4:13])
        if(np.all(ref_label[loc[0]]=='UNK')): 
            print('only unk',np.asarray(ref_label[loc[0]]))
            continue

        ind_arr=np.arange(0,len(flux)-bins,bins)

        cum_mask=[]
        #check out a good chunk.
        phasec=np.zeros(len(ind_arr))
        for tce in range(1,sz+1):
            phase=hdu[tce].data['PHASE']
            i=0
            for ind in ind_arr:
                red_phase=phase[ind:ind+bins]
                ph_ind=[i for i in range(1,bins) if (red_phase[i]*red_phase[i-1]<0 and np.abs(red_phase[i]*red_phase[i-1])<0.001)]
                if(len(ph_ind)>0): phasec[i]+=1
                i+=1

        best_ind=ind_arr[np.argmax(np.asarray(phasec))]
        
        fc=0
        for tce in range(1,sz):
            red_flux=hdu[tce].data['LC_DETREND'][best_ind:best_ind+bins]
            red_res=hdu[tce+1].data['LC_DETREND'][best_ind:best_ind+bins]
            mask=np.asarray([1 if (np.isnan(red_res[x]) and not np.isnan(red_flux[x])) else 0 for x in range(0,len(red_res))])
            if(np.all(mask==0)): continue
            else: 
                fc+=1
                cum_mask.append(mask)

        red_flux=hdu[sz].data['LC_DETREND'][best_ind:best_ind+bins]
        red_res=hdu[sz+1].data['RESIDUAL_LC'][best_ind:best_ind+bins]
        mask=np.asarray([1 if (np.isnan(red_res[x]) and not np.isnan(red_flux[x])) else 0 for x in range(0,len(red_res))])
        if(np.all(mask==0)): continue
        else: 
            fc+=1
            cum_mask.append(mask)
        
        if(fc<maxex):
            for x in range(maxex-fc):
                cum_mask.append(np.zeros(bins))

        if(len(cum_mask)==0):
            print('miss',el[4:13])
            continue    

        if(fc>maxex):
            cum_mask=np.asarray(cum_mask)[:maxex,:]
        
        rflux=flux[best_ind:best_ind+bins]
        remove_nan(rflux,bins)
        lightcurve.append(rflux)
        totmask.append(np.asarray(cum_mask).reshape(-1))

        print('hit:',tick,el[4:13],np.asarray(cum_mask).shape,sz, max(phasec))
        
    net = np.asarray([[lightcurve[i],totmask[i]] for i in range(0,len(lightcurve))], dtype='object')
    gc.write_tfr_record(pathout+'instance_maps',net,
            ['input','mask'],['ar','ar'],['float32','bool'])
    print(np.asarray(lightcurve).shape,np.asarray(totmask).shape)

def plot_instance_maps(pathin,maxex):
    lc, maps = gc.read_tfr_record(pathin,['input','mask'],['ar','ar'],[tf.float32,tf.bool])
    fig, ax = plt.subplots(maxex,maxex, figsize=(20,20))
    i=0
    j=0
    for x in range(0,maxex**2):
        plmp=maps[x].reshape(3,4000)
        ax[i][j].plot(plmp[0]*min(lc[x]))
        ax[i][j].plot(plmp[1]*min(lc[x]))
        ax[i][j].plot(plmp[2]*min(lc[x]))
        ax[i][j].plot(lc[x])
        i+=1
        if(i==maxex):
            i=0
            j+=1


get_instance_maps(FILEPATH_FPS,TRAINING_MODULE,4000,2)
plot_instance_maps(TRAINING_MODULE+'instance_maps',3)
plt.show()