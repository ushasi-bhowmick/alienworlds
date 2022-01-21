import re
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import count
from numpy.core.fromnumeric import cumsum
from numpy.core.numeric import count_nonzero
from numpy.lib.function_base import median
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

def smoothbin(arr):
    for m in range(5,len(arr)-5):
        if(arr[m]==0):
            if(arr[m-1]==-1 and arr[m+1]==-1): 
                arr[m]=-1
            elif(arr[m-2]==-1 and arr[m+2]==-1): 
                arr[m]=-1
            elif(arr[m-3]==-1 and arr[m+3]==-1): 
                arr[m]=-1
            elif(arr[m-4]==-1 and arr[m+4]==-1): 
                arr[m]=-1
            elif(arr[m-5]==-1 and arr[m+5]==-1): 
                arr[m]=-1

def remove_nan(red_flux,bins):
    '''
    for i in range(0,len(red_flux)):
        if(np.isnan(red_flux[i])):
            t=1
            try:
                while(np.isnan(red_flux[i-t]) or np.isnan(red_flux[i+t])):    
                    if(i-t == 0):
                        red_flux[i]=red_flux[i+t]
                        break
                    elif(i+t == bins-1): 
                        red_flux[i]=red_flux[i-t]
                        break
                    t+=1
                red_flux[i]=(red_flux[i-t]+red_flux[i+t])/2
            except:
                red_flux[i]=0'''
    for i in range(0,len(red_flux)):
        if np.isnan(red_flux[i]):
            red_flux[i]=0


def remove_nan_2(red_flux):
    clean = [el for el in red_flux if not np.isnan(el)]
    mid = np.median(clean)
    std = np.std(clean)
    counts = np.isnan(red_flux).sum()
    
    noise_arr = np.random.normal(mid, std*0.5, counts)
    
    j=0
    for i in range(0,len(red_flux)):
        if(np.isnan(red_flux[i])): 
            #print(red_flux[i], noise_arr[j])
            red_flux[i]=noise_arr[j]
            j+=1
          

def inst_segment(pathin,pathout,bins):
    entries=os.listdir(pathin)
    tick=0
    for el in entries:
        tick+=1
        #if(tick==30): break
        hdu = fits.open(pathin+el)
        flux = hdu[1].data['LC_DETREND']
        phase = hdu[1].data['PHASE']
        #good idea to center transits
        ind_arr=[i for i in range(0,len(phase)-1) if (phase[i]*phase[i-1]<0 and np.abs(phase[i]*phase[i-1])<0.0001)]

        lightcurve=[]
        segmap=[]
        for ind in ind_arr:
            red_flux=np.asarray(flux[ind-int(bins/2):ind+int(bins/2)])
            count_nan=np.isnan(red_flux).sum()
            if(count_nan/bins > 0.05): continue
            
            try: res=hdu[len(hdu)-1].data['RESIDUAL_LC'][ind-int(bins/2):ind+int(bins/2)] 
            except: continue
            if(len(res)==0): continue

            segmap =[1 if(np.isnan(res[m]) and not np.isnan(red_flux[m])) else 0 for m in range(0,bins)] 

            remove_nan(red_flux,bins)
            
            lightcurve=red_flux
            break
        if(len(lightcurve)==0):
            print("miss 2",el[4:13])
            continue
        segmap=np.asarray(segmap).reshape(bins)
        lightcurve=np.asarray(lightcurve).reshape(bins)
        np.savetxt(pathout+el[4:13]+'_'+str(len(hdu)-2),(lightcurve,segmap),delimiter=' ')
        print(tick,'hit:',el[4:13],np.array(lightcurve).shape,segmap.shape)

def inst_seg_classifier(pathin,pathout,bins):
    entries=os.listdir(pathin)
    tick=0
    for el in entries:
        tick+=1
        #if(tick==30): break
        hdu = fits.open(pathin+el)
        flux = hdu[1].data['LC_DETREND']

        lightcurve=[]
        for ind in range(int(bins/2),len(flux)-int(bins/2),bins):
            red_flux=np.asarray(flux[ind-int(bins/2):ind+int(bins/2)])
            count_nan=np.isnan(red_flux).sum()
            if(count_nan/bins > 0.1): 
                #print('woopss')
                continue
            remove_nan(red_flux,bins)
            
            lightcurve.append(red_flux)
        if(len(lightcurve)==0):
            print("miss 2",el[4:13])
            continue
       
        np.savetxt(pathout+el[4:13]+'_'+str(len(hdu)-2),lightcurve,delimiter=' ')
        print(tick,'hit:',el[4:13],np.array(lightcurve).shape)
    

def sem_segment(pathin,pathout,bins):
    entries=os.listdir(pathin)
    tick=0
    for el in entries:
        tick+=1
        if(tick==5): break
        hdu = fits.open(pathin+el)
        #if(len(hdu)>3): 
        #    print('too many tce:',el[4:13])
        #    continue
        max_tce = len(hdu)

        for tce in range(1,max_tce-1):
            flux = hdu[tce].data['LC_DETREND']
            phase = hdu[tce].data['PHASE']
            try: tdur = int(hdu[tce].header['TDUR']*60/30)
            except: tdur=0
            try: tpd= int(hdu[tce].header['TPD']*60*24/30)
            except: tpd=0
            if(tpd<tdur*20): tdur=0.1
            #good idea to center transits
            ind_arr=[i for i in range(0,len(phase)-1) if (phase[i]*phase[i-1]<0 and np.abs(phase[i]*phase[i-1])<0.001)]

            lightcurve=[]
            models=[]
            models_true=[]
            #tces=[]
            checks=0
            for ind in ind_arr:
                red_flux=flux[ind-int(bins/2):ind+int(bins/2)]
                red_phase=phase[ind-int(bins/2):ind+int(bins/2)]
                red_flux=np.array(red_flux)
                count_nan=np.isnan(red_flux).sum()
                if(count_nan/bins > 0.05): 
                    print('nanissue?')
                    continue

                remove_nan(red_flux,bins)
                if(len(red_flux)<bins): continue
            
                #models=[hdu[i].data['MODEL_WHITE'][ind-int(bins/2):ind+int(bins/2)] for i in range(1,max_tce-1)]
                models=hdu[tce].data['MODEL_WHITE'][ind-int(bins/2):ind+int(bins/2)]
                if(len(models)==0): 
                    print('modelloaderr')
                    continue

                remove_nan(models,bins)
                models=np.array(models)
                #print('model check:',models.shape,len(red_flux))
                models_true=models
                thres=np.median(models)-0.5*np.std(models)
                thres2=np.median(models)+0.5*np.std(models)
                models=[-1 if (x<thres or x>thres2) else 0 for x in models]
            
                #try: models = [min(models[:,i]) for i in range(0,bins)]
                #except: continue
                if(np.all(np.array(models)> -0.0001)): 
                    print("miss 1",el[4:13])
                    checks+=1
                    if(checks==3): break
                    else: continue
                smoothbin(models)
                smoothbin(models)
                smoothbin(models)
                #print(len(models),len(red_phase))
                for x in range(int(tdur*20),len(models)-int(tdur*20)):
                    if(red_phase[x]*red_phase[x-1]<0 and np.abs(red_phase[x]*red_phase[x-1])<0.001):
                        models[x-int(tdur*20):x+int(tdur*20)]=-np.ones(int(40*tdur))
                    
                lightcurve=red_flux
                break

            if(len(lightcurve)==0):
                print("miss 2",el[4:13])
                continue
            models=np.array(models).reshape(bins)
            models_true=np.array(models_true).reshape(bins)
            lightcurve=np.array(lightcurve).reshape(bins)
            np.savetxt(pathout+el[4:13]+'_'+str(tce),(lightcurve,models,models_true),delimiter=' ')
            print(tick,'hit:',el[4:13],np.array(lightcurve).shape,models.shape)
        
def sem_segment_tot(pathin, pathout, bins):
    entries=os.listdir(pathin)
    av_entry=ascii.read(CATALOG+'autovetter_label.tab')
    #av_entry=ascii.read(CATALOG+'robovetter_label.dat')
    av_pl=np.array(av_entry['tce_plnt_num'])
    ref_kepid=[('0000'+str(el)[:9])[-9:] for el in av_entry['kepid']]
    ref_label=av_entry['av_pred_class']
    #ref_label=av_entry['label']
    tick=0
    for el in entries:
        tick+=1
        #if(tick==1600): break
        hdu = fits.open(pathin+el)
        if(len(hdu)>4): 
            print("too many tce:")
            #continue
        
        flux = hdu[1].data['LC_DETREND']
        try: residue = hdu[len(hdu)-1].data['RESIDUAL_LC']
        except: continue
        #get a preliminary phase... must center at least one transit
        phase = hdu[1].data['PHASE']
        ind_arr=[i for i in range(int(bins/2),len(phase)-int(bins/2)) if (phase[i]*phase[i-1]<0 and np.abs(phase[i]*phase[i-1])<0.001)]
        #print(len(flux),len(phase),ind_arr)

        lightcurve=[]
        mask=[]
        for ind in ind_arr:
            red_flux=flux[ind-int(bins/2):ind+int(bins/2)]
            red_res=residue[ind-int(bins/2):ind+int(bins/2)]
            if(len(red_flux)==0): continue

            #get a clean chunk
            count_nan=np.isnan(red_flux).sum()
            if(count_nan/bins > 0.1): 
                continue

            mask=np.asarray([[1,1,1] if (np.isnan(red_res[x]) and not np.isnan(red_flux[x])) else [0,0,1] for x in range(0,len(red_res))])
            if(np.asarray([(np.asarray(m)==np.asarray([0,0,1])).all() for m in mask]).all()): 
                print('skip')
                continue
            trackrog=np.where(np.isnan(red_flux))
            remove_nan(red_flux,bins)
            
            for tce in range(2,len(hdu)-1):
                try:
                    loc=np.where(np.asarray(ref_kepid)==el[4:13])
                    loc_f=[m for m in loc[0] if str(av_pl[m])==str(tce-1)]
                    if(len(loc_f)==0):
                        print("not in catalog:",tce-1)
                        if(tce-1 == 1 ): break
                        label=[0,0,1]
                    else:
                        if(ref_label[loc_f[0]]=='PC'): label=[1,0,0]
                        elif(ref_label[loc_f[0]]=='AFP'): label=[0,1,0]
                        else: label=[0,1,0]
                except ValueError as ve:
                    print("miss ind:",el[4:13])
                    label=[0,0,1]
                
                new_flux=hdu[tce].data['LC_DETREND']
                new_flux=new_flux[ind-int(bins/2):ind+int(bins/2)]

                for b in trackrog[0]:
                    if(np.isnan(new_flux[b])): 
                        new_flux[b]=0
                for m in range(0,len(mask)):
                    if(np.isnan(new_flux[m]) and (np.asarray(mask[m])==np.asarray([1,1,1])).all()):
                        mask[m]=label

            try:
                loc=np.where(np.asarray(ref_kepid)==el[4:13])
                loc_f=[m for m in loc[0] if str(av_pl[m])==str(len(hdu)-2)]
                if(len(loc_f)==0):
                    print("not in catalog:",len(hdu)-2)
                    if(len(hdu)-2 == 1): break
                    label=[0,0,1]
                else:
                    if(ref_label[loc_f[0]]=='PC'): label=[1,0,0]
                    elif(ref_label[loc_f[0]]=='AFP'): label=[0,1,0]
                    else: label=[0,0,1]
            except ValueError as ve:
                print("miss ind:",el[4:13])
                label=[0,0,1]
            for m in range(0,len(mask)):
                if((np.asarray(mask[m])==np.asarray([1,1,1])).all()):
                    mask[m]=label

            lightcurve=red_flux
            break

        if(len(lightcurve)==0):
            print('miss',el[4:13])
            continue      
        np.savetxt(pathout+'xlabel/'+el[4:13],lightcurve,delimiter=' ')
        np.savetxt(pathout+'ylabel/'+el[4:13],mask,delimiter=' ')
        print(tick,'hit:',el[4:13],np.array(lightcurve).shape,mask.shape,len(hdu)-2)

def sem_segment_clean(pathin, pathout, bins):
    entries=os.listdir(pathin)
    av_entry=ascii.read(CATALOG+'robovetter_label.dat')
    av_pl=np.array(av_entry['tce_plnt_num'])
    ref_kepid=[('0000'+str(el)[:9])[-9:] for el in av_entry['kepid']]
    ref_label=av_entry['label']
    tick=0
    for el in entries:
        tick+=1
        #if(tick==20): break
        hdu = fits.open(pathin+el)
        
        flux = hdu[1].data['LC_DETREND']
        try: residue = hdu[len(hdu)-1].data['RESIDUAL_LC']
        except: continue
        try: tdurs=np.asarray([hdu[x].header['TDUR'] for x in range(1,len(hdu)-1)])
        except: 
            print("woah")
            continue
        if(np.all(tdurs*2<3)): 
            print("will lose out this one")
            continue
        #get a preliminary phase... must center at least one transit
        phase = hdu[1].data['PHASE']
        ind_arr=[i for i in range(int(bins/2),len(phase)-int(bins/2)) if (phase[i]*phase[i-1]<0 and np.abs(phase[i]*phase[i-1])<0.001)]
        #print(len(flux),len(phase),ind_arr)

        lightcurve=[]
        mask=[]
        for ind in ind_arr:
            red_flux=flux[ind-int(bins/2):ind+int(bins/2)]
            red_res=residue[ind-int(bins/2):ind+int(bins/2)]
            if(len(red_flux)==0): continue

            #get a clean chunk
            count_nan=np.isnan(red_flux).sum()
            if(count_nan/bins > 0.2): 
                continue

            mask=np.asarray([[1,1,1] if (np.isnan(red_res[x]) and not np.isnan(red_flux[x])) else [0,0,1] for x in range(0,len(red_res))])
            if(np.asarray([(np.asarray(m)==np.asarray([0,0,1])).all() for m in mask]).all()): 
                print('skip')
                continue
            trackrog=np.where(np.isnan(red_flux))
            remove_nan(red_flux,bins)
            
            for tce in range(2,len(hdu)-1):
                try:
                    loc=np.where(np.asarray(ref_kepid)==el[4:13])
                    loc_f=[m for m in loc[0] if str(av_pl[m])==str(tce-1)]
                    if(len(loc_f)==0):
                        print("not in catalog:",tce-1)
                        label=[0,1,0]
                    else:
                        if(ref_label[loc_f[0]]=='CONFIRMED'): label=[1,0,0]
                        elif(ref_label[loc_f[0]]=='CANDIDATE'): label=[1,0,0]
                        else: label=[0,1,0]
                except ValueError as ve:
                    print("miss ind:",el[4:13])
                    label=[0,0,1]
                
                new_flux=hdu[tce].data['LC_DETREND']
                new_flux=new_flux[ind-int(bins/2):ind+int(bins/2)]

                for b in trackrog[0]:
                    if(np.isnan(new_flux[b])): 
                        new_flux[b]=0
                for m in range(0,len(mask)):
                    if(np.isnan(new_flux[m]) and (np.asarray(mask[m])==np.asarray([1,1,1])).all()):
                        mask[m]=label

            try:
                loc=np.where(np.asarray(ref_kepid)==el[4:13])
                loc_f=[m for m in loc[0] if str(av_pl[m])==str(len(hdu)-2)]
                if(len(loc_f)==0):
                    print("not in catalog:",len(hdu)-2)
                    label=[0,1,0]
                else:
                    if(ref_label[loc_f[0]]=='CONFIRMED'): label=[1,0,0]
                    elif(ref_label[loc_f[0]]=='CANDIDATE'): label=[1,0,0]
                    else: label=[0,1,0]
            except ValueError as ve:
                print("miss ind:",el[4:13])
                label=[0,0,1]
            for m in range(0,len(mask)):
                if((np.asarray(mask[m])==np.asarray([1,1,1])).all()):
                    mask[m]=label

            red_flux=np.asarray([np.mean(red_flux[m:m+3]) for m in range(0,bins,3)])
            mask=np.asarray([[max(mask[m:m+3,0]),max(mask[m:m+3,1]),min(mask[m:m+3,2])] for m in range(0,bins,3)])
            if(np.all(mask[:,0]==1) or np.all(mask[:,1]==1) or np.all(mask[:,2]==1)): 
                print("washed out")
                continue
            lightcurve=red_flux
            break

        if(len(lightcurve)==0):
            print('miss',el[4:13])
            continue      
        np.savetxt(pathout+'xlabel/'+el[4:13],lightcurve,delimiter=' ')
        np.savetxt(pathout+'ylabel/'+el[4:13],mask,delimiter=' ')
        print(tick,'hit:',el[4:13],np.array(lightcurve).shape,mask.shape,len(hdu)-2)

def sem_segment_one(pathin, pathout, bins):
    entries=os.listdir(pathin)
    av_entry=ascii.read(CATALOG+'robovetter_label.dat')
    av_pl=np.array(av_entry['tce_plnt_num'])
    ref_kepid=[('0000'+str(el)[:9])[-9:] for el in av_entry['kepid']]
    ref_label=av_entry['label']
    tick=0
    for el in entries:
        tick+=1
        #if(tick==20): break
        hdu = fits.open(pathin+el)
        
        flux = hdu[1].data['LC_DETREND']
        try: residue = hdu[len(hdu)-1].data['RESIDUAL_LC']
        except: continue
        try: tdurs=np.asarray([hdu[x].header['TDUR'] for x in range(1,len(hdu)-1)])
        except: 
            print("woah")
            continue
        npixper=hdu[1].header['TPERIOD']*24*2
        if(np.all(tdurs*2<3) or npixper>bins/2): 
            print("will lose out this one")
            continue
        #get a preliminary phase... must center at least one transit
        phase = hdu[1].data['PHASE']
        ind_arr=[i for i in range(int(bins/2),len(phase)-int(bins/2)) if (phase[i]*phase[i-1]<0 and np.abs(phase[i]*phase[i-1])<0.001)]
        #print(len(flux),len(phase),ind_arr)

        lightcurve=[]
        mask=[]
        for ind in ind_arr:
            red_flux=flux[ind-int(bins/2):ind+int(bins/2)]
            red_res=residue[ind-int(bins/2):ind+int(bins/2)]
            #red_phase=phase[ind-int(bins/2):ind+int(bins/2)]
            if(len(red_flux)==0): continue

            #get a clean chunk
            count_nan=np.isnan(red_flux).sum()
            if(count_nan/bins > 0.2): 
                continue

            mask=np.asarray([[1,1,1] if (np.isnan(red_res[x]) and not np.isnan(red_flux[x])) else [0,0,1] for x in range(0,len(red_res))])
            if(np.asarray([(np.asarray(m)==np.asarray([0,0,1])).all() for m in mask]).all()): 
                print('skip')
                continue
            trackrog=np.where(np.isnan(red_flux))
            #track_phase=np.asarray([i for i in range(1,bins) if (red_phase[i]*red_phase[i-1]<0 and np.abs(red_phase[i]*red_phase[i-1])<0.001)])
            remove_nan(red_flux,bins)
            
            try:
                loc=np.where(np.asarray(ref_kepid)==el[4:13])
                refs=np.asarray([ref_label[m] for m in loc[0]])
                if((refs=="CONFIRMED").any() and (refs=="FPS").any()): 
                    print("snitch")
                    break
                if((refs=="CANDIDATE").any() and (refs=="FPS").any()): 
                    print("snitch")
                    break
                loc_f=[m for m in loc[0] if str(av_pl[m])==str(1)]
                if(len(loc_f)==0):
                    print("not in catalog:",len(hdu)-2)
                    continue
                else:
                    if(ref_label[loc_f[0]]=='CONFIRMED'): label=[1,0,0]
                    elif(ref_label[loc_f[0]]=='CANDIDATE'): label=[1,0,0]
                    else: label=[0,1,0]
            except:
                print("miss ind:",el[4:13])
                label=[0,0,1]

            '''
            if(len(hdu)>3):
                new_flux=hdu[2].data['LC_DETREND']
                new_flux=new_flux[ind-int(bins/2):ind+int(bins/2)]
                for m in range(0,len(mask)):
                    if((np.asarray(mask[m])==np.asarray([1,1,1])).all() and not np.isnan(new_flux[m]) 
                        and not np.asarray(trackrog==m).any()):
                        mask[m]=label
                    else: mask[m]=[0,0,1]'''
            
            for m in range(0,len(mask)):
                if((np.asarray(mask[m])==np.asarray([1,1,1])).all() and not np.asarray(trackrog==m).any()):
                        mask[m]=label

            '''
            wash=2
            red_flux=np.asarray([np.mean(red_flux[m:m+wash]) for m in range(0,bins,wash)])
            mask=np.asarray([[max(mask[m:m+wash,0]),max(mask[m:m+wash,1]),min(mask[m:m+wash,2])] for m in range(0,bins,wash)])
            if(np.all(mask[:,0]>0.5) or np.all(mask[:,1]>0.5) or np.all(mask[:,2]>0.5)): 
                print("washed out")
                continue'''
            lightcurve=red_flux
            break

        if(len(lightcurve)==0):
            print('miss',el[4:13])
            continue      
        np.savetxt(pathout+'xlabel/'+el[4:13],lightcurve,delimiter=' ')
        np.savetxt(pathout+'ylabel/'+el[4:13],mask,delimiter=' ')
        #np.savetxt(pathout+el[4:13],track_phase,delimiter=' ')
        print(tick,'hit:',el[4:13],np.array(lightcurve).shape,mask.shape,len(hdu)-2)

def getids():
    tfr_testdata = tf.data.TFRecordDataset(['../../training_data/sem_seg_av_zer_aug_test']) 
    testdata = tfr_testdata.map(_parse_tfr_element)
    entries=os.listdir(FILEPATH_FPS)
    ID = [instance[3] for instance in testdata]
    ID = [ID[i].numpy() for i in range(0,len(ID))]
    ID = [str(ID[i])[2:11] for i in range(0,len(ID))]
    neID=[]
    for x in ID:
        temp =[el for el in entries if el.find(x)>0]
        neID.append(temp[0])
    return(neID)

#trying to accumulate all of the worthwhile information...
def compr_sem_seg(pathin,pathout,bins):
    entries=os.listdir(pathin)
    av_entry=ascii.read(CATALOG+'autovetter_label.tab')
    av_pl=np.array(av_entry['tce_plnt_num'])
    ref_kepid=[('0000'+str(el)[:9])[-9:] for el in av_entry['kepid']]
    #ref_label=av_entry['av_training_set']
    ref_label=av_entry['av_pred_class']

    #newids= getids()

    tick=54
    for el in entries:
        tick+=1 
        #if(tick==10): break
        hdu = fits.open(pathin+el)

        try: loc=np.where(np.asarray(ref_kepid)==el[4:13])[0]
        except: continue
        loc = np.asarray([ref_label[m] for m in loc])
        print(loc)
        if(not np.any(loc=='PC')): continue

        
        flux = hdu[1].data['LC_DETREND']
        try: residue = hdu[len(hdu)-1].data['RESIDUAL_LC']
        except: continue

        #get a preliminary phase... must center at least one transit
        ind_arr=np.arange(0,len(flux)-bins,bins)

        lightcurve=[]
        totmask=[]
        counts=[]
        for ind in ind_arr:
            counting=[0,0]
            red_flux=flux[ind:ind+bins]
            red_res=residue[ind:ind+bins]

            #if(len(red_flux)==0): continue

            #get a clean chunk
            #count_nan=np.isnan(red_flux).sum() 
            #if(count_nan/bins > 0.1): 
            #    continue

            mask=np.asarray([[1,1,1] if (np.isnan(red_res[x]) and not np.isnan(red_flux[x])) else [0,0,1] for x in range(0,len(red_res))])
            #if(np.asarray([(np.asarray(m)==np.asarray([0,0,1])).all() for m in mask]).all()): 
                #print('skip')
            #    continue
            trackrog=np.where(np.isnan(red_flux))[0]
            remove_nan(red_flux,bins)
            
            for tce in range(1,len(hdu)-2):
                red_phase=hdu[tce].data['PHASE'][ind:ind+bins]
                ph_ind=[i for i in range(1,bins) if (red_phase[i]*red_phase[i-1]<0 and np.abs(red_phase[i]*red_phase[i-1])<0.001)]
                if(len(ph_ind)==0): 
                    print('no rel phase')
                    continue
                
                #set an index
                try:
                    loc=np.where(np.asarray(ref_kepid)==el[4:13])
                    loc_f=[m for m in loc[0] if str(av_pl[m])==str(tce)]
                    if(len(loc_f)==0):
                        #print("not in catalog:",tce)
                        #if(tce == 1 ): break
                        label=[0,0,1]
                        counting[1]+=1
                    else:
                        if(ref_label[loc_f[0]]=='PC'): 
                            label=[1,0,0]
                            counting[0]+=1
                        elif(ref_label[loc_f[0]]=='AFP'): 
                            label=[0,1,0]
                            counting[1]+=1
                        else: 
                            label=[0,1,0]
                            counting[1]+=1
                except ValueError as ve:
                    print("miss ind:",el[4:13])
                    label=[0,0,1]
                
                new_flux=hdu[tce+1].data['LC_DETREND']
                new_flux=new_flux[ind:ind+bins]

                for b in trackrog:
                    if(np.isnan(new_flux[b])): 
                        new_flux[b]=0
                for m in range(0,len(mask)):
                    if(np.isnan(new_flux[m]) and (np.asarray(mask[m])==np.asarray([1,1,1])).all()):
                        mask[m]=label


            red_phase=hdu[len(hdu)-2].data['PHASE'][ind:ind+bins]
            ph_ind=[i for i in range(1,bins) if (red_phase[i]*red_phase[i-1]<0 and np.abs(red_phase[i]*red_phase[i-1])<0.001)]
            if(len(ph_ind)>0): 
                try:
                    
                    loc=np.where(np.asarray(ref_kepid)==el[4:13])
                    loc_f=[m for m in loc[0] if str(av_pl[m])==str(len(hdu)-2)]
                    if(len(loc_f)==0):
                        #print("not in catalog:",len(hdu)-2)
                        if(len(hdu)-2 == 1): break
                        label=[0,0,1]
                        counting[1]+=1
                    else:
                        if(ref_label[loc_f[0]]=='PC'): 
                            counting[0]+=1
                            label=[1,0,0]
                        elif(ref_label[loc_f[0]]=='AFP'): 
                            counting[1]+=1
                            label=[0,1,0]
                        else: 
                            counting[1]+=1
                            label=[0,1,0]
                except ValueError as ve:
                    #print("miss ind:",el[4:13])
                    label=[0,0,1]
                for m in range(0,len(mask)):
                    if((np.asarray(mask[m])==np.asarray([1,1,1])).all()):
                        mask[m]=label
            else: 
                #print('no rel phase')
                for m in range(0,len(mask)):
                    if((np.asarray(mask[m])==np.asarray([1,1,1])).all()):
                        mask[m]=[0,0,1]

        
            lightcurve.append(red_flux)
            totmask.append(mask.reshape(-1))
            counts.append(counting)
            

        if(len(lightcurve)==0):
            print('miss',el[4:13])
            continue      
        np.savetxt(pathout+'xlabel/'+el[4:13],lightcurve,delimiter=' ')
        np.savetxt(pathout+'ylabel/'+el[4:13],totmask,delimiter=' ')
        np.savetxt(pathout+'counts/'+el[4:13],counts,delimiter=' ')
        print(tick,'hit:',el[4:13],np.asarray(lightcurve).shape,np.asarray(totmask).shape,
            np.asarray(counts).shape,len(hdu)-2)


def compr_sem_seg_2(pathin, pathout, bins):
    entries=os.listdir(pathin)
    av_entry=ascii.read(CATALOG+'autovetter_label.tab')
    av_pl=np.array(av_entry['tce_plnt_num'])
    ref_kepid=[('0000'+str(el)[:9])[-9:] for el in av_entry['kepid']]
    #ref_label=av_entry['av_training_set']
    ref_label=av_entry['av_training_set']
    seconds=av_entry['av_pred_class']

    #newids= getids()

    tick=9000
    for el in entries[9000:]:
        tick+=1 
        #if(tick==20): break 
        hdu = fits.open(pathin+el)
        
        try: flux = hdu[1].data['LC_DETREND']
        except: continue
        try: residue = hdu[len(hdu)-1].data['RESIDUAL_LC']
        except: continue

        loc=np.where(np.asarray(ref_kepid)==el[4:13])
        if(not np.any(ref_label[loc[0]]=='PC')): 
            print('no pl:',np.asarray(ref_label[loc[0]]))
            continue

        #get a preliminary phase... must center at least one transit
        ind_arr=np.arange(0,len(flux)-bins,bins)

        lightcurve=[]
        totmask=[]
        counts=[]
        for ind in ind_arr:
            

            counting=[0,0]
            red_flux=flux[ind:ind+bins]
            red_res=residue[ind:ind+bins]

            if(len(red_flux)==0): continue

            #get a clean chunk
            #count_nan=np.isnan(red_flux).sum() 
            #if(count_nan/bins > 0.1): 
            #    continue

            mask=np.asarray([[1,1,1] if (np.isnan(red_res[x]) and not np.isnan(red_flux[x])) else [0,0,1] for x in range(0,len(red_res))])
            #if(np.asarray([(np.asarray(m)==np.asarray([0,0,1])).all() for m in mask]).all()): 
            #    #print('skip')
            #    continue
            trackrog=np.where(np.isnan(red_flux))[0]
            remove_nan(red_flux,4000)
            
            for tce in range(1,len(hdu)-2):
                red_phase=hdu[tce].data['PHASE'][ind:ind+bins]
                ph_ind=[i for i in range(1,bins) if (red_phase[i]*red_phase[i-1]<0 and np.abs(red_phase[i]*red_phase[i-1])<0.001)]
                #if(len(ph_ind)==0): 
                #    #print('no rel phase')
                #    continue
                
                #set an index
                try:
                    loc=np.where(np.asarray(ref_kepid)==el[4:13])
                    loc_f=[m for m in loc[0] if str(av_pl[m])==str(tce)]
                    if(len(loc_f)==0):
                        #print("not in catalog:",tce)
                        if(tce == 1 ): break
                        label=[0,0,1]
                        counting[1]+=1
                    else:
                        if(ref_label[loc_f[0]]=='PC'): 
                            label=[1,0,0]
                            counting[0]+=1
                        elif(ref_label[loc_f[0]]=='AFP' or ref_label[loc_f[0]]=='NTP'): 
                            label=[0,1,0]
                            counting[1]+=1
                        elif(seconds[loc_f[0]]=='PC'): 
                            label=[1,0,0]
                            counting[0]+=1
                        else: 
                            label=[0,1,0]
                            counting[1]+=1
                except ValueError as ve:
                    #print("miss ind:",el[4:13])
                    label=[0,0,1]
                
                new_flux=hdu[tce+1].data['LC_DETREND']
                new_flux=new_flux[ind:ind+bins]

                for b in trackrog:
                    if(np.isnan(new_flux[b])): 
                        new_flux[b]=0
                for m in range(0,len(mask)):
                    if(np.isnan(new_flux[m]) and (np.asarray(mask[m])==np.asarray([1,1,1])).all()):
                        mask[m]=label


            red_phase=hdu[len(hdu)-2].data['PHASE'][ind:ind+bins]
            ph_ind=[i for i in range(1,bins) if (red_phase[i]*red_phase[i-1]<0 and np.abs(red_phase[i]*red_phase[i-1])<0.001)]
            if(len(ph_ind)>0): 
                try:
                    
                    loc=np.where(np.asarray(ref_kepid)==el[4:13])
                    loc_f=[m for m in loc[0] if str(av_pl[m])==str(len(hdu)-2)]
                    if(len(loc_f)==0):
                        #print("not in catalog:",len(hdu)-2)
                        if(len(hdu)-2 == 1): break
                        label=[0,0,1]
                        #counting[1]+=1
                    else:
                        if(ref_label[loc_f[0]]=='PC'): 
                            counting[0]+=1
                            label=[1,0,0]
                        elif(ref_label[loc_f[0]]=='AFP' or ref_label[loc_f[0]]=='NTP'): 
                            counting[1]+=1
                            label=[0,1,0]
                        elif(seconds[loc_f[0]]=='PC'): 
                            label=[1,0,0]
                            counting[0]+=1
                        else: 
                            counting[1]+=1
                            label=[0,1,0]
                except ValueError as ve:
                    #print("miss ind:",el[4:13])
                    label=[0,0,1]
                for m in range(0,len(mask)):
                    if((np.asarray(mask[m])==np.asarray([1,1,1])).all()):
                        mask[m]=label
            else: 
                #print('no rel phase')
                for m in range(0,len(mask)):
                    if((np.asarray(mask[m])==np.asarray([1,1,1])).all()):
                        mask[m]=[0,0,1]

        
            #if(np.asarray([(np.asarray(m)==np.asarray([0,0,1])).all() for m in mask]).all()): 
            #    #print('skip')
            #    continue
            lightcurve.append(red_flux)
            totmask.append(mask.reshape(-1))
            counts.append(counting)
            

        if(len(lightcurve)==0):
            print('miss',el[4:13])
            continue    
        
        net = np.asarray([[lightcurve[i],totmask[i],counts[i]] for i in range(0,len(counts))], dtype='object')
        gc.write_tfr_record(pathout+el[4:13],net,
            ['input','mask','counts'],['ar','ar','ar'],['float32','bool', 'int8'])
        #np.savetxt(pathout+'xlabel/'+el[4:13],lightcurve,delimiter=' ')
        #np.savetxt(pathout+'ylabel/'+el[4:13],totmask,delimiter=' ')
        #np.savetxt(pathout+'counts/'+el[4:13],counts,delimiter=' ')
        print(tick,'hit:',el[4:13],np.asarray(lightcurve).shape,np.asarray(totmask).shape,
            np.asarray(counts).shape,len(hdu)-2,np.asarray(ref_label[loc[0]]))

def plot_inst_seg(path,r):
    fig,ax = plt.subplots(r,r,figsize=(10,10))
    entry=os.listdir(path)
    np.random.shuffle(entry)
    i=0
    j=0
    for el in entry:
        df=np.loadtxt(path+el,delimiter=' ')
        ax[i][j].plot(df[0])
        ax[i][j].plot(df[1]*min(df[0]))
        i=i+1
        if(i==r):
            j=j+1
            i=0
        if(j==r): break


def plot_seg(path,r):
    fig,ax = plt.subplots(r,r,figsize=(10,10))
    entry=os.listdir(path+'xlabel/')
    np.random.shuffle(entry)
    i=0
    j=0
    for el in entry:
        df=np.loadtxt(path+'xlabel/'+el,delimiter=' ')
        df2=np.loadtxt(path+'ylabel/'+el,delimiter=' ')
        #tp=np.loadtxt(path+el)
        ax[i][j].plot(df2[:,1]*df[np.argmin(df)]*2)
        ax[i][j].plot(df2[:,0]*df[np.argmin(df)]*2)
        ax[i][j].plot(df)
        #ax[i][j].plot(tp, np.zeros(len(tp)),marker=".",ls='None')
        #ax[i][j].set_xlim(2000,3000)
        #ax[i][j].plot(-df[2][:2000]*min(df[0]))
        #ax[i][j].plot(-df[1][:2000]*min(df[0]))
        i=i+1
        if(i==r):
            j=j+1
            i=0
        if(j==r): break

def plot_compr_seg(path,r):
    fig,ax = plt.subplots(r,4,figsize=(10,10))
    entry=os.listdir(path+'xlabel/')
    np.random.shuffle(entry)
    i=0
    j=0
    for el in entry:
        df=np.loadtxt(path+'xlabel/'+el,delimiter=' ')
        if(len(df)<4): continue
        df2=np.loadtxt(path+'ylabel/'+el,delimiter=' ')
        dfc=np.loadtxt(path+'counts/'+el,delimiter=' ')
        df2=df2.reshape((len(df),4000,3))
        for j in range(0,4):
            ax[i][j].plot(df2[j,:,1]*df[j,np.argmin(df[j])]*2,color='black',label=dfc[j])
            ax[i][j].plot(df2[j,:,0]*df[j,np.argmin(df[j])]*2,color='green')
            ax[i][j].plot(df[j])
            ax[i][j].legend()
            ax[i][j].set_xlim(2000,3000)
       
        i=i+1
        if(i==r): break

def plot_compr_seg_2(path,r):
    fig,ax = plt.subplots(r,4,figsize=(10,10))
    entry=os.listdir(path)
    np.random.shuffle(entry)
    i=0
    j=0
    for el in entry:
        df,df2,dfc = gc.read_tfr_record(path+el,
            ['input','mask','counts'],
            ['ar','ar','ar'], 
            [tf.float32, tf.bool, tf.int8])
        if(len(df)<4): continue
        
        df2=np.asarray(df2, dtype='float32').reshape((len(df),4000,3))
        print(np.isnan(df).sum())
        df = np.asarray(df)
        dfc = np.asarray(dfc)
        for j in range(0,4):
            mm=df[j,np.argmin(df[j])]
            ax[i][j].plot(df2[j,:,1]*2*mm,color='black',label=dfc[j])
            ax[i][j].plot(df2[j,:,2]*2*mm,color='green')
            ax[i][j].plot(df[j])
            ax[i][j].legend()
            #ax[i][j].set_xlim(2000,3000)
       
        i=i+1
        if(i==r): break

def inst_training_sample(pathin,bp):
    entry=os.listdir(pathin+'segment_map/')
    np.random.shuffle(entry)
    Xtrain=[]
    Mtrain=[]
    X2train=[]
    Xtest=[]
    X2test=[]
    Mtest=[]
    Ytrain=[]
    Ytest=[]

    av_entry=ascii.read('autovetter_label.tab')
    av_pl=np.array(av_entry['tce_plnt_num'])
    ref_kepid=[('0000'+str(el)[:9])[-9:] for el in av_entry['kepid']]
    ref_label=av_entry['av_training_set']

    tabs=0
    for el in entry:
        #if(tabs==40): break
        df=np.loadtxt(pathin+'segment_map/'+el)
        try:
            loc=np.where(np.array(ref_kepid)==el[0:9])
            all_labs=np.asarray([ref_label[m] for m in loc[0]])
        except ValueError as ve:
            print("miss ind:",el[0:9])
            continue
        if(len(all_labs)==0): continue
        if(np.all(all_labs=='UNK')): continue

        try: nextdf=np.loadtxt(pathin+'classifier/'+el)
        except:
            print('no go furthur')
            continue
        if(len(np.asarray(nextdf).shape)<2): continue
        if(len(nextdf)<2): continue
        if(len(nextdf)<4): 
            print(np.asarray(nextdf).shape)
            tmp=4-len(nextdf)
            for i in range(0,tmp): nextdf=np.append(nextdf,nextdf[i].reshape(1,4800),axis=0)
        else: nextdf = nextdf[:4]

        label=[0,0]
        if(np.any(all_labs=='PC')): label[0]=1
        if(np.any(all_labs=='AFP') or np.any(all_labs=='NTP')): label[1]=1
        
        tabs+=1
        print(tabs,el[0:9], nextdf.shape, label)
        if(tabs<bp):
            Xtrain.append(df[0])
            Mtrain.append(df[1])
            X2train.append(nextdf)
            Ytrain.append(label)
        else:
            Xtest.append(df[0])
            Mtest.append(df[1])
            X2test.append(nextdf)
            Ytest.append(label)

    X2train = np.transpose(np.asarray(X2train).reshape(len(X2train),4*4800))
    X2test = np.transpose(np.asarray(X2test).reshape(len(X2test),4*4800))

    print(np.asarray(Xtrain).shape,np.asarray(X2train).shape,np.asarray(Mtrain).shape,np.asarray(Ytrain).shape)
    print(np.asarray(Xtest).shape,np.asarray(X2test).shape,np.asarray(Mtest).shape,np.asarray(Ytest).shape)
    np.savetxt('training_data/Xtrain_inst_mask.csv',Xtrain,delimiter=',')
    np.savetxt('training_data/Mtrain_inst_mask.csv',Mtrain,delimiter=',')
    np.savetxt('training_data/X2train_inst_mask.csv',X2train,delimiter=',')
    np.savetxt('training_data/Ytrain_inst_mask.csv',Ytrain,delimiter=',')
    np.savetxt('training_data/Xtest_inst_mask.csv',Xtest,delimiter=',')
    np.savetxt('training_data/Mtest_inst_mask.csv',Mtest,delimiter=',')
    np.savetxt('training_data/X2test_inst_mask.csv',X2test,delimiter=',')
    np.savetxt('training_data/Ytest_inst_mask.csv',Ytest,delimiter=',')


def sem_training_sample(pathin):
    X_train=[]
    M_train=[]
    Y_train=[]
    pl_entry=os.listdir(pathin)
    av_entry=ascii.read('autovetter_label.tab')
    av_pl=np.array(av_entry['tce_plnt_num'])
    ref_kepid=[('0000'+str(el)[:9])[-9:] for el in av_entry['kepid']]
    ref_label=av_entry['av_training_set']
    
    for el in pl_entry:
        try:    df=np.loadtxt(pathin+el)
        except: 
            print("miss load:",el[0:9])
            continue

        try:
            loc=np.where(np.array(ref_kepid)==el[0:9])
            loc_f=[m for m in loc[0] if str(av_pl[m])==el[10]]
        except ValueError as ve:
            print("miss ind:",el[0:9])
            continue
        if(len(loc_f)==0): continue
        if(ref_label[loc_f[0]]=='UNK'): continue

        if(ref_label[loc_f[0]]=='PC'): Y_train.append([1,0])
        else: Y_train.append([0,1])
        #else: continue
        X_train.append(df[0])
        M_train.append(df[1])
    
    print(np.array(X_train).shape, np.array(Y_train).shape, np.array(M_train).shape)
    np.savetxt('training_data/Xtrain_seg_mask.csv',X_train,delimiter=',')
    np.savetxt('training_data/Mtrain_seg_mask.csv',M_train,delimiter=',')
    np.savetxt('training_data/Ytrain_seg_mask.csv',Y_train,delimiter=',')

def sem_training_sample_tot(pathin):
    X_train=[]
    Y_train=[]
    pl_entry=os.listdir(pathin+'xlabel/')
    m=0
    for el in pl_entry:
        m+=1
        #if(m==10): break
        df=np.loadtxt(pathin+'xlabel/'+el)
        df2=np.loadtxt(pathin+'ylabel/'+el)
        
        check_noise=[(df2[i]==np.array([0,0,1])).all() for i in range(0,len(df2))]
        #print(np.array(check_noise).sum(),np.isnan(df).sum())
        if(np.array(check_noise).all()): 
            print('noise',el[0:9])
            continue
        
        low = min(df)
        ch = np.median(df)-2*np.std(df)
        if(low>ch): 
            print('too insignificant')
            continue

        X_train.append(df)
        Y_train.append(df2)
        print('hit',el[0:9])
    
    Y_train=np.asarray(Y_train)
    X_train=np.asarray(X_train)
    plind=np.asarray([i for i in range(0,len(Y_train)) if (Y_train[i,:,0]==1).any()])
    fpsind=np.setdiff1d(np.arange(0,len(Y_train)), plind)
    print(len(plind),len(fpsind))
    np.random.shuffle(plind)
    np.random.shuffle(fpsind)
    nXtrain=[]
    nYtrain=[]
    for i in range(0,min(len(plind),len(fpsind))):
        nXtrain.append(X_train[plind[i]])
        nYtrain.append(Y_train[plind[i]])
        nXtrain.append(X_train[fpsind[i]])
        nYtrain.append(Y_train[fpsind[i]])

    nYtrain=np.transpose(np.array(nYtrain).reshape(len(nYtrain),12000))
    print(np.array(nXtrain).shape, np.array(nYtrain).shape)
    np.savetxt('../../training_data/Xtrain_seg_mask_av.csv',nXtrain,delimiter=',')
    np.savetxt('../../training_data/Ytrain_seg_mask_av.csv',nYtrain,delimiter=',')


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def create_feature(inp,op,ct,el):
    desc = {
        'input':_bytes_feature(serialize_array(inp)),
        'map': _bytes_feature(serialize_array(op)),
        'counts': _bytes_feature(serialize_array(ct)), 
        'id': _bytes_feature(serialize_array(el)),
    }

    out = tf.train.Example(features=tf.train.Features(feature=desc))

    return(out)

def get_tfr_records(inparr,oparr,ctarr, elarr, filepath):
    writer = tf.io.TFRecordWriter(filepath) #create a writer that'll store our data to disk
    count = 0

    for index in range(len(inparr)):
        #get the data we want to write
        current_image = inparr[index] 
        current_label = oparr[index]
        current_count = ctarr[index]
        current_el = elarr[index]

        out = create_feature(inp=current_image, op=current_label, ct=current_count, el=current_el)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count

def compr_sem_ts(pathin,maxex, outpath):
    X_train=[]
    Y_train=[]
    C_train=[]
    el_track=[]
    pl_entry=os.listdir(pathin)
    np.random.shuffle(pl_entry)
    m=0

    av_entry=ascii.read(CATALOG+'autovetter_label.tab')
    av_pl=np.array(av_entry['tce_plnt_num'])
    ref_kepid=[('0000'+str(el)[:9])[-9:] for el in av_entry['kepid']]
    ref_label=np.asarray(av_entry['av_training_set'])
    tce_dur=np.asarray(av_entry['tce_period'])
    #av_entry=ascii.read(CATALOG+'robovetter_label.dat')
    #av_pl=np.array(av_entry['tce_plnt_'])
    #ref_kepid=[('0000'+str(el)[:9])[-9:] for el in av_entry['kepid']]
    #ref_label=np.asarray(av_entry['label'])

    for el in pl_entry:
        
        try:
            loc=np.where(np.array(ref_kepid)==el[0:9])[0]

        except ValueError as ve:
            print("miss ind:",el[0:9])
            continue
        if(len(loc)==0): continue
        if(np.all(ref_label[loc]=='UNK')): continue
        if(np.all(tce_dur[loc]>81)): continue

        m+=1
        #if(m==60): break
        df,dfy,dfc = gc.read_tfr_record(pathin+el,
            ['input','mask','counts'],
            ['ar','ar','ar'], 
            [tf.float32, tf.bool, tf.int8])
        
        df = np.asarray(df)
        dfc = np.asarray(dfc)
        dfy = np.asarray(dfy)

        els,cts = np.unique(dfc,axis=0,return_counts=True)
        bestarr = els[np.argmax(cts)]
        best_inds = [i for i in range(0,len(dfc)) if(np.all(np.asarray(dfc[i])==np.asarray(bestarr)))]

        try: temp=dfy.reshape(len(dfc),4000,3)
        except: continue
        ex = 0
        mintab = int(min(1,len(best_inds)))
        for inds in best_inds[mintab:]:
            check_noise=[(temp[inds,i]==np.asarray([0,0,1])).all() for i in range(0,len(temp[inds]))]
            if(np.asarray(check_noise).all()): 
                print('noise',el[0:9])
                continue
            check_overload = [(temp[inds,i]==np.asarray([0,1,0])).all() for i in range(0,len(temp[inds]))]
            if(np.asarray(check_overload).sum()/4000 >0.8): 
                print('too much fps:',el[0:9])
                #continue

            #check_overload = [(temp[inds,i]==np.asarray([1,0,0])).all() for i in range(0,len(temp[inds]))]
            #if(np.asarray(check_overload).sum()/4000 >0.8): 
            #    print('too much pl:',el[0:9])
            #    continue
            prop = temp[inds,:,0].sum()
            prop2 = temp[inds,:,1].sum()
            #if(prop>prop2 and prop<1000): 
            #    print('double:')
            #   X_train.append(df[best_inds[-1]])
            #    Y_train.append(dfy[best_inds[-1]])
            #    C_train.append(dfc[best_inds[-1]])
            #    el_track.append(el[0:9])
         
            X_train.append(df[inds])
            Y_train.append(dfy[inds])
            C_train.append(dfc[inds])
            el_track.append(el[0:9])
            ex+=1
            if(ex==maxex): break
        print('hit',el[0:9],len(best_inds), tce_dur[loc])
    
    Y_train=np.asarray(Y_train, np.bool)
    X_train=np.asarray(X_train, np.float32)
    C_train=np.asarray(C_train, np.int8)

    medinds_p=np.asarray([i for i in range(0,len(C_train)) if (C_train[i,0]>0)])
    medinds_fps=np.setdiff1d(np.arange(0,len(C_train)), medinds_p)
    print(len(medinds_p),len(medinds_fps))
    
    pind = len(medinds_p)
    fpsind = len(medinds_fps)

    #fpsind = min(pind,fpsind)
    #pind = fpsind
    
    nXtrain=[]
    nYtrain=[]
    nCtrain=[]
    eltrain=[]
    nXtest=[]
    nYtest=[]
    nCtest=[]
    eltest=[]

    [nXtrain.append(X_train[medinds_p[i]]) for i in range(0,int(pind*0.8),1)]
    [nXtest.append(X_train[medinds_p[i]]) for i in range(int(pind*0.8),pind,1)]
    [nXtrain.append(X_train[medinds_fps[i]]) for i in range(0,int(fpsind*0.8),1)]
    [nXtest.append(X_train[medinds_fps[i]]) for i in range(int(fpsind*0.8),fpsind,1)]
    [nYtrain.append(Y_train[medinds_p[i]]) for i in range(0,int(pind*0.8),1)]
    [nYtest.append(Y_train[medinds_p[i]]) for i in range(int(pind*0.8),pind,1)]
    [nYtrain.append(Y_train[medinds_fps[i]]) for i in range(0,int(fpsind*0.8),1)]
    [nYtest.append(Y_train[medinds_fps[i]]) for i in range(int(fpsind*0.8),fpsind,1)]
    [nCtrain.append(C_train[medinds_p[i]]) for i in range(0,int(pind*0.8),1)]
    [nCtest.append(C_train[medinds_p[i]]) for i in range(int(pind*0.8),pind,1)]
    [nCtrain.append(C_train[medinds_fps[i]]) for i in range(0,int(fpsind*0.8),1)]
    [nCtest.append(C_train[medinds_fps[i]]) for i in range(int(fpsind*0.8),fpsind,1)]
    [eltrain.append(el_track[medinds_p[i]]) for i in range(0,int(pind*0.8),1)]
    [eltest.append(el_track[medinds_p[i]]) for i in range(int(pind*0.8),pind,1)]
    [eltrain.append(el_track[medinds_fps[i]]) for i in range(0,int(fpsind*0.8),1)]
    [eltest.append(el_track[medinds_fps[i]]) for i in range(int(fpsind*0.8),fpsind,1)]

    #randomise
    rarr=np.arange(0,len(nCtrain))
    rarrt=np.arange(0,len(nCtest))
    np.random.shuffle(rarr)
    np.random.shuffle(rarrt)
    nXtrain = [nXtrain[i] for i in rarr]
    nYtrain = [nYtrain[i] for i in rarr]
    nCtrain = [nCtrain[i] for i in rarr]
    eltrain = [eltrain[i] for i in rarr]
    nXtest = [nXtest[i] for i in rarrt]
    nYtest = [nYtest[i] for i in rarrt]
    nCtest = [nCtest[i] for i in rarrt]
    eltest = [eltest[i] for i in rarrt]
    '''
    for i in range(0,min(len(medinds_p),len(medinds_fps))):
        nXtrain.append(X_train[medinds_fps[i]])
        nYtrain.append(Y_train[medinds_fps[i]])
        nCtrain.append(C_train[medinds_fps[i]])
        nXtrain.append(X_train[medinds_p[i]])
        nYtrain.append(Y_train[medinds_p[i]])
        nCtrain.append(C_train[medinds_p[i]])'''

    net = np.asarray([[nXtrain[i],nYtrain[i],nCtrain[i],eltrain[i]] for i in range(len(eltrain))], dtype='object')
    nett = np.asarray([[nXtest[i],nYtest[i],nCtest[i],eltest[i]] for i in range(len(eltest))], dtype='object')
    gc.write_tfr_record(outpath+'train',net,['input','map','counts','id'],
        ['ar','ar','ar','b'],['float32','bool','int8','byte'])
    gc.write_tfr_record(outpath+'test',nett,['input','map','counts','id'],
        ['ar','ar','ar','b'],['float32','bool','int8','byte'])
    
    print(np.asarray(nXtrain).shape, np.asarray(nYtrain).shape, np.asarray(nCtrain).shape)
    print(np.asarray(nXtest).shape, np.asarray(nYtest).shape, np.asarray(nCtest).shape)

def compr_sem_ts_2(pathin,maxex):
    X_train=[]
    Y_train=[]
    C_train=[]
    el_track=[]
    pl_entry=os.listdir(pathin)
    np.random.shuffle(pl_entry)

    av_entry=ascii.read(CATALOG+'robovetter_label.dat')
    av_pl=np.array(av_entry['tce_plnt_num'])
    ref_kepid=[('0000'+str(el)[:9])[-9:] for el in av_entry['kepid']]
    ref_label=np.asarray(av_entry['label'])

    m=0
    for el in pl_entry:
        m+=1
        #if(m==60): break

        try:
            loc=np.where(np.array(ref_kepid)==el[0:9])[0]

        except ValueError as ve:
            print("miss ind:",el[0:9])
            continue
        if(len(loc)==0): continue
        if(np.any(ref_label[loc]=='CANDIDATE')): continue

        df,dfy,dfc = gc.read_tfr_record(pathin+el,
            ['input','mask','counts'],
            ['ar','ar','ar'], 
            [tf.float32, tf.bool, tf.int8])

        els,cts = np.unique(dfc,axis=0,return_counts=True)
        bestarr = els[np.argmax(cts)]
        best_inds = [i for i in range(0,len(dfc)) if(np.all(np.asarray(dfc[i])==np.asarray(bestarr)))]

        temp=np.asarray(dfy).reshape(len(dfc),4000,3)
    
        ex = 0
        for inds in best_inds:
            check_noise=[(temp[inds,i]==np.asarray([0,0,1])).all() for i in range(0,len(temp[inds]))]
            if(np.asarray(check_noise).all()): 
                print('noise',el[0:9])
                continue
            #check_overload = [(temp[inds,i]==np.asarray([0,1,0])).all() for i in range(0,len(temp[inds]))]
            #if(np.asarray(check_overload).sum()/4000 >0.8): 
            #    print('too much fps:',el[0:9])
            #    continue

            #check_overload = [(temp[inds,i]==np.asarray([1,0,0])).all() for i in range(0,len(temp[inds]))]
            #if(np.asarray(check_overload).sum()/4000 >0.8): 
            #    print('too much pl:',el[0:9])
            #    continue

            #if(dfc[inds,1]<0.5): 
            #    print('double:')
            #    X_train.append(df[best_inds[-1]])
            #    Y_train.append(dfy[best_inds[-1]])
            #    C_train.append(dfc[best_inds[-1]])
            #    el_track.append(el[0:9])
         
            X_train.append(df[inds])
            Y_train.append(dfy[inds])
            C_train.append(dfc[inds])
            el_track.append(el[0:9])
            ex+=1
            if(ex==maxex): break
        print('hit',el[0:9],len(best_inds))
    
    Y_train=np.asarray(Y_train, np.bool)
    X_train=np.asarray(X_train, np.float32)
    C_train=np.asarray(C_train, np.int8)

    medinds_p=np.asarray([i for i in range(0,len(C_train)) if (C_train[i,0]>0)])
    medinds_fps=np.setdiff1d(np.arange(0,len(C_train)), medinds_p)
    print(len(medinds_p),len(medinds_fps))
    
    pind = len(medinds_p)
    fpsind = len(medinds_fps)

    #fpsind = min(pind,fpsind)
    #pind = fpsind
    
    nXtrain=[]
    nYtrain=[]
    nCtrain=[]
    eltrain=[]
    nXtest=[]
    nYtest=[]
    nCtest=[]
    eltest=[]

    [nXtrain.append(X_train[medinds_p[i]]) for i in range(0,int(pind*0.8),1)]
    [nXtest.append(X_train[medinds_p[i]]) for i in range(int(pind*0.8),pind,1)]
    [nXtrain.append(X_train[medinds_fps[i]]) for i in range(0,int(fpsind*0.8),1)]
    [nXtest.append(X_train[medinds_fps[i]]) for i in range(int(fpsind*0.8),fpsind,1)]
    [nYtrain.append(Y_train[medinds_p[i]]) for i in range(0,int(pind*0.8),1)]
    [nYtest.append(Y_train[medinds_p[i]]) for i in range(int(pind*0.8),pind,1)]
    [nYtrain.append(Y_train[medinds_fps[i]]) for i in range(0,int(fpsind*0.8),1)]
    [nYtest.append(Y_train[medinds_fps[i]]) for i in range(int(fpsind*0.8),fpsind,1)]
    [nCtrain.append(C_train[medinds_p[i]]) for i in range(0,int(pind*0.8),1)]
    [nCtest.append(C_train[medinds_p[i]]) for i in range(int(pind*0.8),pind,1)]
    [nCtrain.append(C_train[medinds_fps[i]]) for i in range(0,int(fpsind*0.8),1)]
    [nCtest.append(C_train[medinds_fps[i]]) for i in range(int(fpsind*0.8),fpsind,1)]
    [eltrain.append(el_track[medinds_p[i]]) for i in range(0,int(pind*0.8),1)]
    [eltest.append(el_track[medinds_p[i]]) for i in range(int(pind*0.8),pind,1)]
    [eltrain.append(el_track[medinds_fps[i]]) for i in range(0,int(fpsind*0.8),1)]
    [eltest.append(el_track[medinds_fps[i]]) for i in range(int(fpsind*0.8),fpsind,1)]

    #randomise
    rarr=np.arange(0,len(nCtrain))
    rarrt=np.arange(0,len(nCtest))
    np.random.shuffle(rarr)
    np.random.shuffle(rarrt)
    nXtrain = [nXtrain[i] for i in rarr]
    nYtrain = [nYtrain[i] for i in rarr]
    nCtrain = [nCtrain[i] for i in rarr]
    eltrain = [eltrain[i] for i in rarr]
    nXtest = [nXtest[i] for i in rarrt]
    nYtest = [nYtest[i] for i in rarrt]
    nCtest = [nCtest[i] for i in rarrt]
    eltest = [eltest[i] for i in rarrt]
    '''
    for i in range(0,min(len(medinds_p),len(medinds_fps))):
        nXtrain.append(X_train[medinds_fps[i]])
        nYtrain.append(Y_train[medinds_fps[i]])
        nCtrain.append(C_train[medinds_fps[i]])
        nXtrain.append(X_train[medinds_p[i]])
        nYtrain.append(Y_train[medinds_p[i]])
        nCtrain.append(C_train[medinds_p[i]])'''
    
    print(np.asarray(nXtrain).shape, np.asarray(nYtrain).shape, np.asarray(nCtrain).shape)
    print(np.asarray(nXtest).shape, np.asarray(nYtest).shape, np.asarray(nCtest).shape)
    get_tfr_records(nXtrain, nYtrain, nCtrain, eltrain, '../../training_data/seg_mask_training_rv_filt')
    get_tfr_records(nXtest, nYtest, nCtest, eltest, '../../training_data/seg_mask_test_rv_filt')

def _parse_tfr_element(element):
  desc = {
        'input':tf.io.FixedLenFeature([], tf.string),
        'map':tf.io.FixedLenFeature([], tf.string),
        'counts': tf.io.FixedLenFeature([], tf.string),
        'id': tf.io.FixedLenFeature([], tf.string),  
    }
  example_message = tf.io.parse_single_example(element, desc)

  #return(example_message['counts'])
  binp = example_message['input'] # get byte 
  bmap = example_message['map'] # get byte string
  bcts = example_message['counts'] # get byte string
  bid = example_message['id'] # get byte string
  print(binp.shape,bmap.shape,bcts.shape)
  inp = tf.io.parse_tensor(binp, out_type=tf.float32) # restore 2D array from byte string
  map = tf.io.parse_tensor(bmap, out_type=tf.bool)
  cts = tf.io.parse_tensor(bcts, out_type=tf.int8)
  id = tf.io.parse_tensor(bid, out_type=tf.string)
  return (inp,map,cts,id)

def expand_ts(pathin, proc_path): 
    
    tfr_testdata = tf.data.TFRecordDataset([pathin]) 
    testdata = tfr_testdata.map(_parse_tfr_element)

    ID = [instance[3] for instance in testdata]
    ID = [ID[i].numpy() for i in range(0,len(ID))]
    ID = [str(ID[i])[2:11] for i in range(0,len(ID))]
    print(ID[2:10])
    #ID = os.listdir('../../processed_directories/expand_test_and_noise/counts')

    for el in ID:
        writer = tf.io.TFRecordWriter('../../processed_directories/expand_test_av/'+el) #create a writer that'll store our data to disk
        df=np.loadtxt(proc_path+'xlabel/'+el)
        dfy=np.loadtxt(proc_path+'ylabel/'+el)
        dfc=np.loadtxt(proc_path+'counts/'+el)
        count = 0
        for i in range(0,len(df)):
            desc = {
                'input':_bytes_feature(serialize_array(df[i])),
                'map': _bytes_feature(serialize_array(dfy[i])),
            }
            out = tf.train.Example(features=tf.train.Features(feature=desc))
            writer.write(out.SerializeToString())
            count += 1

        writer.close()
        print(f"Wrote {count} elements to {el}")


def expand_ts_2(pathin, proc_path): 
    
    tfr_testdata = tf.data.TFRecordDataset([pathin]) 
    testdata = tfr_testdata.map(_parse_tfr_element)

    ID = [instance[3] for instance in testdata]
    ID = [ID[i].numpy() for i in range(0,len(ID))]
    ID = [str(ID[i])[2:11] for i in range(0,len(ID))]
    print(ID[2:10])
    #ID = os.listdir('../../processed_directories/expand_test_and_noise/counts')

    for el in ID:
        df,dfy,dfc = gc.read_tfr_record(proc_path+el,
            ['input','mask','counts'],['ar','ar','ar'], 
            [tf.float32, tf.bool, tf.int8])
        
        net = np.asarray([[df[i], dfy[i]] for i in range(0,len(df))], dtype='object')
        gc.write_tfr_record('../../processed_directories/expand_test_av/'+el,net,
        ['input','map'],['ar','ar'],['float32','float32'])
        

def cumulative_ts(phfold_path, raw_path, outpath):
    #phf_entry = os.listdir(phfold_path)
    print(raw_path)
    raw_entry = os.listdir(raw_path)
    catalog = ascii.read(CATALOG+'autovetter_label.tab')
    av_no=np.asarray(catalog['tce_plnt_num'])
    catkepid=[('0000'+str(el)[:9])[-9:] for el in catalog['kepid']]
    catlabel=np.asarray(catalog['av_training_set'])

    X_train=[]
    M_train=[]
    PL_train=[]
    PG_train=[]
    C_train=[]
    L_train=[]
    el_train=[]

    note=0
    for el in raw_entry:
        tempx=[]
        tempm=[]
        tempc=[]

        try: loc=np.where(np.asarray(catkepid)==el[0:9])[0]
            #loc_f=[m for m in loc if str(av_no[m])==el[-3]]
        except ValueError as ve:
            print("miss ind:",el[0:9])
            continue
        #check if the raw lc is atleast in catalog
        if(len(loc)==0): continue

        nums = np.asarray(av_no[loc])

        dfr_i,dfr_m,dfr_c = gc.read_tfr_record(raw_path+el,['input','mask','counts'],
            ['ar','ar','ar'], [tf.float32, tf.bool, tf.int8])

        dfr_i = np.asarray(dfr_i)
        dfr_m = np.asarray(dfr_m)
        dfr_c = np.asarray(dfr_c)

        els,cts = np.unique(dfr_c,axis=0,return_counts=True)
        bestarr = els[np.argmax(cts)]
        best_inds = [i for i in range(0,len(dfr_c)) if(np.all(dfr_c[i]==np.asarray(bestarr)))]

        try: temp=dfr_m.reshape(len(dfr_c),4000,3)
        except: continue
        mintab = int(min(1,len(best_inds)))
        for inds in best_inds[mintab:]:
            if(len(dfr_i[inds])<4000): 
                print('odd')
                continue
            check_noise=[(temp[inds,i]==np.asarray([0,0,1])).all() for i in range(0,len(temp[inds]))]
            if(np.asarray(check_noise).all()): 
                print('noise',el[0:9])
                continue
            
            tempx.append(dfr_i[inds])
            tempm.append(dfr_m[inds])
            tempc.append(dfr_c[inds])
            break
        if(len(tempx)<1): 
            print('miss')
            continue
        note+=1
        #if(note==50): break
        for x in nums:
            try: dfg = pd.read_csv(phfold_path+'global/'+el+'_'+str(x)+'_g',sep=" ")
            except: continue

            loc_f=[m for m in loc if av_no[m]==x]
            #print(loc,nums,loc_f)
            if(len(loc_f)==0): continue
            if(catlabel[loc_f[0]]=='UNK'): continue
            elif(catlabel[loc_f[0]]=='PC'): L_train.append([1,0])
            else: L_train.append([0,1])

            dfl = pd.read_csv(phfold_path+'local/'+el+'_'+str(x)+'_l',sep=" ")


            X_train.append(tempx[0])
            M_train.append(tempm[0])
            C_train.append(tempc[0])
            PL_train.append(np.asarray(dfl['flux']))
            PG_train.append(np.asarray(dfg['flux']))
            el_train.append(el+'_'+str(av_no[loc_f[0]]))
 
            print(el,loc_f[0], len(dfl),len(dfg),tempx[0].shape,tempm[0].shape,tempc[0],catlabel[loc_f[0]])

    M_train=np.asarray(M_train, np.bool)
    X_train=np.asarray(X_train, np.float32)
    C_train=np.asarray(C_train, np.int8)
    PL_train=np.asarray(PL_train, np.float32)
    PG_train=np.asarray(PG_train, np.float32)
    L_train=np.asarray(L_train, np.bool)
    el_train = np.asarray(el_train)

    medinds_p=np.asarray([i for i in range(0,len(L_train)) if (L_train[i,0])])
    medinds_fps=np.setdiff1d(np.arange(0,len(C_train)), medinds_p)
    print(len(medinds_p),len(medinds_fps))
    
    pind = len(medinds_p)
    fpsind = len(medinds_fps)

    np.random.shuffle(medinds_fps)
    np.random.shuffle(medinds_p)

    nXtrain=[]
    nMtrain=[]
    nCtrain=[]
    nPLtrain=[]
    nPGtrain=[]
    nLtrain=[]
    neltrain=[]
    nXtest=[]
    nMtest=[]
    nCtest=[]
    nPLtest=[]
    nPGtest=[]
    nLtest=[]
    neltest=[]

    [nXtrain.append(X_train[medinds_p[i]]) for i in range(0,int(pind*0.8),1)]
    [nXtest.append(X_train[medinds_p[i]]) for i in range(int(pind*0.8),pind,1)]
    [nXtrain.append(X_train[medinds_fps[i]]) for i in range(0,int(fpsind*0.8),1)]
    [nXtest.append(X_train[medinds_fps[i]]) for i in range(int(fpsind*0.8),fpsind,1)]
    [nMtrain.append(M_train[medinds_p[i]]) for i in range(0,int(pind*0.8),1)]
    [nMtest.append(M_train[medinds_p[i]]) for i in range(int(pind*0.8),pind,1)]
    [nMtrain.append(M_train[medinds_fps[i]]) for i in range(0,int(fpsind*0.8),1)]
    [nMtest.append(M_train[medinds_fps[i]]) for i in range(int(fpsind*0.8),fpsind,1)]
    [nCtrain.append(C_train[medinds_p[i]]) for i in range(0,int(pind*0.8),1)]
    [nCtest.append(C_train[medinds_p[i]]) for i in range(int(pind*0.8),pind,1)]
    [nCtrain.append(C_train[medinds_fps[i]]) for i in range(0,int(fpsind*0.8),1)]
    [nCtest.append(C_train[medinds_fps[i]]) for i in range(int(fpsind*0.8),fpsind,1)]
    [nPLtrain.append(PL_train[medinds_p[i]]) for i in range(0,int(pind*0.8),1)]
    [nPLtest.append(PL_train[medinds_p[i]]) for i in range(int(pind*0.8),pind,1)]
    [nPLtrain.append(PL_train[medinds_fps[i]]) for i in range(0,int(fpsind*0.8),1)]
    [nPLtest.append(PL_train[medinds_fps[i]]) for i in range(int(fpsind*0.8),fpsind,1)]
    [nPGtrain.append(PG_train[medinds_p[i]]) for i in range(0,int(pind*0.8),1)]
    [nPGtest.append(PG_train[medinds_p[i]]) for i in range(int(pind*0.8),pind,1)]
    [nPGtrain.append(PG_train[medinds_fps[i]]) for i in range(0,int(fpsind*0.8),1)]
    [nPGtest.append(PG_train[medinds_fps[i]]) for i in range(int(fpsind*0.8),fpsind,1)]
    [nLtrain.append(L_train[medinds_p[i]]) for i in range(0,int(pind*0.8),1)]
    [nLtest.append(L_train[medinds_p[i]]) for i in range(int(pind*0.8),pind,1)]
    [nLtrain.append(L_train[medinds_fps[i]]) for i in range(0,int(fpsind*0.8),1)]
    [nLtest.append(L_train[medinds_fps[i]]) for i in range(int(fpsind*0.8),fpsind,1)]
    [neltrain.append(el_train[medinds_p[i]]) for i in range(0,int(pind*0.8),1)]
    [neltest.append(el_train[medinds_p[i]]) for i in range(int(pind*0.8),pind,1)]
    [neltrain.append(el_train[medinds_fps[i]]) for i in range(0,int(fpsind*0.8),1)]
    [neltest.append(el_train[medinds_fps[i]]) for i in range(int(fpsind*0.8),fpsind,1)]

    #randomise
    rarr=np.arange(0,len(nCtrain))
    rarrt=np.arange(0,len(nCtest))
    np.random.shuffle(rarr)
    np.random.shuffle(rarrt)
    nXtrain = [nXtrain[i] for i in rarr]
    nMtrain = [nMtrain[i] for i in rarr]
    nCtrain = [nCtrain[i] for i in rarr]
    nPLtrain = [nPLtrain[i] for i in rarr]
    nPGtrain = [nPGtrain[i] for i in rarr]
    nLtrain = [nLtrain[i] for i in rarr]
    neltrain = [neltrain[i] for i in rarr]
    nXtest = [nXtest[i] for i in rarrt]
    nMtest = [nMtest[i] for i in rarrt]
    nCtest = [nCtest[i] for i in rarrt]
    nPLtest = [nPLtest[i] for i in rarrt]
    nPGtest = [nPGtest[i] for i in rarrt]
    nLtest = [nLtest[i] for i in rarrt]
    neltest = [neltest[i] for i in rarrt]
    
    print(np.asarray(nXtrain).shape, np.asarray(nMtrain).shape)
    print(np.asarray(nXtest).shape, np.asarray(nMtest).shape)
    
    net = np.asarray([[nXtrain[i],nMtrain[i],nPLtrain[i],nPGtrain[i],nCtrain[i],
        nLtrain[i],neltrain[i]] for i in range(len(nLtrain))], dtype='object')
    nett = np.asarray([[nXtest[i],nMtest[i],nPLtest[i],nPGtest[i],nCtest[i],
        nLtest[i],neltest[i]] for i in range(len(nLtest))], dtype='object')
    gc.write_tfr_record(outpath+'train',net,['input','map','local','global','counts','label','id'],
        ['ar','ar','ar','ar','ar','ar','b'],['float32','bool','float32','float32','int8','bool','byte'])
    gc.write_tfr_record(outpath+'test',nett,['input','map','local','global','counts','label','id'],
        ['ar','ar','ar','ar','ar','ar','b'],['float32','bool','float32','float32','int8','bool','byte'])
    
    
def cross_val_ts(pathin,maxex, outpath):
    X_train=[]
    Y_train=[]
    C_train=[]
    el_track=[]
    pl_entry=os.listdir(pathin)
    np.random.shuffle(pl_entry)
    m=0

    for el in pl_entry:
        
        m+=1
        #if(m==60): break
        df,dfy,dfc = gc.read_tfr_record(pathin+el,
            ['input','mask','counts'],
            ['ar','ar','ar'], 
            [tf.float32, tf.bool, tf.int8])
        
        df = np.asarray(df)
        dfc = np.asarray(dfc)
        dfy = np.asarray(dfy)

        els,cts = np.unique(dfc,axis=0,return_counts=True)
        bestarr = els[np.argmax(cts)]
        best_inds = [i for i in range(0,len(dfc)) if(np.all(np.asarray(dfc[i])==np.asarray(bestarr)))]

        try: temp=dfy.reshape(len(dfc),4000,3)
        except: continue
        ex = 0
        mintab = int(min(1,len(best_inds)))
        for inds in best_inds[mintab:]:
            check_noise=[(temp[inds,i]==np.asarray([0,0,1])).all() for i in range(0,len(temp[inds]))]
            if(np.asarray(check_noise).all()): 
                print('noise',el[0:9])
                continue
            check_overload = [(temp[inds,i]==np.asarray([0,1,0])).all() for i in range(0,len(temp[inds]))]
            if(np.asarray(check_overload).sum()/4000 >0.8): 
                print('too much fps:',el[0:9])
                #continue

            prop = temp[inds,:,0].sum()
            prop2 = temp[inds,:,1].sum()
            if(prop>prop2): 
                print('double:')
                X_train.append(df[best_inds[-1]])
                Y_train.append(dfy[best_inds[-1]])
                C_train.append(dfc[best_inds[-1]])
                el_track.append(el[0:9])
         
            X_train.append(df[inds])
            Y_train.append(dfy[inds])
            C_train.append(dfc[inds])
            el_track.append(el[0:9])
            ex+=1
            if(ex==maxex): break
        print('hit',el[0:9],len(best_inds))
    
    Y_train=np.asarray(Y_train, np.bool)
    X_train=np.asarray(X_train, np.float32)
    C_train=np.asarray(C_train, np.int8)

    medinds_p=np.asarray([i for i in range(0,len(C_train)) if (C_train[i,0]>0)])
    medinds_fps=np.setdiff1d(np.arange(0,len(C_train)), medinds_p)
    print(len(medinds_p),len(medinds_fps))
    
    pind = len(medinds_p)
    fpsind = len(medinds_fps)

    for m in range(0,5):
        nXtrain=[]
        nYtrain=[]
        nCtrain=[]
        eltrain=[]

        [nXtrain.append(X_train[medinds_p[i]]) for i in range(int(pind*0.2*m),int(pind*0.2*(m+1)),1)]
        [nXtrain.append(X_train[medinds_fps[i]]) for i in range(int(fpsind*0.2*m),int(fpsind*0.2*(m+1)),1)]
        [nYtrain.append(Y_train[medinds_p[i]]) for i in range(int(pind*0.2*m),int(pind*0.2*(m+1)),1)]
        [nYtrain.append(Y_train[medinds_fps[i]]) for i in range(int(fpsind*0.2*m),int(fpsind*0.2*(m+1)),1)]
        [nCtrain.append(C_train[medinds_p[i]]) for i in range(int(pind*0.2*m),int(pind*0.2*(m+1)),1)]
        [nCtrain.append(C_train[medinds_fps[i]]) for i in range(int(fpsind*0.2*m),int(fpsind*0.2*(m+1)),1)]
        [eltrain.append(el_track[medinds_p[i]]) for i in range(int(pind*0.2*m),int(pind*0.2*(m+1)),1)]
        [eltrain.append(el_track[medinds_fps[i]]) for i in range(int(fpsind*0.2*m),int(fpsind*0.2*(m+1)),1)]

        #randomise
        rarr=np.arange(0,len(nCtrain))
        np.random.shuffle(rarr)
        nXtrain = [nXtrain[i] for i in rarr]
        nYtrain = [nYtrain[i] for i in rarr]
        nCtrain = [nCtrain[i] for i in rarr]
        eltrain = [eltrain[i] for i in rarr]
    

        net = np.asarray([[nXtrain[i],nYtrain[i],nCtrain[i],eltrain[i]] for i in range(len(eltrain))], dtype='object')
        gc.write_tfr_record(outpath+'_s'+str(m),net,['input','map','counts','id'],
            ['ar','ar','ar','b'],['float32','bool','int8','byte'])
        
        print(np.asarray(nXtrain).shape, np.asarray(nYtrain).shape, np.asarray(nCtrain).shape)


def anomalies_ts(inpdir, pathout, maxex,bin, binpr):
    entries = os.listdir(inpdir)
    netinp=[]
    netmap=[]
    netop=[]
    netopmp=[]
    for el in entries:
        df,dfy,dfc = gc.read_tfr_record(inpdir+el,
            ['input','mask','counts'],['ar','ar','ar'], 
            [tf.float32, tf.bool, tf.int8])
        shar=np.arange(0,17,1)
        np.random.shuffle(shar)
        df=np.asarray([df[i] for i in shar])
        dfy=np.asarray([dfy[i] for i in shar])
        
        dfy = np.reshape(dfy,(17,4000,3))
        for i in range(maxex):
            netinp.append(df[i,0:int(bin/2)])
            netop.append(df[i,int(bin/2):int(bin/2+binpr)])
            netmap.append(dfy[i,0:int(2000+bin/2),0])
            netopmp.append(dfy[i,int(bin/2):int(bin/2+binpr),0])
    net = np.asarray([[netinp[i], netmap[i], netop[i], netopmp[i]] for i in range(0,len(netmap))], dtype='object')
    gc.write_tfr_record(pathout,net,
        ['input','map', 'output','opmap'],['ar','ar','ar','ar'],['float32','float32','float32','float32'])


anomalies_ts(TRAINING_MODULE+'full_lc_planets/','../../training_data/anomalies_ts2000_1000',4,2000,500)
#cumulative_ts(TRAINING_MODULE+'new_loc_glob/',TRAINING_MODULE+'sem_seg_av/','../../training_data/total_ts_av_')
#inst_segment(FILEPATH_FPS,'data_seg/',5000)
#training_sample('data_seg/')
#
#inst_seg_classifier(FILEPATH_DATA,'inst_seg/classifier/',4800)
#inst_seg_classifier(FILEPATH_FPS,'inst_seg/classifier/',4800)
#sem_segment_one(FILEPATH_FPS,TRAINING_MODULE+'sem_seg_one/',4000)
#sem_segment_clean(FILEPATH_DATA,TRAINING_MODULE+'sem_seg_clean/',12000)
#compr_sem_seg_2(FILEPATH_FPS,TRAINING_MODULE+'full_lc_planets/',4000)
#sem_segment_tot(FILEPATH_DATA,TRAINING_MODULE+'sem_seg_av/',4000)
#sem_segment_tot(FILEPATH_FPS,TRAINING _MODULE+'sem_seg_av/',4000)
#plot_inst_seg('inst_seg/segment_map/',5)
#sem_segment(FILEPATH_DATA,'sem_seg2/',4800)
#inst_training_sample('inst_seg/',3200)
#compr_sem_ts(TRAINING_MODULE+'sem_seg_av_2/',1,'../../training_data/trial_av_' )
#cross_val_ts(TRAINING_MODULE+'sem_seg_av_2/',1,'../../training_data/cross_validation/zero_aug_' )
#sem_training_sample_tot(TRAINING_MODULE+'sem_seg_av/')
#expand_ts_2('../../training_data/total_ts_av_test', '../../processed_directories/sem_seg_av/')

#plot_compr_seg_2(TRAINING_MODULE+'sem_seg_rv/',4)
#plt.show()
