import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import count
from numpy.core.numeric import count_nonzero
from numpy.lib.function_base import median
import pandas as pd
import os
from scipy.interpolate import interp1d
from astropy.io import fits,ascii

#ohkay... me trying out more and more ideas which i dont think will work. Now if we take this as a segmentation 
#problem at face value... its worth a shot. similar to the recinstruction idea but more fancy ig.
FILEPATH_FPS="E:\Masters_Project_Data\\alienworlds_fps\\"
FILEPATH_DATA="E:\Masters_Project_Data\\alienworlds_data\\"

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
                red_flux[i]=0
    for i in range(0,len(red_flux)):
        if np.isnan(red_flux[i]):
            red_flux[i]=0
        
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
    #av_entry=ascii.read('autovetter_label.tab')
    av_entry=ascii.read('robovetter_label.dat')
    av_pl=np.array(av_entry['tce_plnt_num'])
    ref_kepid=[('0000'+str(el)[:9])[-9:] for el in av_entry['kepid']]
    #ref_label=av_entry['av_training_set']
    ref_label=av_entry['label']
    tick=0
    for el in entries[140:]:
        tick+=1
        if(tick==1600): break
        hdu = fits.open(pathin+el)
        
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
            if(count_nan/bins > 0.2): 
                continue

            mask=np.array([[1,1] if (np.isnan(red_res[x]) and not np.isnan(red_flux[x])) else [0,0] for x in range(0,len(red_res))])
            if(np.array([(np.array(m)==np.array([0,0])).all() for m in mask]).all()): 
                print('skip')
                continue
            trackrog=np.where(np.isnan(red_flux))
            remove_nan(red_flux,bins)
            
            for tce in range(2,len(hdu)-1):
                try:
                    loc=np.where(np.array(ref_kepid)==el[4:13])
                    loc_f=[m for m in loc[0] if str(av_pl[m])==str(tce-1)]
                    if(len(loc_f)==0):
                        print("not in catalog:",tce-1)
                        label=[0,1]
                    else:
                        if(ref_label[loc_f[0]]=='CONFIRMED'): label=[1,0]
                        elif(ref_label[loc_f[0]]=='CANDIDATE'): label=[1,0]
                        else: label=[0,1]
                except ValueError as ve:
                    print("miss ind:",el[4:13])
                    label=[0,0]
                
                new_flux=hdu[tce].data['LC_DETREND']
                new_flux=new_flux[ind-int(bins/2):ind+int(bins/2)]

                for b in trackrog[0]:
                    if(np.isnan(new_flux[b])): 
                        new_flux[b]=0
                for m in range(0,len(mask)):
                    if(np.isnan(new_flux[m]) and (np.array(mask[m])==np.array([1,1])).all()):
                        mask[m]=label

            try:
                loc=np.where(np.array(ref_kepid)==el[4:13])
                loc_f=[m for m in loc[0] if str(av_pl[m])==str(len(hdu)-2)]
                if(len(loc_f)==0):
                    print("not in catalog:",len(hdu)-2)
                    label=[0,1]
                else:
                    if(ref_label[loc_f[0]]=='CONFIRMED'): label=[1,0]
                    elif(ref_label[loc_f[0]]=='CANDIDATE'): label=[1,0]
                    else: label=[0,1]
            except ValueError as ve:
                print("miss ind:",el[4:13])
                label=[0,0]
            for m in range(0,len(mask)):
                if((np.array(mask[m])==np.array([1,1])).all()):
                    mask[m]=label

            lightcurve=red_flux
            break

        if(len(lightcurve)==0):
            print('miss',el[4:13])
            continue      
        np.savetxt(pathout+'xlabel/'+el[4:13],lightcurve,delimiter=' ')
        np.savetxt(pathout+'ylabel/'+el[4:13],mask,delimiter=' ')
        print(tick,'hit:',el[4:13],np.array(lightcurve).shape,mask.shape,len(hdu)-2)

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
        ax[i][j].plot(df2*df[np.argmin(df)])
        ax[i][j].plot(df)
        #ax[i][j].plot(-df[2][:2000]*min(df[0]))
        #ax[i][j].plot(-df[1][:2000]*min(df[0]))
        i=i+1
        if(i==r):
            j=j+1
            i=0
        if(j==r): break

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
        
        check_noise=[(df2[i]==np.array([0,0])).all() for i in range(0,len(df2))]
        #print(np.array(check_noise).sum(),np.isnan(df).sum())
        if(np.array(check_noise).all()): 
            print('noise',el[0:9])
            continue
        
        X_train.append(df)
        Y_train.append(df2)
        print('hit',el[0:9])
    
    print(np.array(X_train).shape, np.array(Y_train).shape)
    Y_train=np.array(Y_train).reshape(len(Y_train),9600)
    np.savetxt('training_data/Xtrain_seg_mask_rv_bal.csv',X_train,delimiter=',')
    np.savetxt('training_data/Ytrain_seg_mask_rv_bal.csv',Y_train,delimiter=',')

#inst_segment(FILEPATH_FPS,'data_seg/',5000)
#training_sample('data_seg/')
#
#inst_seg_classifier(FILEPATH_DATA,'inst_seg/classifier/',4800)
#inst_seg_classifier(FILEPATH_FPS,'inst_seg/classifier/',4800)
#plot_inst_seg('inst_seg/segment_map/',5)
#sem_segment(FILEPATH_DATA,'sem_seg2/',4800)
inst_training_sample('inst_seg/',3200)

#sem_training_sample_tot('sem_seg2/')

#plot_seg('sem_seg/',4)
#plt.show()

