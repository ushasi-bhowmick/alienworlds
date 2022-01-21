from itertools import count
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from astropy.io import ascii, fits
import tensorflow as tf
import GetLightcurves as gc
import os

FILEPATH = 'E:\Masters_Project_Data\\alienworlds_fps\\'

def stitched_lc(pathin,tot,bp):
    X_train=[]
    X_test=[]
    Y_train=[]
    Y_test=[]
    pl_entry=os.listdir(pathin)
    av_entry=ascii.read('autovetter_label.tab')
    av_pl=np.array(av_entry['tce_plnt_num'])
    ref_kepid=[('0000'+str(el)[:9])[-9:] for el in av_entry['kepid']]
    print(ref_kepid[:10])
    ref_label=av_entry['av_training_set']
    
    i=0
    for el in list(pl_entry):
        try:    df=np.loadtxt(pathin+el)
        except: continue
        if(len(df)<3): continue
        if(len(df)>8): continue
        #if(len(df)!=3): continue
        #here we are checking the label from the autovetter catalog, removing stuff labeled unknown and stuff not in catalog.
        try:
            loc=np.where(np.array(ref_kepid)==el[0:9])
            loc_f=[m for m in loc[0] if str(av_pl[m])==str(1)]
        except ValueError as ve:
            continue
        if(len(loc_f)==0): continue
        if(ref_label[loc_f[0]]=='UNK'): continue

        print(el[0:9],df.shape)
        #next is the stitching algorithm...take 600 pixels as ballpark number 
        lightcurve=[]
        for val in df:
            row=[]
            if(len(val)<200):
                med=np.median(val)
                std=np.std(val)
                new=[m for m in val if(m > med-0.7*std)]
                noise=np.random.normal(np.median(new),np.std(new),size=200-len(val))
                row=np.concatenate((noise[:int(len(noise)/2)],val,noise[int(len(noise)/2):]),axis=None)
                #print("checkdim:",row.shape)
            elif(len(val)>200):
                bndn=int(len(val)/200)
                if(bndn>1): row=[np.median(val[i:i+bndn]) for i in range(0,len(val),bndn)]
                else: row=val
                mp=int(len(row)/2)
                row=row[mp-100:mp+100]
            lightcurve=np.concatenate((lightcurve,row),axis=None)
        #print(lightcurve.shape)
        #then we have appending routines.
        if(i>=bp):
            X_test.append(np.array(lightcurve))
            if(ref_label[loc_f[0]]=='PC'): Y_test.append([1,0]) 
            else: Y_test.append([0,1])
            #elif(ref_label[loc_f[0]]=='FPS'): [Y_test.append([0,1]) for k in chosen_ind]
            #else: [Y_test.append([0,1,0]) for k in chosen_ind]
        else:
            X_train.append(np.array(lightcurve)) 
            if(ref_label[loc_f[0]]=='PC'): Y_train.append([1,0])
            else: Y_train.append([0,1]) 
            #elif(ref_label[loc_f[0]]=='FPS'): [Y_train.append([0,1]) for k in chosen_ind]
            #else: [Y_train.append([0,1,0]) for k in chosen_ind]
        i=i+1
        if(i==tot): break
    print("unique count:",i)
    print(np.array(X_train).shape,np.array(Y_train).shape)
    print(np.array(X_test).shape,np.array(Y_test).shape)

    #shuffle up the array to make better training samples.
    arr=np.arange(0,len(X_train),1)
    np.random.shuffle(arr)
    X_train=[X_train[p] for p in arr]
    Y_train=[Y_train[p] for p in arr]

    #this is only to ensure the training set has uniform distribution. Remove if not needed.
    temp1=[]
    temp2=[]
    filtind=[i for i in range(0,len(Y_train)) if (Y_train[i]==np.array([1,0])).all()]
    filtind2=[i for i in range(0,len(Y_train)) if (Y_train[i]==np.array([0,1])).all()]
    print(len(filtind),len(filtind2)) 
    #print(min(len(filtind[0]),len(filtind2[0])))
    for i in range(0,min(len(filtind),len(filtind2))):
        temp1.append(X_train[filtind[i]])
        temp2.append(Y_train[filtind[i]])
        temp1.append(X_train[filtind2[i]])
        temp2.append(Y_train[filtind2[i]])
    print(np.array(temp1).shape,np.array(temp2).shape)
    #print(Y_train,len(Y_train))
    #print(Y_test,len(Y_test))
    np.savetxt('training_data/Xtrain_av_stitch.csv', np.array(temp1), delimiter=',')
    np.savetxt('training_data/Ytrain_av_stitch.csv', np.array(temp2), delimiter=',')
    #np.savetxt('training_data/Xtest_av.csv', np.array(X_test), delimiter=',')
    #np.savetxt('training_data/Ytest_av.csv', np.array(Y_test), delimiter=',')


def compr_TS(pathin,tot,bp,max_ex):
    X_train=[]
    X_test=[]
    Y_train=[]
    Y_test=[]
    I_train=[]
    I_test=[]
    pl_entry=os.listdir(pathin)
    #pl_entry_proc=os.listdir('temp_dir/local/')
    av_entry=ascii.read('autovetter_label.tab')
    av_pl=np.array(av_entry['tce_plnt_num'])
    ref_kepid=[('0000'+str(el)[:9])[-9:] for el in av_entry['kepid']]
    print(ref_kepid[:10])
    ref_label=av_entry['av_training_set']

    i=0
    for el in list(pl_entry):
        try:    df=np.loadtxt(pathin+el)
        except: 
            print("miss load:",el[0:9])
            continue
        if(len(df.shape)<2): df=df.reshape(1,len(df))

        try:
            loc=np.where(np.array(ref_kepid)==el[0:9])
            loc_f=[m for m in loc[0] if str(av_pl[m])==el[10]]
        except ValueError as ve:
            print("miss ind:",el[0:9])
            continue
        if(len(loc_f)==0): continue
        if(ref_label[loc_f[0]]=='UNK'): continue

        try: 
            int_df=pd.read_csv('new_loc_glob/local/'+el[0:11]+'_l',sep=" ")
            intf=int_df['flux'].iloc[0:200]
        except: 
            print('miss I:',el[0:9])
            continue

        #if(len(intf)<200): continue

        print(el[0:9],df.shape,intf.shape)
        max_ex=min(len(df),max_ex)

        if(i>=bp):
            check=np.array([len(df[k]) for k in range(0,max_ex)])
            if(np.any(check<6000)): continue 
            if(ref_label[loc_f[0]]=='PC'): [Y_test.append([1,0]) for k in range(0,max_ex)]
            else: [Y_test.append([0,1]) for k in range(0,max_ex)]
            [ X_test.append(np.array(df[k])) for k in range(0,max_ex)]
            [I_test.append(intf) for k in range(0,max_ex)]
            #else: [Y_test.append([0,1,0]) for k in chosen_ind]
        else:
            check=np.array([len(df[k]) for k in range(0,max_ex)])
            if(np.any(check<6000)): continue 
            if(ref_label[loc_f[0]]=='PC'): [Y_train.append([1,0]) for k in range(0,max_ex)]
            else: [Y_train.append([0,1]) for k in range(0,max_ex)]
            [X_train.append(np.array(df[k])) for k in range(0,max_ex)]
            [I_train.append(intf) for k in range(0,max_ex)]
            #else: [Y_train.append([0,1,0]) for k in chosen_ind]
        i=i+1
        if(i==tot): break
    print("unique count:",i)
    print(np.array(X_train).shape,np.array(I_train).shape,np.array(Y_train).shape)
    print(np.array(X_test).shape,np.array(I_test).shape,np.array(Y_test).shape)

    #shuffle up the array to make better training samples.
    arr=np.arange(0,len(X_train),1)
    np.random.shuffle(arr)
    X_train=[X_train[p] for p in arr] 
    Y_train=[Y_train[p] for p in arr]
    I_train=[I_train[p] for p in arr]

    #this is only to ensure the training set has uniform distribution. Remove if not needed.
    tempX=[]
    tempY=[]
    tempI=[]
    filtind=[i for i in range(0,len(Y_train)) if (Y_train[i]==np.array([1,0])).all()]
    filtind2=[i for i in range(0,len(Y_train)) if (Y_train[i]==np.array([0,1])).all()]
    print(len(filtind),len(filtind2)) 
    #print(min(len(filtind[0]),len(filtind2[0])))
    for i in range(0,min(len(filtind),len(filtind2))):
        tempX.append(X_train[filtind[i]])
        tempY.append(Y_train[filtind[i]])
        tempI.append(I_train[filtind[i]])
        tempX.append(X_train[filtind2[i]])
        tempY.append(Y_train[filtind2[i]])
        tempI.append(I_train[filtind2[i]])
    print('new: ',np.array(tempX).shape,np.array(tempI).shape,np.array(tempY).shape)
   
    np.savetxt('training_data/Xtrain_av_rec.csv', np.array(tempX), delimiter=',')
    np.savetxt('training_data/Itrain_av_rec.csv', np.array(tempI), delimiter=',')
    np.savetxt('training_data/Ytrain_av_rec.csv', np.array(tempY), delimiter=',')
    #np.savetxt('training_data/Xtest_av.csv', np.array(X_test), delimiter=',')
    #np.savetxt('training_data/Ytest_av.csv', np.array(Y_test), delimiter=',')

def compr_TS_avg(pathin,tot,bp,max_ex):
    X_train=[]
    X_test=[]
    Y_train=[]
    Y_test=[]
    I_train=[]
    I_test=[]
    pl_entry=os.listdir(pathin)
    #pl_entry_proc=os.listdir('temp_dir/local/')
    av_entry=ascii.read('autovetter_label.tab')
    av_pl=np.array(av_entry['tce_plnt_num'])
    ref_kepid=[('0000'+str(el)[:9])[-9:] for el in av_entry['kepid']]
    print(ref_kepid[:10])
    ref_label=av_entry['av_training_set']

    i=0
    for el in list(pl_entry):
        try:    df=np.loadtxt(pathin+el)
        except: 
            print("miss load:",el[0:9])
            continue
        if(len(df.shape)<2): df=df.reshape(1,len(df))

        try:
            loc=np.where(np.array(ref_kepid)==el[0:9])
            loc_f=[m for m in loc[0] if str(av_pl[m])==el[10]]
        except ValueError as ve:
            print("miss ind:",el[0:9])
            continue
        if(len(loc_f)==0): continue
        if(ref_label[loc_f[0]]=='UNK'): continue

        try: 
            int_df=pd.read_csv('temp_dir/local/'+el[0:11]+'_l',sep=" ")
            intf=int_df['flux'].iloc[0:200]
        except: 
            print('miss I:',el[0:9])
            continue

        if(len(intf)<200): continue

        print(el[0:9],df.shape)
        max_ex=min(len(df),max_ex)

        if(i>=bp):
            check=np.array([len(df[k]) for k in range(0,max_ex)])
            if(np.any(check<2000)): continue 
            temp=[np.array(df[k]) for k in range(0,max_ex)]
            temp=np.array(np.sum(np.array(temp),axis=0))/6
            if(len(temp)!=2000): continue
            X_test.append(temp)
            if(ref_label[loc_f[0]]=='PC'): Y_test.append([1,0]) 
            else: Y_test.append([0,1]) 
            I_test.append(intf) 
        else:
            check=np.array([len(df[k]) for k in range(0,max_ex)])
            if(np.any(check<2000)): continue 
            temp=[np.array(df[k]) for k in range(0,max_ex)]
            temp=np.array(np.sum(np.array(temp),axis=0))/6
            if(len(temp)!=2000): continue
            X_train.append(temp)
            if(ref_label[loc_f[0]]=='PC'): Y_train.append([1,0])
            else: Y_train.append([0,1]) 
            I_train.append(intf) 
        i=i+1
        if(i==tot): break
    print("unique count:",i)
    print(np.array(X_train).shape,np.array(I_train).shape,np.array(Y_train).shape)
    print(np.array(X_test).shape,np.array(I_test).shape,np.array(Y_test).shape)

    #shuffle up the array to make better training samples.
    arr=np.arange(0,len(X_train),1)
    np.random.shuffle(arr)
    X_train=[X_train[p] for p in arr] 
    Y_train=[Y_train[p] for p in arr]
    I_train=[I_train[p] for p in arr]

    #this is only to ensure the training set has uniform distribution. Remove if not needed.
    tempX=[]
    tempY=[]
    tempI=[]
    filtind=[i for i in range(0,len(Y_train)) if (Y_train[i]==np.array([1,0])).all()]
    filtind2=[i for i in range(0,len(Y_train)) if (Y_train[i]==np.array([0,1])).all()]
    print(len(filtind),len(filtind2)) 
    #print(min(len(filtind[0]),len(filtind2[0])))
    for i in range(0,min(len(filtind),len(filtind2))):
        tempX.append(X_train[filtind[i]])
        tempY.append(Y_train[filtind[i]])
        tempI.append(I_train[filtind[i]])
        tempX.append(X_train[filtind2[i]])
        tempY.append(Y_train[filtind2[i]])
        tempI.append(I_train[filtind2[i]])
    print('new: ',np.array(tempX).shape,np.array(tempI).shape,np.array(tempY).shape)
   
    np.savetxt('training_data/Xtrain_av_rec_avg.csv', np.array(tempX), delimiter=',')
    np.savetxt('training_data/Itrain_av_rec_avg.csv', np.array(tempI), delimiter=',')
    np.savetxt('training_data/Ytrain_av_rec_avg.csv', np.array(tempY), delimiter=',')
    #np.savetxt('training_data/Xtest_av.csv', np.array(X_test), delimiter=',')
    #np.savetxt('training_data/Ytest_av.csv', np.array(Y_test), delimiter=',')

def interp_TS(pathin,tot,intsize,max_ex):
    X_train=[]
    Y_train=[]
    pl_entry=os.listdir(pathin)
    av_entry=ascii.read('autovetter_label.tab')
    av_pl=np.array(av_entry['tce_plnt_num'])
    ref_kepid=[('0000'+str(el)[:9])[-9:] for el in av_entry['kepid']]
    print(ref_kepid[:10])
    ref_label=av_entry['av_training_set']

    i=0
    for el in list(pl_entry):
        try:    df=np.loadtxt(pathin+el)
        except: 
            print("miss load:",el[0:9])
            continue
        if(len(df.shape)<2): df=df.reshape(1,len(df))
        if(len(df[0])<2): continue

        try:
            loc=np.where(np.array(ref_kepid)==el[0:9])
            loc_f=[m for m in loc[0] if str(av_pl[m])==str(1)]
        except ValueError as ve:
            print("miss ind:",el[0:9])
            continue
        if(len(loc_f)==0): continue
        if(ref_label[loc_f[0]]=='UNK'): continue

        print(el[0:9],df.shape)
        max_ex=min(len(df),max_ex)

        #this is the interpolation
        interpx=np.arange(0,len(df[0]),1)
        newx=np.linspace(0,interpx[-1],intsize,endpoint=True)
        for k in range(0,max_ex):
            func=interp1d(interpx,df[k],kind='quadratic')
            newy=func(newx)
            if(ref_label[loc_f[0]]=='PC'): Y_train.append([1,0])
            else: Y_train.append([0,1]) 
            X_train.append(np.array(newy)) 
            print(el[0:9],len(newx),len(newy))

        i=i+1
        if(i==tot): break
    print("unique count:",i)
    print(np.array(X_train).shape,np.array(Y_train).shape)

    #shuffle up the array to make better training samples.
    arr=np.arange(0,len(X_train),1)
    np.random.shuffle(arr)
    X_train=[X_train[p] for p in arr] 
    Y_train=[Y_train[p] for p in arr]

    #this is only to ensure the training set has uniform distribution. Remove if not needed.
    '''
    tempX=[]
    tempY=[]
    filtind=[i for i in range(0,len(Y_train)) if (Y_train[i]==np.array([1,0])).all()]
    filtind2=[i for i in range(0,len(Y_train)) if (Y_train[i]==np.array([0,1])).all()]
    print(len(filtind),len(filtind2)) 
    #print(min(len(filtind[0]),len(filtind2[0])))
    for i in range(0,min(len(filtind),len(filtind2))):
        tempX.append(X_train[filtind[i]])
        tempY.append(Y_train[filtind[i]])
        tempX.append(X_train[filtind2[i]])
        tempY.append(Y_train[filtind2[i]])
    print('new: ',np.array(tempX).shape,np.array(tempY).shape)'''
   
    np.savetxt('training_data/Xtrain_av_interp_800_raw.csv', np.array(X_train), delimiter=',')
    np.savetxt('training_data/Ytrain_av_interp_800_raw.csv', np.array(Y_train), delimiter=',')

def raw_loc_glob(pathin,tot,bp,max_ex):
    X_train=[]
    X_test=[]
    Y_train=[]
    Y_test=[]
    X_trainl=[]
    X_testl=[]
    pl_entry=os.listdir(pathin)
    #pl_entry_proc=os.listdir('temp_dir/local/')
    av_entry=ascii.read('autovetter_label.tab')
    av_pl=np.array(av_entry['tce_plnt_num'])
    ref_kepid=[('0000'+str(el)[:9])[-9:] for el in av_entry['kepid']]
    print(ref_kepid[:10])
    ref_label=av_entry['av_training_set']

    i=0
    for el in list(pl_entry):
        try:    df=np.loadtxt(pathin+el)
        except: 
            print("miss load:",el[0:9])
            continue
        if(len(df.shape)<2): df=df.reshape(1,len(df))

        try:
            loc=np.where(np.array(ref_kepid)==el[0:9])
            loc_f=[m for m in loc[0] if str(av_pl[m])==el[10]]
        except ValueError as ve:
            print("miss ind:",el[0:9])
            continue
        if(len(loc_f)==0): continue
        if(ref_label[loc_f[0]]=='UNK'): continue

        try: 
            int_df=np.loadtxt('data_red_raw_dirty200/'+el[0:11]+'.dat',delimiter=" ")
            intf=int_df[0]
            if(len(intf)<200): continue
        except: 
            print('miss I:',el[0:9])
            continue

        #if(len(intf)<200): continue
        print(el[0:9],df.shape,intf.shape)
        max_ex=min(len(df),max_ex)

        if(i>=bp):
            check=np.array([len(df[k]) for k in range(0,max_ex)])
            if(np.any(check<6000)): continue 
            if(ref_label[loc_f[0]]=='PC'): [Y_test.append([1,0]) for k in range(0,max_ex)]
            else: [Y_test.append([0,1]) for k in range(0,max_ex)]
            [ X_test.append(np.array(df[k])) for k in range(0,max_ex)]
            [X_testl.append(intf) for k in range(0,max_ex)]
            #else: [Y_test.append([0,1,0]) for k in chosen_ind]
        else:
            check=np.array([len(df[k]) for k in range(0,max_ex)])
            if(np.any(check<6000)): continue 
            if(ref_label[loc_f[0]]=='PC'): [Y_train.append([1,0]) for k in range(0,max_ex)]
            else: [Y_train.append([0,1]) for k in range(0,max_ex)]
            [X_train.append(np.array(df[k])) for k in range(0,max_ex)]
            [X_trainl.append(intf) for k in range(0,max_ex)]
            #else: [Y_train.append([0,1,0]) for k in chosen_ind]
        i=i+1
        if(i==tot): break
    print("unique count:",i)
    print(np.array(X_train).shape,np.array(X_trainl).shape,np.array(Y_train).shape)
    print(np.array(X_test).shape,np.array(X_testl).shape,np.array(Y_test).shape)

    #shuffle up the array to make better training samples.
    arr=np.arange(0,len(X_train),1)
    np.random.shuffle(arr)
    X_train=[X_train[p] for p in arr] 
    Y_train=[Y_train[p] for p in arr]
    X_trainl=[X_trainl[p] for p in arr]

    #this is only to ensure the training set has uniform distribution. Remove if not needed.
    '''
    tempX=[]
    tempY=[]
    tempI=[]
    filtind=[i for i in range(0,len(Y_train)) if (Y_train[i]==np.array([1,0])).all()]
    filtind2=[i for i in range(0,len(Y_train)) if (Y_train[i]==np.array([0,1])).all()]
    print(len(filtind),len(filtind2)) 
    #print(min(len(filtind[0]),len(filtind2[0])))
    for i in range(0,min(len(filtind),len(filtind2))):
        tempX.append(X_train[filtind[i]])
        tempY.append(Y_train[filtind[i]])
        tempI.append(I_train[filtind[i]])
        tempX.append(X_train[filtind2[i]])
        tempY.append(Y_train[filtind2[i]])
        tempI.append(X_trainl[filtind2[i]])
    print('new: ',np.array(tempX).shape,np.array(tempI).shape,np.array(tempY).shape)'''
   
    np.savetxt('training_data/Xtrain_av_gl6000.csv', np.array(X_train), delimiter=',')
    np.savetxt('training_data/Xtrainl_av_gl6000.csv', np.array(X_trainl), delimiter=',')
    np.savetxt('training_data/Ytrain_av_gl6000.csv', np.array(Y_train), delimiter=',')
    #np.savetxt('training_data/Xtest_av.csv', np.array(X_test), delimiter=',')
    #np.savetxt('training_data/Ytest_av.csv', np.array(Y_test), delimiter=',')

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

def remove_nan_2(red_flux,bins):
    for i in range(0,len(red_flux)):
        if np.isnan(red_flux[i]):
            red_flux[i]=0

def open_the_file_and_chunk(filepath, hdu_no):
    hdu = fits.open(filepath)
    flux=hdu[hdu_no].data['LC_DETREND']
    #flux=hdu[len(hdu)-1].data['RESIDUAL_LC']
    tdurs = [hdu[n].header['TPERIOD'] for n in range(1,len(hdu)-1)]
    
    remove_nan_2(flux,4000)
    chunks = np.asarray([flux[i:i+4000] for i in range(0,len(flux)-4000, 4000)])
    return(chunks, tdurs)

def training_sample_dir(pathin, indir, fileout):
    testid,label = gc.read_tfr_record(pathin, ['id','label'],['b','ar'],[tf.string,tf.bool])
    #print(testid)
    #testid=np.asarray(testid[:,0])
    #print(len(testid))
    #TestID2 = [testid[i].numpy() for i in range(0,len(testid))]
    TestID2 = [str(testid[i])[2:11] for i in range(0,len(testid))]
    #print(TestID2)
    entries = os.listdir(indir)
    oparr=[]
    i=0
    for ids in TestID2:
        print(ids)
        x=[el for el in entries if(el.find(ids)>0)]
        print(x)
        chunks, tdurs = open_the_file_and_chunk(indir+x[0],1)
        oparr.append([chunks.reshape(-1),tdurs,ids,label[i]])
        i+=1
    gc.write_tfr_record(fileout, oparr, ['input','period','id','label'],
        ['ar','ar','b','ar'], ['float32','float32','string','bool'])



#stitched_lc('data_prelim_stitch/',6000,5000)
#compr_TS('data_red_shortdur_6000/',6000,5000,2)
#interp_TS('data_red_raw_dirty200/',6000,800,2) 
#raw_loc_glob('data_red_shortdur_6000/',6000,5000,2)
training_sample_dir('../../training_data/total_tstest',FILEPATH,'../../training_data/tstest')