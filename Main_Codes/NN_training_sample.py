#get a training sample from the labeled directory structure. Consolidate stuff here
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import ascii
import os

np.random.seed(122334)

#these first three functions create training samples as per what we want PL, NPL, AFP for some various configurations

def global_view(tot,bp):
    #first we need to accumulate the data... we start off with a small NN... 2000 samples of global view? 500 from each?
    X_train=[]
    X_test=[]
    Y_train=[]
    Y_test=[]
    pl_entry=os.scandir('data_red/global')
    fps_entry=os.scandir('fps_red/global')
    npl_entry=os.scandir('nonpl_data_red2/global')
    nfps_entry=os.scandir('nonpl_fps_red2/global')

    pl_entry=list(pl_entry)
    fps_entry=list(fps_entry)
    npl_entry=list(npl_entry)
    nfps_entry=list(nfps_entry)

    np.random.shuffle(pl_entry)
    np.random.shuffle(fps_entry)
    np.random.shuffle(npl_entry)
    np.random.shuffle(nfps_entry)

    i=0
    for el in list(pl_entry):
        df=pd.read_csv('data_red/global/'+el.name,sep=" ")
        if(len(df['flux'])<2000): continue
        if(i>=bp):
            X_test.append(np.array(df['flux'].iloc[0:2000]))
            Y_test.append([1,0,0])
        else:
            X_train.append(np.array(df['flux'].iloc[0:2000]))
            Y_train.append([1,0,0])
        i=i+1
        if(i==tot): break

    j=0
    for el in list(fps_entry):
        df=pd.read_csv('fps_red/global/'+el.name,sep=" ")
        if(len(df['flux'])<2000): continue
        if(j>=bp):
            X_test.append(np.array(df['flux'].iloc[:2000],dtype='float32'))
            Y_test.append([0,1,0])
        else:
            X_train.append(np.array(df['flux'].iloc[:2000]))
            Y_train.append([0,1,0])
        j=j+1
        if(j==tot): break 

    k=0
    for el in list(npl_entry):
        df=pd.read_csv('nonpl_data_red2/global/'+el.name,sep=" ")
        if(len(df['flux'])<2000): continue
        if(k>=bp/2):
            X_test.append(np.array(df['flux'].iloc[0:2000]))
            Y_test.append([0,0,1])
        else:
            X_train.append(np.array(df['flux'].iloc[0:2000]))
            Y_train.append([0,0,1])
        k=k+1 
        if(k==tot/2): break

    l=0
    for el in list(nfps_entry):
        df=pd.read_csv('nonpl_fps_red2/global/'+el.name,sep=" ")
        if(len(df['flux'])<2000): continue
        if(l>=bp/2):
            X_test.append(np.array(df['flux'].iloc[:2000]))
            Y_test.append([0,0,1])
        else:
            X_train.append(np.array(df['flux'].iloc[:2000]))
            Y_train.append([0,0,1])
        l=l+1
        if(l==tot/2): break

    print(np.array(X_train).shape,np.array(Y_train).shape)
    print(np.array(X_test).shape,np.array(Y_test).shape)
    np.savetxt('Xtrain.csv', np.array(X_train), delimiter=',')
    np.savetxt('Ytrain.csv', np.array(Y_train), delimiter=',')
    np.savetxt('Xtest.csv', np.array(X_test), delimiter=',')
    np.savetxt('Ytest.csv', np.array(Y_test), delimiter=',')

def local_global_view(tot,bp):
    #first we need to accumulate the data... we start off with a small NN... 2000 samples of global view? 500 from each?
    X_train_l=[]
    X_test_l=[]
    Y_train=[]
    Y_test=[]
    X_train_g=[]
    X_test_g=[]
    pl_entry=os.scandir('data_red/local')
    fps_entry=os.scandir('fps_red/local')
    npl_entry=os.scandir('nonpl_data_red2/local')
    nfps_entry=os.scandir('nonpl_fps_red2/local')
    np.random.seed(122334)

    pl_entry=list(pl_entry)
    fps_entry=list(fps_entry)
    npl_entry=list(npl_entry)
    nfps_entry=list(nfps_entry)

    np.random.shuffle(pl_entry)
    np.random.shuffle(fps_entry)
    np.random.shuffle(npl_entry)
    np.random.shuffle(nfps_entry)

    i=0
    for el in list(pl_entry):
        dfl=pd.read_csv('data_red/local/'+el.name,sep=" ")
        dfg=pd.read_csv('data_red/global/'+el.name[:11]+'_g',sep=" ")
        if(len(dfl['flux'])<200): continue
        if(i>=bp):
            X_test_l.append(np.array(dfl['flux'].iloc[0:200]))
            X_test_g.append(np.array(dfg['flux'].iloc[0:2000]))
            Y_test.append([1,0,0])
        else:
            X_train_l.append(np.array(dfl['flux'].iloc[0:200]))
            X_train_g.append(np.array(dfg['flux'].iloc[0:2000]))
            Y_train.append([1,0,0])
        i=i+1
        if(i==tot): break

    j=0
    for el in list(fps_entry):
        dfl=pd.read_csv('fps_red/local/'+el.name,sep=" ")
        dfg=pd.read_csv('fps_red/global/'+el.name[:11]+'_g',sep=" ")
        if(len(dfl['flux'])<200): continue
        if(j>=bp):
            X_test_l.append(np.array(dfl['flux'].iloc[:200],dtype='float32'))
            X_test_g.append(np.array(dfg['flux'].iloc[:2000],dtype='float32'))
            Y_test.append([0,1,0])
        else:
            X_train_l.append(np.array(dfl['flux'].iloc[:200]))
            X_train_g.append(np.array(dfg['flux'].iloc[:2000]))
            Y_train.append([0,1,0])
        j=j+1
        if(j==tot): break 

    k=0
    for el in list(npl_entry):
        dfl=pd.read_csv('nonpl_data_red2/local/'+el.name,sep=" ")
        dfg=pd.read_csv('nonpl_data_red2/global/'+el.name[:11],sep=" ")
        if(len(dfl['flux'])<200): continue
        if(k>=bp/2):
            X_test_l.append(np.array(dfl['flux'].iloc[0:200]))
            X_test_g.append(np.array(dfg['flux'].iloc[0:2000]))
            Y_test.append([0,0,1])
        else:
            X_train_l.append(np.array(dfl['flux'].iloc[0:200]))
            X_train_g.append(np.array(dfg['flux'].iloc[0:2000]))
            Y_train.append([0,0,1])
        k=k+1 
        if(k==tot/2): break

    l=0
    for el in list(nfps_entry):
        dfl=pd.read_csv('nonpl_fps_red2/local/'+el.name,sep=" ")
        dfg=pd.read_csv('nonpl_fps_red2/global/'+el.name[:11],sep=" ")
        if(len(dfl['flux'])<200): continue
        if(l>=bp/2):
            X_test_l.append(np.array(dfl['flux'].iloc[:200]))
            X_test_g.append(np.array(dfg['flux'].iloc[:2000]))
            Y_test.append([0,0,1])
        else:
            X_train_l.append(np.array(dfl['flux'].iloc[:200]))
            X_train_g.append(np.array(dfg['flux'].iloc[:2000]))
            Y_train.append([0,0,1])
        l=l+1
        if(l==tot/2): break

    print(np.array(X_train_l).shape,np.array(X_train_g).shape,np.array(Y_train).shape)
    print(np.array(X_test_l).shape,np.array(X_test_g).shape,np.array(Y_test).shape)
    np.savetxt('Xtrain2g.csv', np.array(X_train_g), delimiter=',')
    np.savetxt('Xtrain2l.csv', np.array(X_train_l), delimiter=',')
    np.savetxt('Ytrain2.csv', np.array(Y_train), delimiter=',')
    np.savetxt('Xtest2g.csv', np.array(X_test_g), delimiter=',')
    np.savetxt('Xtest2l.csv', np.array(X_test_l), delimiter=',')
    np.savetxt('Ytest2.csv', np.array(Y_test), delimiter=',')

def global_view_nononpl(tot,bp):
    #first we need to accumulate the data... we start off with a small NN... 2000 samples of global view? 500 from each?
    X_train=[]
    X_test=[]
    Y_train=[]
    Y_test=[]
    pl_entry=os.scandir('data_red/global')
    npl_entry=os.scandir('nonpl_data_red2/global')
    nfps_entry=os.scandir('nonpl_fps_red2/global')

    pl_entry=list(pl_entry)
    npl_entry=list(npl_entry)
    nfps_entry=list(nfps_entry)

    np.random.shuffle(pl_entry)
    np.random.shuffle(npl_entry)
    np.random.shuffle(nfps_entry)

    i=0
    for el in list(pl_entry):
        df=pd.read_csv('data_red/global/'+el.name,sep=" ")
        if(len(df['flux'])<2000): continue
        if(i>=bp):
            X_test.append(np.array(df['flux'].iloc[0:2000]))
            Y_test.append([1,0])
        else:
            X_train.append(np.array(df['flux'].iloc[0:2000]))
            Y_train.append([1,0])
        i=i+1
        if(i==tot): break

    k=0
    for el in list(npl_entry):
        df=pd.read_csv('nonpl_data_red2/global/'+el.name,sep=" ")
        if(len(df['flux'])<2000): continue
        if(k>=bp/2):
            X_test.append(np.array(df['flux'].iloc[0:2000]))
            Y_test.append([0,1])
        else:
            X_train.append(np.array(df['flux'].iloc[0:2000]))
            Y_train.append([0,1])
        k=k+1 
        if(k==tot/2): break

    l=0
    for el in list(nfps_entry):
        df=pd.read_csv('nonpl_fps_red2/global/'+el.name,sep=" ")
        if(len(df['flux'])<2000): continue
        if(l>=bp/2):
            X_test.append(np.array(df['flux'].iloc[:2000]))
            Y_test.append([0,1])
        else:
            X_train.append(np.array(df['flux'].iloc[:2000]))
            Y_train.append([0,1])
        l=l+1
        if(l==tot/2): break 

    print(np.array(X_train).shape,np.array(Y_train).shape)
    print(np.array(X_test).shape,np.array(Y_test).shape)
    np.savetxt('training_data/Xtrain_no_fps.csv', np.array(X_train), delimiter=',')
    np.savetxt('training_data/Ytrain_no_fps.csv', np.array(Y_train), delimiter=',')
    np.savetxt('training_data/Xtest_no_fps.csv', np.array(X_test), delimiter=',')
    np.savetxt('training_data/Ytest_no_fps.csv', np.array(Y_test), delimiter=',')

#this creates a trainig sample as per autovetters labels.
def get_strings(ids):
    str_b=[]
    for el in ids:
        no=len(str(el))
        val=str(el)
        for i in range(0,9-no):
            val='0'+val
        str_b.append(val)
    return(str_b)

def autovetter_labels(tot,bp):
    X_train=[]
    X_test=[]
    Y_train=[]
    Y_test=[]
    pl_entry=os.scandir('temp_dir/global')
    av_entry=ascii.read('autovetter_label.tab')
    ref_kepid=av_entry['kepid']
    ref_kepid=get_strings(ref_kepid)
    ref_label=av_entry['av_training_set']
    pl_entry=list(pl_entry)

    np.random.shuffle(pl_entry)

    i=0
    for el in list(pl_entry):
        df=pd.read_csv('temp_dir/global/'+el.name,sep=" ")
        if(len(df['flux'])<2000): continue
        try:
            loc=ref_kepid.index(el.name[0:9])
        except ValueError as ve:
            continue
        if(ref_label[loc]=='UNK'): continue
        if(i>=bp):
            X_test.append(np.array(df['flux'].iloc[0:2000]))
            if(ref_label[loc]=='PC'): Y_test.append([1,0])
            else: Y_test.append([0,1])
        else:
            X_train.append(np.array(df['flux'].iloc[0:2000]))
            if(ref_label[loc]=='PC'): Y_train.append([1,0])
            else: Y_train.append([0,1])
        i=i+1
        if(i==tot): break

    #print(np.array(X_train).shape,np.array(Y_train).shape)
    #print(np.array(X_test).shape,np.array(Y_test).shape)
    #print(Y_train,len(Y_train))
    #print(Y_test,len(Y_test))
    np.savetxt('training_data/Xtrain_av.csv', np.array(X_train), delimiter=',')
    np.savetxt('training_data/Ytrain_av.csv', np.array(Y_train), delimiter=',')
    np.savetxt('training_data/Xtest_av.csv', np.array(X_test), delimiter=',')
    np.savetxt('training_data/Ytest_av.csv', np.array(Y_test), delimiter=',')

def autovetter_robovetter_labels(tot,bp):
    X_train=[]
    X_test=[]
    Y_trainR=[]
    Y_trainA=[]
    Y_testR=[]
    Y_testA=[]
    pl_entry=os.listdir('temp_dir/global')
    av_entry=ascii.read('autovetter_label.tab')
    rv_entry=ascii.read('robovetter_label.dat')
    av_kepid=av_entry['kepid']
    av_kepid=get_strings(av_kepid)
    av_pl=np.array(av_entry['tce_plnt_num'])
    rv_pl=np.array(rv_entry['tce_plnt_num'])
    rv_kepid=rv_entry['kepid']
    rv_kepid=get_strings(rv_kepid)
    av_label=av_entry['av_training_set']
    rv_label=rv_entry['label']
  
    np.random.shuffle(pl_entry)

    i=0
    for el in list(pl_entry):
        df=pd.read_csv('temp_dir/global/'+el,sep=" ")
        if(len(df['flux'])<2000): continue
        try:
            av_loc=np.where(np.array(av_kepid)==el[0:9])
            rv_loc=np.where(np.array(rv_kepid)==el[0:9])

            rv_loc_f=[m for m in rv_loc[0] if str(rv_pl[m])==el[-3]]
            av_loc_f=[m for m in av_loc[0] if str(av_pl[m])==el[-3]]
            #print(el[-3],av_loc_f,rv_loc_f)
        except ValueError as ve:
            continue
        
        if(len(rv_loc_f)==0 or len(av_loc_f)==0): continue
        if(av_label[av_loc_f[0]]=='UNK'): continue
        if(av_label[rv_loc_f[0]]=='CANDIDATE'): continue
        if(i>=bp):
            X_test.append(np.array(df['flux'].iloc[0:2000]))
            if(rv_label[rv_loc_f[0]]=='FPS'): Y_testR.append([0,1])
            else: Y_testR.append([1,0])
            if(av_label[av_loc_f[0]]=='PC'): Y_testA.append([1,0])
            else: Y_testA.append([0,1])
        else:
            X_train.append(np.array(df['flux'].iloc[0:2000]))
            if(rv_label[rv_loc_f[0]]=='FPS'): Y_trainR.append([0,1])
            else: Y_trainR.append([1,0])
            if(av_label[av_loc_f[0]]=='PC'): Y_trainA.append([1,0])
            else: Y_trainA.append([0,1])
        i=i+1
        if(i==tot): break

    print(np.array(X_train).shape,np.array(Y_trainA).shape,np.array(Y_trainR).shape)
    print(np.array(X_test).shape,np.array(Y_testA).shape,np.array(Y_testR).shape)
    
    np.savetxt('training_data/Xtrain_big.csv', np.array(X_train), delimiter=',')
    np.savetxt('training_data/YtrainR_big.csv', np.array(Y_trainR), delimiter=',')
    np.savetxt('training_data/Xtest_big.csv', np.array(X_test), delimiter=',')
    np.savetxt('training_data/YtestR_big.csv', np.array(Y_testR), delimiter=',')
    np.savetxt('training_data/YtestA_big.csv', np.array(Y_testA), delimiter=',')
    np.savetxt('training_data/YtrainA_big.csv', np.array(Y_trainA), delimiter=',')

def cleaned_data_NN(tot,bp):
    #cleanned data with autovetter labels for now, can be done with robovetter too
    X_train=[]
    X_trainl=[]
    X_test=[]
    X_testl=[]
    Y_train=[]
    Y_test=[]
    pl_entry=os.listdir('temp_dir/global')
    av_entry=ascii.read('autovetter_label.tab')
    av_pl=np.array(av_entry['tce_plnt_num'])
    ref_kepid=av_entry['kepid']
    ref_kepid=get_strings(ref_kepid)
    ref_label=av_entry['av_training_set']
    pl_entry=list(pl_entry)

    np.random.shuffle(pl_entry)

    i=0
    for el in list(pl_entry):
        df=pd.read_csv('temp_dir/global/'+el,sep=" ")
        dfl=pd.read_csv('temp_dir/local/'+el[0:11]+'_l',sep=" ")
        if(len(df['flux'])<2000): continue
        if(len(dfl['flux'])<200): continue
        try:
            loc=np.where(np.array(ref_kepid)==el[0:9])
            loc_f=[m for m in loc[0] if str(av_pl[m])==el[-3]]
        except ValueError as ve:
            continue
        if(len(loc_f)==0): continue
        #the idea behind putting the zero is only coz loc_f is inherently a tuple... the way the alg is it will have only
        #one element tho at all time.
        if(ref_label[loc_f[0]]=='UNK'): continue
        if(av_pl[loc_f[0]]!=1): continue

        flux=df['flux'].iloc[0:2000]
        fluxl=dfl['flux'].iloc[0:200]
        #flux=flux-np.median(flux)
        #flux=flux/np.abs(flux[np.argmin(flux)])
        #fluxl=fluxl-np.median(fluxl)
        #fluxl=fluxl/np.abs(fluxl[np.argmin(fluxl)])

        if(i>=bp):
            X_test.append(np.array(flux))
            X_testl.append(np.array(fluxl))
            if(ref_label[loc_f[0]]=='PC'): Y_test.append([1,0])
            else: Y_test.append([0,1])
        else:
            X_train.append(np.array(flux))
            X_trainl.append(np.array(fluxl))
            if(ref_label[loc_f[0]]=='PC'): Y_train.append([1,0])
            else: Y_train.append([0,1])
        i=i+1
        print(av_pl[loc_f[0]],print(len(flux),len(fluxl)))
        if(i==tot): break

    #print(np.array(X_train).shape,np.array(Y_train).shape)
    #print(np.array(X_test).shape,np.array(Y_test).shape)
    #print(Y_train,len(Y_train))
    #print(Y_test,len(Y_test))
    np.savetxt('training_data/Xtrain_av_clean.csv', np.array(X_train), delimiter=',')
    np.savetxt('training_data/Ytrain_av_clean.csv', np.array(Y_train), delimiter=',')
    np.savetxt('training_data/Xtrainloc_av_clean.csv', np.array(X_trainl), delimiter=',')
    #np.savetxt('training_data/Xtest_av.csv', np.array(X_test), delimiter=',')
    #np.savetxt('training_data/Ytest_av.csv', np.array(Y_test), delimiter=',')

def raw_bins(tot,bp,pathin):
    X_train=[]
    X_test=[]
    Y_train=[]
    Y_test=[]
    pl_entry=os.listdir(pathin)
    av_entry=ascii.read('robovetter_label.dat')
    eb_entry=ascii.read('eb_label.dat')
    av_pl=np.array(av_entry['tce_plnt_num'])
    ref_kepid=av_entry['kepid']
    eb_kepid=eb_entry['kepid']
    ref_kepid=get_strings(ref_kepid)
    eb_kepid=np.array(get_strings(eb_kepid))
    ref_label=av_entry['label']
    pl_entry=list(pl_entry)

    np.random.shuffle(pl_entry)

    i=0
    for el in list(pl_entry):
        df=np.loadtxt(pathin+el)
        try:
            loc=np.where(np.array(ref_kepid)==el[0:9])
            loc_f=[m for m in loc[0] if str(av_pl[m])=='1']
        except ValueError as ve:
            continue
        if(len(loc_f)==0): continue
        #the idea behind putting the zero is only coz loc_f is inherently a tuple... the way the alg is it will have only
        #one element tho at all time.
        if(ref_label[loc_f[0]]=='UNK'): continue
        #if(av_pl[loc_f[0]]!=1): continue

        if(len(df.shape)<2):    df=df.reshape(1,500)
        if(len(df)<3): chosen_ind=np.arange(0,len(df),1)
        else: chosen_ind=np.random.randint(0,len(df),size=3)
        
        #cleaner training samples?
         
        #med=[np.median(df[k]) for k in chosen_ind]
        #std=[np.std(df[k]) for k in chosen_ind]
        count_tr=[(df[k] < np.median(df[k])-2.5*np.std(df[k])).sum() for k in chosen_ind]
        chosen_ind=np.delete(chosen_ind,np.where(np.array(count_tr)<7)[0])
        if(len(chosen_ind)==0): continue

        print(el[0:9],len(chosen_ind))
    
        if(i>=bp):
            [ X_test.append(np.array(df[k])) for k in chosen_ind]
            if(ref_label[loc_f[0]]=='CONFIRMED'): [Y_test.append([1,0,0]) for k in chosen_ind]
            elif(np.any(eb_kepid==el[0:9])): [Y_test.append([0,0,1]) for k in chosen_ind]
            elif(ref_label[loc_f[0]]=='FPS'): [Y_test.append([0,1,0]) for k in chosen_ind]
            #else: [Y_test.append([0,1,0]) for k in chosen_ind]
        else:
            [ X_train.append(np.array(df[k])) for k in chosen_ind]
            if(ref_label[loc_f[0]]=='CONFIRMED'): [Y_train.append([1,0,0]) for k in chosen_ind]
            elif(np.any(eb_kepid==el[0:9])): [Y_train.append([0,0,1]) for k in chosen_ind]
            elif(ref_label[loc_f[0]]=='FPS'): [Y_train.append([0,1,0]) for k in chosen_ind]
            #else: [Y_train.append([0,1,0]) for k in chosen_ind]
        i=i+1
        if(i==tot): break

    print(np.array(X_train).shape,np.array(Y_train).shape)
    print(np.array(X_test).shape,np.array(Y_test).shape)

    temp1=[]
    temp2=[]
    filtind=[i for i in range(0,len(Y_train)) if (Y_train[i]==np.array([1,0,0])).all()]
    filtind2=[i for i in range(0,len(Y_train)) if (Y_train[i]==np.array([0,1,0])).all()]
    filtindeb=[i for i in range(0,len(Y_train)) if (Y_train[i]==np.array([0,0,1])).all()]
    print(len(filtind),len(filtind2),len(filtindeb)) 
    #print(min(len(filtind[0]),len(filtind2[0])))
    for i in range(0,min(len(filtind),len(filtind2),len(filtindeb))):
        temp1.append(X_train[filtind[i]])
        temp2.append(Y_train[filtind[i]])
        temp1.append(X_train[filtind2[i]])
        temp2.append(Y_train[filtind2[i]])
        temp1.append(X_train[filtindeb[i]])
        temp2.append(Y_train[filtindeb[i]])

    print(np.array(temp1).shape,np.array(temp2).shape)
    #print(Y_train,len(Y_train))
    #print(Y_test,len(Y_test))
    np.savetxt('training_data/Xtrain_rv_raw500_3.csv', np.array(temp1), delimiter=',')
    np.savetxt('training_data/Ytrain_rv_raw500_3.csv', np.array(temp2), delimiter=',')
    #np.savetxt('training_data/Xtest_av.csv', np.array(X_test), delimiter=',')
    #np.savetxt('training_data/Ytest_av.csv', np.array(Y_test), delimiter=',')

def raw_bins_2class(tot,bp,pathin):
    X_train=[]
    X_test=[]
    Y_train=[]
    Y_test=[]
    pl_entry=os.listdir(pathin)
    av_entry=ascii.read('autovetter_label.tab')
    av_pl=np.array(av_entry['tce_plnt_num'])
    ref_kepid=av_entry['kepid']
    ref_kepid=get_strings(ref_kepid)
    ref_label=av_entry['av_training_set']
    pl_entry=list(pl_entry)

    np.random.shuffle(pl_entry)

    i=0
    tab=0
    for el in list(pl_entry):
        df=np.loadtxt(pathin+el)
        try:
            loc=np.where(np.array(ref_kepid)==el[0:9])
            loc_f=[m for m in loc[0] if str(av_pl[m])=='1']
        except ValueError as ve:
            continue
        if(len(loc_f)==0): continue
        #the idea behind putting the zero is only coz loc_f is inherently a tuple... the way the alg is it will have only
        #one element tho at all time.
        if(ref_label[loc_f[0]]=='UNK'): continue
        #if(av_pl[loc_f[0]]!=1): continue

        if(len(df.shape)<2):    df=df.reshape(1,200)
        if(len(df)<5): chosen_ind=np.arange(0,len(df),1)
        else: chosen_ind=np.random.randint(0,len(df),size=5)
        
        #cleaner training samples?
         
        #med=[np.median(df[k]) for k in chosen_ind]
        #std=[np.std(df[k]) for k in chosen_ind]
        count_tr=[(df[k] < np.median(df[k])-3*np.std(df[k])).sum() for k in chosen_ind]
        chosen_ind=np.delete(chosen_ind,np.where(np.array(count_tr)<7)[0])
        if(len(chosen_ind)==0): continue

        print(el[0:9],len(chosen_ind))
    
        if(i>=bp):
            [ X_test.append(np.array(df[k])) for k in chosen_ind]
            if(ref_label[loc_f[0]]=='PC'): [Y_test.append([1,0]) for k in chosen_ind]
            else: [Y_test.append([0,1]) for k in chosen_ind]
            #elif(ref_label[loc_f[0]]=='FPS'): [Y_test.append([0,1]) for k in chosen_ind]
            #else: [Y_test.append([0,1,0]) for k in chosen_ind]
        else:
            [ X_train.append(np.array(df[k])) for k in chosen_ind]
            if(ref_label[loc_f[0]]=='PC'): [Y_train.append([1,0]) for k in chosen_ind]
            else: [Y_train.append([0,1]) for k in chosen_ind]
            #elif(ref_label[loc_f[0]]=='FPS'): [Y_train.append([0,1]) for k in chosen_ind]
            #else: [Y_train.append([0,1,0]) for k in chosen_ind]
        i=i+1
        if(i==tot): break

    print(np.array(X_train).shape,np.array(Y_train).shape)
    print(np.array(X_test).shape,np.array(Y_test).shape)

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
    np.savetxt('training_data/Xtrain_av_raw200_2_3d0.csv', np.array(temp1), delimiter=',')
    np.savetxt('training_data/Ytrain_av_raw200_2_3d0.csv', np.array(temp2), delimiter=',')
    #np.savetxt('training_data/Xtest_av.csv', np.array(X_test), delimiter=',')
    #np.savetxt('training_data/Ytest_av.csv', np.array(Y_test), delimiter=',')

#global_view_nononpl(2100,2000)
#cleaned_data_NN(7000,6000)
raw_bins_2class(6000,5000,'data_red_raw_dirty200/')
