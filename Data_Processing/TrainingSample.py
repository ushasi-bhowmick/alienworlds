#Story so far: We took in the downloaded data from preparing_data.py, now we have to consolidate it and all so that the final thing going into
#NN is in fact a CSV file bearing the training sample- X and Y labels.
from itertools import count
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import ascii
import os

np.random.seed(122334)

#these first three functions create training samples as per what we want PL, NPL, AFP for some various configurations

#these extract only global view light curves from the planet transit. This means phase folded on the transit period and binned 
#to a desired value. Right now its tuned to a three classifier thing, with planets, non planets and astrophysical false positives. Problematic 
#labels
def global_view(tot,bp):
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

#if we want both global and local views, this is our thing. once again global view is phase folded to  transit period and local
#view phase folded to transit duration. Function not recommended for use since labels are not done properly
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

#this is only global view for a smaller problem... no non planets are included in this function, and classification is only for 
#planets vs false positives. Function is not fit for use as the labels are not done properly
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

#this fixes the label issue by referencing autovetter labels from the catalog. much better and cleaner data now. This is a two classifier 
#with only global view. Not recommended coz it has the issue of misplaced planet labels. 
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


#this can create global view LC for both autovetter and robovetter labels. Misplaced label issue is solved
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

#this creates both local and global view LC for autovetter labels. This is called clean coz misplaced label issue is solved.
def cleaned_data_NN(tot,bp):
    #cleanned data with autovetter labels for now, can be done with robovetter too
    X_train=[]
    X_trainl=[]
    X_test=[]
    X_testl=[]
    Y_train=[]
    Y_test=[]
    pl_entry=os.listdir('new_loc_glob/global')
    av_entry=ascii.read('robovetter_label.dat')
    eb_entry=ascii.read('eb_label.dat')
    eb_kepid=eb_entry['kepid']
    eb_kepid=np.array(get_strings(eb_kepid))
    av_pl=np.array(av_entry['tce_plnt_num'])
    ref_kepid=av_entry['kepid']
    ref_kepid=get_strings(ref_kepid)
    ref_label=av_entry['label']
    pl_entry=list(pl_entry)

    np.random.shuffle(pl_entry)

    i=0
    for el in list(pl_entry):
        df=pd.read_csv('new_loc_glob/global/'+el,sep=" ")
        dfl=pd.read_csv('new_loc_glob/local/'+el[0:11]+'_l',sep=" ")
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
        #if(ref_label[loc_f[0]]=='UNK'): continue
        if(av_pl[loc_f[0]]!=1): continue

        flux=df['flux'].iloc[0:2000]
        fluxl=dfl['flux'].iloc[0:200]
        #flux=flux-np.median(flux)
        #flux=flux/np.abs(flux[np.argmin(flux)])
        #fluxl=fluxl-np.median(fluxl)
        #fluxl=fluxl/np.abs(fluxl[np.argmin(fluxl)])

        if(i>=bp):
            if(ref_label[loc_f[0]]=='CONFIRMED'): Y_test.append([1,0])
            #elif(np.any(eb_kepid==el[0:9])):Y_test.append([0,0,1])
            elif(ref_label[loc_f[0]]=='FPS'): Y_test.append([0,1])
            else: continue
            X_test.append(np.array(flux))
            X_testl.append(np.array(fluxl))
            
        else:
            if(ref_label[loc_f[0]]=='CONFIRMED'): Y_train.append([1,0])
            #elif(np.any(eb_kepid==el[0:9])):Y_train.append([0,0,1])
            elif(ref_label[loc_f[0]]=='FPS'): Y_train.append([0,1])
            else: continue
            X_train.append(np.array(flux))
            X_trainl.append(np.array(fluxl))
        i=i+1
        print(i,el[0:11],av_pl[loc_f[0]])
        if(i==tot): break

    arr=np.arange(0,len(X_train),1)
    np.random.shuffle(arr)
    X_train=[X_train[p] for p in arr]
    Y_train=[Y_train[p] for p in arr]
    X_trainl=[X_trainl[p] for p in arr]

    temp1=[]
    temp2=[]
    templ=[]
    filtind=[i for i in range(0,len(Y_train)) if (Y_train[i]==np.array([1,0])).all()]
    filtind2=[i for i in range(0,len(Y_train)) if (Y_train[i]==np.array([0,1])).all()]
    #filtindeb=[i for i in range(0,len(Y_train)) if (Y_train[i]==np.array([0,0,1])).all()]
    print(len(filtind),len(filtind2)) 
    #print(min(len(filtind[0]),len(filtind2[0])))
    
    for i in range(0,min(len(filtind),len(filtind2))):
        temp1.append(X_train[filtind[i]])
        temp2.append(Y_train[filtind[i]])
        templ.append(X_trainl[filtind[i]])
        temp1.append(X_train[filtind2[i]])
        temp2.append(Y_train[filtind2[i]])
        templ.append(X_trainl[filtind2[i]])
        #temp1.append(X_train[filtindeb[i]])
        #temp2.append(Y_train[filtindeb[i]])
        #templ.append(X_trainl[filtindeb[i]])

    print(np.array(temp1).shape,np.array(temp2).shape)

    #print(np.array(X_train).shape,np.array(Y_train).shape)
    #print(np.array(X_test).shape,np.array(Y_test).shape)
    #print(Y_train,len(Y_train))
    #print(Y_test,len(Y_test))
    np.savetxt('training_data/Xtrain_rv_clean_eq.csv', np.array(temp1), delimiter=',')
    np.savetxt('training_data/Ytrain_rv_clean_eq.csv', np.array(temp2), delimiter=',')
    np.savetxt('training_data/Xtrainloc_rv_clean_eq.csv', np.array(templ), delimiter=',')
    #np.savetxt('training_data/Xtest_av.csv', np.array(X_test), delimiter=',')
    #np.savetxt('training_data/Ytest_av.csv', np.array(Y_test), delimiter=',')

#Now begins creation of raw LC. Here we are randomly selecting a fixed number of examples from each data in the input directory. 
#made for a three classifier problem: planets, false positives, eclipsing binaries. Also ensured dtaa is uniformly distributed in TS
def raw_bins(tot,bp,pathin,max_ex,thres):
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

    inc=0
    for el in list(pl_entry):
        df=np.loadtxt(pathin+el)
        try:
            loc=np.where(np.array(ref_kepid)==el[0:9])
            loc_f=[m for m in loc[0] if str(av_pl[m])==el[10]]
        except ValueError as ve:
            continue
        if(len(loc_f)==0): continue
        #the idea behind putting the zero is only coz loc_f is inherently a tuple... the way the alg is it will have only
        #one element tho at all time.
        if(ref_label[loc_f[0]]=='UNK'): continue
        #if(av_pl[loc_f[0]]!=1): continue

        if(len(df.shape)<2):    df=df.reshape(1,len(df))
        chosen_ind=[]
        #else: chosen_ind=np.random.randint(0,len(df),size=5)

        for k in range(0,len(df)):
            med=np.median(df[k])
            std=np.std(df[k])
            cut=int(len(df[k])/3)
            count_tr=[(df[k,int(it):int(it+cut)] < med-thres*std).sum() for it in np.linspace(0,len(df[k])-cut,cut)]
            if(np.any(np.array(count_tr)>7)): chosen_ind.append(k)
            if(len(chosen_ind)==max_ex): break
        #cleaner training samples?
        if(len(chosen_ind)==0): continue

        print(inc,el[0:9],len(chosen_ind))
    
        if(inc>=bp):
            check=np.array([len(df[k]) for k in chosen_ind])
            if(np.any(check<2000)): continue 
            if(ref_label[loc_f[0]]=='CONFIRMED'): [Y_test.append([1,0,0]) for k in chosen_ind]
            elif(np.any(eb_kepid==el[0:9])): [Y_test.append([0,0,1]) for k in chosen_ind]
            elif(ref_label[loc_f[0]]=='FPS'): [Y_test.append([0,1,0]) for k in chosen_ind]
            else: continue
            [ X_test.append(np.array(df[k])) for k in chosen_ind]
            #else: [Y_test.append([0,1,0]) for k in chosen_ind]
        else:
            check=np.array([len(df[k]) for k in chosen_ind])
            if(np.any(check<2000)): continue 
            if(ref_label[loc_f[0]]=='CONFIRMED'): [Y_train.append([1,0,0]) for k in chosen_ind]
            elif(np.any(eb_kepid==el[0:9])): [Y_train.append([0,0,1]) for k in chosen_ind]
            elif(ref_label[loc_f[0]]=='FPS'): [Y_train.append([0,1,0]) for k in chosen_ind]
            else: continue
            [ X_train.append(np.array(df[k])) for k in chosen_ind]
            #else: [Y_train.append([0,1,0]) for k in chosen_ind]
        inc=inc+1
        if(inc==tot): break

    print(np.array(X_train).shape,np.array(Y_train).shape)
    print(np.array(X_test).shape,np.array(Y_test).shape)

    arr=np.arange(0,len(X_train),1)
    np.random.shuffle(arr)
    X_train=[X_train[p] for p in arr]
    Y_train=[Y_train[p] for p in arr]

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
    np.savetxt('training_data/Xtrain_rv_global.csv', np.array(temp1), delimiter=',')
    np.savetxt('training_data/Ytrain_rv_global.csv', np.array(temp2), delimiter=',')
    #np.savetxt('training_data/Xtest_av.csv', np.array(X_test), delimiter=',')
    #np.savetxt('training_data/Ytest_av.csv', np.array(Y_test), delimiter=',')

#best func for raw LC. The random selection has been replaced by a comprehensive search through all the examples, and a cleaning routine 
#added so we can get data of different levels of significance, according to our requirement
def raw_bins_2class(tot,bp,pathin,max_ex,thres):
    X_train=[]
    X_test=[]
    Y_train=[]
    Y_test=[]
    X_extra_tr=[]
    X_extra_ts=[]
    pl_entry=os.listdir(pathin)
    av_entry=ascii.read('autovetter_label.tab')
    extra_info=ascii.read('data_summary_2.dat')
    tperiod=np.array(extra_info['transit_period'])
    tdur=np.array(extra_info['transit_duration'])
    tdepth=np.array(extra_info['transit_depth'])
    tID=extra_info['ID']
    av_pl=np.array(av_entry['tce_plnt_num'])
    ref_kepid=av_entry['kepid']
    ref_kepid=get_strings(ref_kepid)
    ref_label=av_entry['av_training_set']
    pl_entry=list(pl_entry)
    np.random.shuffle(pl_entry)
    i=0
    for el in list(pl_entry):
        df=np.loadtxt(pathin+el)
        try:
            loc=np.where(np.array(ref_kepid)==el[0:9])
            loc_f=[m for m in loc[0] if str(av_pl[m])==el[10]]
            #eloc=np.where(np.array(tID)==el[0:11])
        except ValueError as ve:
            continue
        #if(len(loc_f)==0 or len(eloc[0])==0): continue
        if(len(loc_f)==0): continue
        #the idea behind putting the zero is only coz loc_f is inherently a tuple... the way the alg is it will have only
        #one element tho at all time.
        if(ref_label[loc_f[0]]=='UNK'): continue
        #if(av_pl[loc_f[0]]!=1): continue

        if(len(df.shape)<2):    df=df.reshape(1,len(df))
        chosen_ind=[]
        #else: chosen_ind=np.random.randint(0,len(df),size=5)

        for k in range(0,len(df)):
            #med=np.median(df[k])
            #std=np.std(df[k])
            #cut=int(len(df[k])/3)
            #count_tr=[(df[k,int(it):int(it+cut)] < med-thres*std).sum() for it in np.linspace(0,len(df[k])-cut,cut)]
            #if(np.any(np.array(count_tr)>7)): chosen_ind.append(k)
            chosen_ind.append(k)
            if(len(chosen_ind)==max_ex): break
        #cleaner training samples?
        if(len(chosen_ind)==0): continue

        print(i,el[0:9],len(chosen_ind))
    
        if(i>=bp):
            #check=[len(df[k]) for k in chosen_ind]
            #if(np.any(check<2000)): continue 
            [ X_test.append(np.array(df[k])) for k in chosen_ind]
            #[ X_extra_ts.append([tperiod[eloc[0][0]],tdur[eloc[0][0]],tdepth[eloc[0][0]]]) for k in chosen_ind]
            if(ref_label[loc_f[0]]=='PC'): [Y_test.append([1,0]) for k in chosen_ind]
            else: [Y_test.append([0,1]) for k in chosen_ind]
            #elif(ref_label[loc_f[0]]=='FPS'): [Y_test.append([0,1]) for k in chosen_ind]
            #else: [Y_test.append([0,1,0]) for k in chosen_ind]
        else:
            #check=np.array([len(df[k]) for k in chosen_ind])
            #if(np.any(check<2000)): continue 
            [ X_train.append(np.array(df[k])) for k in chosen_ind]
            #[ X_extra_tr.append([tperiod[eloc[0][0]],tdur[eloc[0][0]],tdepth[eloc[0][0]]]) for k in chosen_ind]
            if(ref_label[loc_f[0]]=='PC'): [Y_train.append([1,0]) for k in chosen_ind]
            else: [Y_train.append([0,1]) for k in chosen_ind]
            #elif(ref_label[loc_f[0]]=='FPS'): [Y_train.append([0,1]) for k in chosen_ind]
            #else: [Y_train.append([0,1,0]) for k in chosen_ind]
        i=i+1
        if(i==tot): break
    print("unique count:",i)
    print(np.array(X_train).shape,np.array(Y_train).shape)
    print(np.array(X_test).shape,np.array(Y_test).shape)
    #print(np.array(X_extra_tr).shape,np.array(X_extra_ts).shape)

    arr=np.arange(0,len(X_train),1)
    np.random.shuffle(arr)
    X_train=[X_train[p] for p in arr]
    Y_train=[Y_train[p] for p in arr]

    
    temp1=[]
    temp2=[]
    temp3=[]
    filtind=[i for i in range(0,len(Y_train)) if (Y_train[i]==np.array([1,0])).all()]
    filtind2=[i for i in range(0,len(Y_train)) if (Y_train[i]==np.array([0,1])).all()]
    print(len(filtind),len(filtind2)) 
    filtind=[i for i in range(0,len(Y_test)) if (Y_test[i]==np.array([1,0])).all()]
    filtind2=[i for i in range(0,len(Y_test)) if (Y_test[i]==np.array([0,1])).all()]
    print(len(filtind),len(filtind2)) 
    '''
    #print(min(len(filtind[0]),len(filtind2[0])))
    for i in range(0,min(len(filtind),len(filtind2))):
        temp1.append(X_train[filtind[i]])
        temp2.append(Y_train[filtind[i]])
        temp3.append(X_extra_tr[filtind[i]])
        temp1.append(X_train[filtind2[i]])
        temp2.append(Y_train[filtind2[i]])
        temp3.append(X_extra_tr[filtind2[i]])
    print(np.array(temp1).shape,np.array(temp2).shape,np.array(temp3).shape)
    #print(Y_train,len(Y_train))
    #print(Y_test,len(Y_test))'''
    np.savetxt('training_data/Xtrain_av_raw100.csv', np.array(X_train), delimiter=',')
    np.savetxt('training_data/Ytrain_av_raw100.csv', np.array(Y_train), delimiter=',')
    #np.savetxt('training_data/Xextra_av_unstitch.csv', np.array(temp3), delimiter=',')
    np.savetxt('training_data/Xtest_av_raw100.csv', np.array(X_test), delimiter=',')
    np.savetxt('training_data/Ytest_av_raw100.csv', np.array(Y_test), delimiter=',')

#not a very useful function... thought of taking a bigger chunked LC and bring down the bins to create a smaller one instead of going through
#the whole preparing the data thing. but it seems like a bad idea
def bin_down_NN(pathinX,pathinY,bs):
    TS=np.loadtxt(pathinX,delimiter=',')
    label=np.loadtxt(pathinY,delimiter=',')
    nTS=[]
    nlabel=[]
    l=len(TS[0])
    print(TS.shape,label.shape)
    for i in range(0,l):
        nel=TS[i]
        nlb_el=label[i]
        nel=nel[int(l/2-bs/2):int(l/2+bs/2)]
        count_tr=(nel < np.median(TS[i])-2*np.std(TS[i])).sum() 
        if(count_tr<5): continue
        nTS.append(nel)
        nlabel.append(nlb_el)
    nTS=np.array(nTS)
    nlabel=np.array(nlabel)
    print(nTS.shape,nlabel.shape)

    temp1=[]
    temp2=[]
    filtind=[i for i in range(0,len(nlabel)) if (nlabel[i]==np.array([1,0])).all()]
    filtind2=[i for i in range(0,len(nlabel)) if (nlabel[i]==np.array([0,1])).all()]
    print(len(filtind),len(filtind2)) 
    #print(min(len(filtind[0]),len(filtind2[0])))
    for i in range(0,min(len(filtind),len(filtind2))):
        temp1.append(nTS[filtind[i]])
        temp2.append(nlabel[filtind[i]])
        temp1.append(nTS[filtind2[i]])
        temp2.append(nlabel[filtind2[i]])
    print(np.array(temp1).shape,np.array(temp2).shape)
    #print(Y_train,len(Y_train))
    #print(Y_test,len(Y_test))
    np.savetxt('training_data/Xtrain_av_raw100_2_2d5.csv', np.array(temp1), delimiter=',')
    np.savetxt('training_data/Ytrain_av_raw100_2_2d5.csv', np.array(temp2), delimiter=',')

        
def raw_bins_2class_avg(tot,bp,pathin,max_ex,thres):
    X_train=[]
    X_test=[]
    Y_train=[]
    Y_test=[]
    X_extra_tr=[]
    X_extra_ts=[]
    pl_entry=os.listdir(pathin)
    av_entry=ascii.read('autovetter_label.tab')
    extra_info=ascii.read('data_summary_2.dat')
    tperiod=np.array(extra_info['transit_period'])
    tdur=np.array(extra_info['transit_duration'])
    tdepth=np.array(extra_info['transit_depth'])
    tID=extra_info['ID']
    av_pl=np.array(av_entry['tce_plnt_num'])
    ref_kepid=av_entry['kepid']
    ref_kepid=get_strings(ref_kepid)
    ref_label=av_entry['av_training_set']
    pl_entry=list(pl_entry)
    #np.random.shuffle(pl_entry)
    i=0
    for el in list(pl_entry):
        df=np.loadtxt(pathin+el)
        try:
            loc=np.where(np.array(ref_kepid)==el[0:9])
            loc_f=[m for m in loc[0] if str(av_pl[m])==el[10]]
            eloc=np.where(np.array(tID)==el[0:11])
        except ValueError as ve:
            continue
        if(len(loc_f)==0 or len(eloc[0])==0): continue
        #the idea behind putting the zero is only coz loc_f is inherently a tuple... the way the alg is it will have only
        #one element tho at all time.
        if(ref_label[loc_f[0]]=='UNK'): continue
        #if(av_pl[loc_f[0]]!=1): continue

        if(len(df.shape)<2):    df=df.reshape(1,len(df))
        chosen_ind=[]
        #else: chosen_ind=np.random.randint(0,len(df),size=5)

        for k in range(0,len(df)):
            med=np.median(df[k])
            std=np.std(df[k])
            cut=int(len(df[k])/3)
            count_tr=[(df[k,int(it):int(it+cut)] < med-thres*std).sum() for it in np.linspace(0,len(df[k])-cut,cut)]
            if(np.any(np.array(count_tr)>7)): chosen_ind.append(k)
            if(len(chosen_ind)==max_ex): break
        #cleaner training samples?
        if(len(chosen_ind)==0): continue

        print(i,el[0:9],len(chosen_ind),tperiod[eloc[0][0]])
    
        if(i>=bp):
            temp=[np.array(df[k]) for k in chosen_ind]
            temp=np.array(np.sum(np.array(temp),axis=0))/3
            if(len(temp)!=2000): continue
            X_test.append(temp)
            X_extra_ts.append([tperiod[eloc[0][0]],tdur[eloc[0][0]],tdepth[eloc[0][0]]])
            if(ref_label[loc_f[0]]=='PC'): Y_test.append([1,0]) 
            else: Y_test.append([0,1]) 
            #elif(ref_label[loc_f[0]]=='FPS'): [Y_test.append([0,1]) for k in chosen_ind]
            #else: [Y_test.append([0,1,0]) for k in chosen_ind]
        else:
            temp=[np.array(df[k]) for k in chosen_ind]
            #print(np.array(temp).shape,np.sum(np.array(temp),axis=0).shape)
            temp=np.array(np.sum(np.array(temp),axis=0))/3
            if(len(temp)!=2000): continue
            X_train.append(temp)
            X_extra_tr.append([tperiod[eloc[0][0]],tdur[eloc[0][0]],tdepth[eloc[0][0]]])
            if(ref_label[loc_f[0]]=='PC'): Y_train.append([1,0])
            else: Y_train.append([0,1]) 
            #elif(ref_label[loc_f[0]]=='FPS'): [Y_train.append([0,1]) for k in chosen_ind]
            #else: [Y_train.append([0,1,0]) for k in chosen_ind]
        i=i+1
        if(i==tot): break
    print("unique count:",i)
    print(np.array(X_train).shape,np.array(Y_train).shape)
    #print(np.array(X_test).shape,np.array(Y_test).shape)
    print(np.array(X_extra_tr).shape,np.array(X_extra_ts).shape)

    arr=np.arange(0,len(X_train),1)
    np.random.shuffle(arr)
    X_train=[X_train[p] for p in arr]
    Y_train=[Y_train[p] for p in arr]

    temp1=[]
    temp2=[]
    temp3=[]
    filtind=[i for i in range(0,len(Y_train)) if (Y_train[i]==np.array([1,0])).all()]
    filtind2=[i for i in range(0,len(Y_train)) if (Y_train[i]==np.array([0,1])).all()]
    print(len(filtind),len(filtind2)) 
    #print(min(len(filtind[0]),len(filtind2[0])))
    for i in range(0,min(len(filtind),len(filtind2))):
        temp1.append(X_train[filtind[i]])
        temp2.append(Y_train[filtind[i]])
        temp3.append(X_extra_tr[filtind[i]])
        temp1.append(X_train[filtind2[i]])
        temp2.append(Y_train[filtind2[i]])
        temp3.append(X_extra_tr[filtind2[i]])
    print(np.array(temp1).shape,np.array(temp2).shape,np.array(temp3).shape)
    #print(Y_train,len(Y_train))
    #print(Y_test,len(Y_test))
    np.savetxt('training_data/Xtrain_av_stitch.csv', np.array(temp1), delimiter=',')
    np.savetxt('training_data/Ytrain_av_stitch.csv', np.array(temp2), delimiter=',')
    np.savetxt('training_data/Xextra_av_stitch.csv', np.array(temp3), delimiter=',')
    #np.savetxt('training_data/Xtest_av.csv', np.array(X_test), delimiter=',')
    #np.savetxt('training_data/Ytest_av.csv', np.array(Y_test), delimiter=',')

def raw_bins_rebinned(tot,pathin):
    X_train=[]
    Y_train=[]
    pl_entry=os.listdir(pathin)
    av_entry=ascii.read('autovetter_label.tab')
    av_pl=np.array(av_entry['tce_plnt_num'])
    ref_kepid=av_entry['kepid']
    ref_kepid=get_strings(ref_kepid)
    ref_label=av_entry['av_training_set']
    i=0
    for el in list(pl_entry):
        df=np.loadtxt(pathin+el)
        try:
            loc=np.where(np.array(ref_kepid)==el[0:9])
            loc_f=[m for m in loc[0] if str(av_pl[m])==el[10]]
        except ValueError as ve:
            continue
        if(len(loc_f)==0): continue
        #the idea behind putting the zero is only coz loc_f is inherently a tuple... the way the alg is it will have only
        #one element tho at all time.
        if(ref_label[loc_f[0]]=='UNK'): continue
    
        X_train.append(df)
        if(ref_label[loc_f[0]]=='PC'): Y_train.append([1,0])
        else: Y_train.append([0,1]) 
        i=i+1
        if(i==tot): break
    print("unique count:",i)
    print(np.array(X_train).shape,np.array(Y_train).shape)

    arr=np.arange(0,len(X_train),1)
    np.random.shuffle(arr)
    X_train=[X_train[p] for p in arr]
    Y_train=[Y_train[p] for p in arr]

    np.savetxt('training_data/Xtrain_av_reb_raw.csv', np.array(X_train), delimiter=',')
    np.savetxt('training_data/Ytrain_av_reb_raw.csv', np.array(Y_train), delimiter=',')


def remove_sig(arr):
    mid=np.median(arr)
    std=np.std(arr)
    count=np.asarray(arr<mid-1.5*std).sum()
    ran = np.random.normal(mid,std,size=count)
    i=0
    ind=np.where(arr<mid-1.5*std)[0]
    for i in range(0,len(ind)): arr[ind[i]]=ran[i]
    return arr


def TS_out_of_new_bins(tot,bp,pathin):
    Xtrain=[]
    Ytrain=[]
    Xtest=[]
    Ytest=[]
    entries=os.listdir(pathin+'xlabel/')
    np.random.shuffle(entries)
    top=20
    count=0
    for el in entries:
        data=np.loadtxt(pathin+'xlabel/'+el)
        labels=np.loadtxt(pathin+'ylabel/'+el)
        #print(data.shape,labels.shape)
        background=np.array([i for i in range(0,len(labels)) if (labels[i]==np.array([0,0,1])).all()])
        foreground=np.setdiff1d(np.arange(0,len(labels)), background)
        minority=min(len(background),len(foreground))

        if(len(foreground)<2): 
            print('no suitable transits:',el[0:9])
            continue

        foreground=np.array(foreground)
        background=np.array(background)
        tracker=0
        if(count<bp):
            for i in range(minority):
                #print(len(data[foreground[i]]),len(data[background[i]]),len(labels[foreground[i]]),len(labels[background[i]]))
                
                Xtrain.append(data[foreground[i]])
                #Xtrain.append(remove_sig(data[foreground[i]]))
                Xtrain.append(data[background[i]])
                Ytrain.append(labels[foreground[i]])
                Ytrain.append(labels[background[i]])
                tracker+=2
                if(i==top): break
            if(minority<len(foreground) and minority<top):
                for i in range(minority,len(foreground)):
                    try: 
                        x=len(data[foreground[i]])
                        y=len(labels[foreground[i]])
                    except: continue
                    Xtrain.append(data[foreground[i]])
                    Ytrain.append(labels[foreground[i]])
                    tracker+=1
                    if(i==top): break

        else:
            for i in range(minority):
                Xtest.append(data[foreground[i]])
                Xtest.append(data[background[i]])
                #Xtest.append(remove_sig(data[foreground[i]]))
                Ytest.append(labels[foreground[i]])
                Ytest.append(labels[background[i]])
                tracker+=2
                if(i==top): break
            if(minority<len(foreground) and minority<top):
                for i in range(minority,len(foreground)):
                    try: 
                        x=len(data[foreground[i]])
                        y=len(labels[foreground[i]])
                    except: continue
                    Xtest.append(data[foreground[i]])
                    Ytest.append(labels[foreground[i]])
                    tracker+=1
                    if(i==top): break
        
        count+=1
        print(count,el[0:9],len(foreground),len(background),tracker)
        if(count==tot): break

    print(np.array(Xtrain).shape, np.array(Ytrain).shape)
    print(np.array(Xtest).shape, np.array(Ytest).shape)

    filtind=[i for i in range(0,len(Ytrain)) if (Ytrain[i]==np.array([1,0,0])).all()]
    filtind2=[i for i in range(0,len(Ytrain)) if (Ytrain[i]==np.array([0,1,0])).all()]
    filtind3=[i for i in range(0,len(Ytrain)) if (Ytrain[i]==np.array([0,0,1])).all()]
    print(len(filtind),len(filtind2),len(filtind3))
    filtind=[i for i in range(0,len(Ytest)) if (Ytest[i]==np.array([1,0,0])).all()]
    filtind2=[i for i in range(0,len(Ytest)) if (Ytest[i]==np.array([0,1,0])).all()]
    filtind3=[i for i in range(0,len(Ytest)) if (Ytest[i]==np.array([0,0,1])).all()]
    print(len(filtind),len(filtind2),len(filtind3))

    np.savetxt('training_data/Xtrain_av_raw500.csv', np.array(Xtrain), delimiter=',')
    np.savetxt('training_data/Ytrain_av_raw500.csv', np.array(Ytrain), delimiter=',')
    np.savetxt('training_data/Xtest_av_raw500.csv', np.array(Xtest), delimiter=',')
    np.savetxt('training_data/Ytest_av_raw500.csv', np.array(Ytest), delimiter=',')

#global_view_nononpl(2100,2000)
#cleaned_data_NN(8000,7500)
#raw_bins(6000,5000,'temp_dir/global/',3,1.2)
TS_out_of_new_bins(6400,5400,'data_red_raw_dirty500/')
#bin_down_NN('training_data/Xtrain_av_raw200_2_2d0.csv','training_data/Ytrain_av_raw200_2_2d0.csv',200)
#raw_bins_rebinned(4000,'raw_rebin2000/')