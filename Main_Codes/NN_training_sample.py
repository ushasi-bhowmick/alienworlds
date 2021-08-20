import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

np.random.seed(122334)

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

global_view_nononpl(2100,2000)