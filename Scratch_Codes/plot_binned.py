#"a spectrum is equal to a thousand pictures"... does this apply to lightcurves too?
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
import pandas as pd
import os

#from tensorflow.python.ops.gen_array_ops import size_eager_fallback

#I'm whimsical and i know it...
color_arr=['#5b3b8c','#4B9CD3','#800020','#df6985','#ffa600','#00a86b','grey','red','#003153','#d1e231']
col_arr2=['#6F00FF','#5D76A9','#002244','#6F00FF','#5D76A9','#002244','#6F00FF','#5D76A9','#002244']

#here we're plotting sample light curves from the binned data that we made...
FILEPATH='E:\\Masters_Project_Data\\new_loc_glob\\global\\'

#add a directory, and get a fixed number of plots of randomly chosen samples from that
def plot_func(path,title,size_arr):
    entries=os.scandir(path)
    entries=list(entries)
    np.random.shuffle(entries)
    i=0
    j=0
    tabs=0
    plt.style.use('seaborn-bright')
    #remove the sharey if needed... the representations can be weird sometimes.
    fig,ax=plt.subplots(int(np.sqrt(size_arr)),int(np.sqrt(size_arr)),figsize=(25,10))

    plt.suptitle(title,size=20)
    for el in entries:
        tabs+=1
        df=pd.read_csv(path+el.name, sep=" ")
        ax[i][j].set_title(el.name[0:11],size=9)
        med=np.median(df['flux'])
        std=np.std(df['flux'])
        ax[i][j].plot(df['phase'],df['flux'],marker='.',ls='None',color=color_arr[i-j])
        #ax[i][j].plot(df['phase'],(med-2.5*std)*np.ones(len(df['flux'])),color='black')
        #ax[i][j].plot(df['phase'],(med+2.5*std)*np.ones(len(df['flux'])),color='black')


        if(j==0): ax[i][j].set_ylabel('flux',size=15)
        if(i==int(np.sqrt(size_arr))-1): ax[i][j].set_xlabel('phase',size=15)
        i=i+1
        if(i==int(np.sqrt(size_arr))):
            i=0
            j+=1
        if(tabs==size_arr): break

def plot_simul_ts(pathx,pathxlox,pathy,title,size_arr):
    df=np.loadtxt(pathx,delimiter=',')
    df2=np.loadtxt(pathxlox,delimiter=',')
    df3=np.loadtxt(pathy,delimiter=',')
    i=0
    tabs=0
    plt.style.use('seaborn-bright')
    #remove the sharey if needed... the representations can be weird sometimes.
    fig,ax=plt.subplots(int(size_arr),2,figsize=(15,10))

    plt.suptitle(title,size=15)
    for el in range(0,len(df)):
        tabs+=1
        if(df3[el][0]==1):
             ylab='planet'
        else: 
            ylab='fps'
            continue
            
        #print(len(df[el]))
        ax[i][0].plot(np.linspace(-0.25,0.75,len(df[el]),endpoint=True),df[el],marker='.',ls='None',color=col_arr2[2],label=ylab)
        ax[i][1].plot(np.linspace(-1,1,len(df2[el]),endpoint=True),df2[el],marker='.',ls='None',color=col_arr2[1])
        ax[i][0].legend()
        ax[i][0].set_ylabel('flux',size=13)
        if(i==size_arr-1): 
            ax[i][0].set_xlabel('phase',size=13)
            ax[i][1].set_xlabel('phase',size=13)
        i=i+1
        if(i==size_arr): break


#routine for plotting the removed tranist LC over original value
def plot_func_supimposed(path_m,path_r,title,size_arr):
    entries=os.scandir(path_m)
    entries=list(entries)[50:80]
    #np.random.shuffle(entries)
    i=0
    j=0
    tabs=0
    plt.style.use('seaborn-bright')
    #remove the sharey if needed... the representations can be weird sometimes.
    fig,ax=plt.subplots(5,5,figsize=(25,10),sharex=True)

    plt.suptitle(title,size=20)
    for el in entries:
       
        df=pd.read_csv(path_m+el.name, sep=" ")
        try:
            df2=pd.read_csv(path_r+el.name[:11]+'_l',sep=" ")
        except:
            continue

        tabs+=1
        ax[i][j].set_title(el.name[0:9],size=9)
        ax[i][j].plot(df2['phase'],df2['flux'],marker='.',ls='None',color='black')
        ax[i][j].plot(df['phase'],df['flux'],marker='.',ls='None',color=color_arr[i])
        
        if(j==0): ax[i][j].set_ylabel('flux',size=15)
        if(i==int(np.sqrt(size_arr))-1): ax[i][j].set_xlabel('phase',size=15)
        i=i+1
        if(i==int(np.sqrt(size_arr))):
            i=0
            j+=1
        if(tabs==size_arr): break

#this is the plotting for the raw directory, here each file has a number of chunks. so we pick some no. of files, and this chunks 
#4 exaples out of each and plots it
def plot_func_raw(path,title,size_arr):
    entries=os.listdir(path)
    np.random.shuffle(entries)
    i=0
    j=0
    tabs=0
    plt.style.use('seaborn-bright')
    #remove the sharey if needed... the representations can be weird sometimes.
    fig,ax=plt.subplots(3,size_arr,figsize=(25,10))
    #ind=[2,10,20,100,120]
    #entries=[entries[m] for m in ind]
    plt.suptitle(title,size=20)
    for el in entries:
        tabs+=1
        data=np.loadtxt(path+el)
        arr=np.arange(0,len(data),1)
        np.random.shuffle(arr)
        data=[data[z] for z in arr]
        if(len(np.array(data).shape) <2): continue
        if(len(data)<3): continue
        
        for i in range(0,1):
            med=np.median(data[i])
            std=np.std(data[i])
            if(i==0): ax[i][j].set_title(el[0:9],size=9)
            ax[i][j].plot(data[i],marker='.',ls='None',color=color_arr[i-j])
            ax[i][j].plot((med-1.5*std)*np.ones(len(data[i])),color='black')
            if(j==0): ax[i][j].set_ylabel('flux',size=15)
            if(i==size_arr-1): ax[i][j].set_xlabel('phase',size=15)
    
        j=j+1
        if(j==size_arr): break

#here we can input a training sample, instead of a directory, and create plots with that
def plot_ts(path,labpath,title,size_arr):
    TS=np.loadtxt(path,delimiter=',')
    lab=np.loadtxt(labpath,delimiter=',')
    arr=np.arange(0,len(TS),1)
    np.random.shuffle(arr)
    TS=[TS[i] for i in arr]
    lab=[lab[i] for i in arr]
    i=0
    j=0
    tabs=0
    plt.style.use('seaborn-bright')
    #remove the sharey if needed... the representations can be weird sometimes.
    szn=int(np.sqrt(size_arr))
    fig,ax=plt.subplots(szn,szn,figsize=(12,8),sharex=True)
    #ind=[2,10,20,100,120]
    #entries=[entries[m] for m in ind]
    plt.suptitle(title,size=20)
    for el in range(0,len(TS)):
        tabs+=1
        med=np.median(TS[el])
        #min=el[np.argmin(TS[el])]
        std=np.std(TS[el])
        #el=(TS[el]-med)/(med-min)
        #clean=[x for x in el if x > med-std]
        #wt=np.std(clean)
        ax[i][j].plot((med-0*std)*np.ones(len(TS[el])),color='black')
        ax[i][j].plot(TS[el],color=col_arr2[i-j],marker='.',ls='none',label=lab[el])
        if(j==0): ax[i][j].set_ylabel('flux',size=15)
        ax[i][j].legend()
        if(i==szn-1): ax[i][j].set_xlabel('phase',size=15)
        i=i+1
        if(i==szn):
            i=0
            j+=1
        if(tabs==size_arr): break

def plot_func_raw_v2(path,title,size_arr):
    entries=os.listdir(path)
    np.random.shuffle(entries)
    i=0
    j=0
    tabs=0
    plt.style.use('seaborn-bright')
    #remove the sharey if needed... the representations can be weird sometimes.
    fig,ax=plt.subplots(size_arr,size_arr,figsize=(25,10))
    #ind=[2,10,20,100,120]
    #entries=[entries[m] for m in ind]
    plt.suptitle(title,size=20)
    for el in entries:
        tabs+=1
        data=np.loadtxt(path+el)
        
        ax[i][j].set_title(el[0:9],size=9)
        ax[i][j].plot(data,marker='.',ls='None',color=color_arr[i-j])
        if(j==0): ax[i][j].set_ylabel('flux',size=15)
        if(i==size_arr-1): ax[i][j].set_xlabel('phase',size=15)

        i=i+1
        if(i==size_arr):
            i=0
            j=j+1
        if(j==size_arr): break

#plot_func_raw_v2('raw_rebin2000/','Example lightcurves(Before stitch)\n200 pixels',4)
#plt.savefig('plots/LC_stitch_200pix.png')
#plot_func(FILEPATH,'Example lightcurves',9)
plot_ts('training_data/Xtrain_av_raw200_2_2d0_v3.csv','training_data/Ytrain_av_raw200_2_2d0_v3.csv','Raw LC',9)
#plt.savefig('plots/LC_interp_800.png')
#plt.savefig('present_samples')
#plot_simul_ts('training_data/Xtrain_rv_clean.csv','training_data/Xtrainloc_rv_clean.csv','training_data/Ytrain_rv_clean.csv',
#   'Sample Lightcurve (Planets)',5)

plt.savefig('present_sample_planets_dirt')


plt.show()
