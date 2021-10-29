#"a spectrum is equal to a thousand pictures"... does this apply to lightcurves too?
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
import pandas as pd
import os

#from tensorflow.python.ops.gen_array_ops import size_eager_fallback

#I'm whimsical and i know it...
color_arr=['#5b3b8c','#4B9CD3','#800020','#df6985','#ffa600','#00a86b','grey','red','#003153','#d1e231']

#here we're plotting sample light curves from the binned data that we made...
FILEPATH='data_red/global/'

#add a directory, and get a fixed number of plots of randomly chosen samples from that
def plot_func(path,title,size_arr):
    entries=os.scandir(path)
    entries=list(entries)
    #np.random.shuffle(entries)
    i=0
    j=0
    tabs=0
    plt.style.use('seaborn-bright')
    #remove the sharey if needed... the representations can be weird sometimes.
    fig,ax=plt.subplots(int(np.sqrt(size_arr)),int(np.sqrt(size_arr)),figsize=(25,10),sharex=True)

    plt.suptitle(title,size=20)
    for el in entries:
        tabs+=1
        df=pd.read_csv(path+el.name, sep=" ")
        ax[i][j].set_title(el.name[0:11],size=9)
        ax[i][j].plot(df['time'],df['flux'],marker='.',ls='None',color=color_arr[i-j])

        if(j==0): ax[i][j].set_ylabel('flux',size=15)
        if(i==int(np.sqrt(size_arr))-1): ax[i][j].set_xlabel('phase',size=15)
        i=i+1
        if(i==int(np.sqrt(size_arr))):
            i=0
            j+=1
        if(tabs==size_arr): break

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
        
        for i in range(0,3):
            med=np.median(data[i])
            std=np.std(data[i])
            if(i==0): ax[i][j].set_title(el[0:9],size=9)
            ax[i][j].plot(data[i],marker='.',ls='None',color=color_arr[i-j])
            ax[i][j].plot((med-std)*np.ones(len(data[i])),color='black')
            if(j==0): ax[i][j].set_ylabel('flux',size=15)
            if(i==size_arr-1): ax[i][j].set_xlabel('phase',size=15)
    
        j=j+1
        if(j==size_arr): break

#here we can input a training sample, instead of a directory, and create plots with that
def plot_ts(path,title,size_arr):
    TS=np.loadtxt(path,delimiter=',')
    np.random.shuffle(TS)
    i=0
    j=0
    tabs=0
    plt.style.use('seaborn-bright')
    #remove the sharey if needed... the representations can be weird sometimes.
    szn=int(np.sqrt(size_arr))
    fig,ax=plt.subplots(szn,szn,figsize=(25,10),sharex=True)
    #ind=[2,10,20,100,120]
    #entries=[entries[m] for m in ind]
    plt.suptitle(title,size=20)
    for el in TS:
        tabs+=1
        med=np.median(el)
        std=np.std(el)
        ax[i][j].plot((med-2*std)*np.ones(len(el)),color='black')
        ax[i][j].plot(el,marker='.',ls='None',color=color_arr[i-j])
        if(j==0): ax[i][j].set_ylabel('flux',size=15)
        if(i==szn-1): ax[i][j].set_xlabel('phase',size=15)
        i=i+1
        if(i==szn):
            i=0
            j+=1
        if(tabs==size_arr): break



plot_func_raw('data_prelim_stitch/','Example lightcurves(Raw)\n200 pixels',5)
#plt.savefig('plots/Cleaned_LC_200pix.png')

#plot_ts('training_data/Xtrain_av_raw500_2_2d0_v2.csv','testing',25)

plt.show()
