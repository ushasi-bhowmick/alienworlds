import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from astropy.io import fits

#we'll take one TCE per data set right now, but some generalisation is pending... or else stuff can get pretty bizarre
#here we try to rebin the data... set up a fixed number of input nodes maybe? Lets take a ballpark number to start with,
#say 1000 in global view 100 in local view?
#empty bins is a major problem
#shallue and vandenberg have normalised their LCs x-axis to 1. Good idea for the neural network, VERY BAD IDEA FOR THE FALSE POSITIVES.
#thats why their false positives detections were so messed up...

GLOBAL_VIEW=2000
LOCAL_VIEW=200
def rebin(x,y,tr_dur,tr_pd):
    tr_dur=tr_dur/24
    tempx=[]
    tempy=[]
    for i in range(0,len(y)):
        if(not np.isnan(y[i])):
            tempx.append(x[i])
            tempy.append(y[i])
    x=tempx
    y=tempy
    df = pd.DataFrame(list(zip(x, y)),columns =['phase', 'flux'])
    
    low=x[np.argmin(x)]
    high=x[np.argmax(x)]
    bins=np.linspace(low,high,GLOBAL_VIEW)
    bins_lc=np.linspace(-tr_dur,tr_dur,LOCAL_VIEW+1)

    groups = df.groupby(np.digitize(df['phase'], bins))
    df_gl=groups.mean()
    
    tot=pd.Series(np.arange(0,GLOBAL_VIEW))
    left=tot.index.difference(df_gl.index)
    #print(left)
    for el in left:
        if (el==0 or el==GLOBAL_VIEW): continue
        i=1
        while el-i in left or el+i in left:
            if (el-i==0 or el+i==GLOBAL_VIEW): break
            i=i+1
        if (el-i==0 or el+i==GLOBAL_VIEW): continue
        df_gl.loc[el]=[(df_gl.loc[el-i]['phase']+df_gl.loc[el+i]['phase'])/2,(df_gl.loc[el-i]['flux']+df_gl.loc[el+i]['flux'])/2]
    df_gl=df_gl.sort_index(axis=0)
    df_gl['phase']=df_gl['phase']/tr_pd

    df_lc=df[(df["phase"] > -tr_dur) & (df["phase"] < tr_dur)]
    lc_groups = df_lc.groupby(np.digitize(df_lc['phase'], bins_lc))
    df_lc_f=lc_groups.mean()

    tot=pd.Series(np.arange(0,LOCAL_VIEW))
    left=tot.index.difference(df_lc_f.index)
    #print(left)
    for el in left:
        if (el==0 or el==LOCAL_VIEW): continue
        i=1
        while el-i in left or el+i in left:
            if (el-i==0 or el+i==LOCAL_VIEW): break
            i=i+1
        if (el-i==0 or el+i==LOCAL_VIEW): continue
        df_lc_f.loc[el]=[(df_lc_f.loc[el-i]['phase']+df_lc_f.loc[el+i]['phase'])/2,(df_lc_f.loc[el-i]['flux']+df_lc_f.loc[el+i]['flux'])/2]
    df_lc_f=df_lc_f.sort_index(axis=0)
    df_lc_f['phase']=df_lc_f['phase']/tr_dur

    #print(len(df_lc_f),len(df_gl))
    #plt.plot(df_gl['phase'],df_gl['flux'],marker='.',ls='none')
    #plt.plot(df_lc_f['phase'],df_lc_f['flux'],marker='.',ls='none')
    #plt.xlim(-1,1)
    return (df_lc_f,df_gl)


#rebin(phase2,flux2,hdu[4].header['TDUR'],hdu[4].header['TPERIOD'])
#rebin(phase1,flux1,hdu[1].header['TDUR'],hdu[1].header['TPERIOD'])


#in this section we run the loop for bringing sample out of database and write it into a matrix
#this matrix is gonna be our training sample
FILEPATH="F:\Masters Project Data\\alienworlds_fps\\"
entries=os.scandir("F:\Masters Project Data\\alienworlds_fps\\")
x=0
for el in entries:
    hdu = fits.open(FILEPATH+el.name)
    n=len(hdu)
    x=x+1
    if(x==20): break
    for i in range(1,n-1):
        if(hdu[i].header['TDUR']==None or hdu[i].header['TPERIOD']==None): continue
        phase=hdu[i].data['PHASE']
        period=hdu[i].header['TPERIOD']
        flux=hdu[i].data['LC_DETREND']
        df_lc,df_gl=rebin(phase,flux,hdu[i].header['TDUR'],period)
        df_lc.to_csv('fps_red/local/'+el.name[4:13]+'_'+str(i)+'_l',sep=' ',index=False)
        df_gl.to_csv('fps_red/global/'+el.name[4:13]+'_'+str(i)+'_g',sep=' ',index=False)
        print(x,len(df_gl),len(df_lc),el.name[4:13])
        #print(hdu[1].header['TDUR'],hdu[1].header['TDEPTH'],hdu[1].header['TPERIOD'])
        #print(hdu[2].header['TDUR'],hdu[2].header['TDEPTH'],hdu[1].header['TPERIOD'])