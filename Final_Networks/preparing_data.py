import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import count
from numpy.core.numeric import count_nonzero
import pandas as pd
import os
from astropy.io import fits

#final version of the bin code... after loads of modifications
#problem? does not work for the false positives, if we at all wanna remove the tranist signature from them... if I figure that our, I'll put it
#in here as a function.
#shallue and vandenberg have normalised their LCs x-axis to 1. Good idea for the neural network, VERY BAD IDEA FOR THE FALSE POSITIVES.
#thats why their false positives detections were so messed up...
#the first function - rebin works for binning data and fps while the second one will remove the transit signature and then rebin.

np.random.seed(12345)

#feel free to change these globlal variables as per requirement.
GLOBAL_VIEW=2000
LOCAL_VIEW=200
FILEPATH_FPS="F:\Masters_Project_Data\\alienworlds_fps\\"
FILEPATH_DATA="F:\Masters_Project_Data\\alienworlds_data\\"

def rebin(x,y,tr_dur,tr_pd):
    #change relevant stuff to days format from hours format
    tr_dur=tr_dur/24
    tempx=[]
    tempy=[]
    #remove Nan Values
    for i in range(0,len(y)):
        if(not np.isnan(y[i])):
            tempx.append(x[i])
            tempy.append(y[i])
    x=tempx
    y=tempy
    df = pd.DataFrame(list(zip(x, y)),columns =['phase', 'flux'])
    
    #create bins needed according to local and global view
    low=x[np.argmin(x)]
    high=x[np.argmax(x)]
    bins=np.linspace(low,high,GLOBAL_VIEW)
    bins_lc=np.linspace(-tr_dur,tr_dur,LOCAL_VIEW+1)

    #median out the contents of a group
    groups = df.groupby(np.digitize(df['phase'], bins))
    df_gl=groups.median()
    
    tot=pd.Series(np.arange(0,GLOBAL_VIEW))
    left=tot.index.difference(df_gl.index)
  
    #this is to fill up empty bins with values via interpolation.
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

    #filter out the local view
    df_lc=df[(df["phase"] > -tr_dur) & (df["phase"] < tr_dur)]
    lc_groups = df_lc.groupby(np.digitize(df_lc['phase'], bins_lc))
    df_lc_f=lc_groups.median()

    tot=pd.Series(np.arange(0,LOCAL_VIEW))
    left=tot.index.difference(df_lc_f.index)

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

    return (df_lc_f,df_gl)

#define here a function for getting redidual LC...we'll take two approaches, one where we eliminate the transit from the final header file,
#and the second where we process the statistic of the redidue... check if thats better. Eliminate transit works well but fails to bin fps, 
#and i CANNOT FIGURE OUT FOR THE LIFE OF ME HOW THESE PEOPLE DETRENDED THE LC...
def remove_rebin(x,y,tr_dur,tr_pd):
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
    count=0
    for i in range(0,len(df)):
        if (df['phase'].iloc[i]>-tr_dur*0.75 and df['phase'].iloc[i]<tr_dur*0.75):
            df['flux'].iloc[i]=np.NaN
            count+=1
    clean=np.array([val for val in df['flux'] if not np.isnan(val)])
    mean=np.mean(clean)
    sigma=np.std(clean)
    #print(mean,sigma,count)

    noise=np.random.normal(mean,sigma,size=count)

    j=0
    for i in range(1,len(df)):
        if (np.isnan(df['flux'].iloc[i])):
            df['flux'].iloc[i]=noise[j]
            j=j+1            
    
    bins=np.linspace(low,high,GLOBAL_VIEW)

    groups = df.groupby(np.digitize(df['phase'], bins))
    df_gl=groups.median()
    
    tot=pd.Series(np.arange(0,GLOBAL_VIEW))
    left=tot.index.difference(df_gl.index)
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
    df_lc=df_gl.iloc[int(GLOBAL_VIEW/2-100):int(GLOBAL_VIEW/2+100)]
    return(df_lc,df_gl)


#adding somewhat of a fix for the false positive scenario... no idea how well it works, but it might... also worried how computationally 
#expensive it may turn out to be.
def remove_rebin_fps(phase,flux,tdur,tperiod):
    tempx=[]
    tempy=[]
    count=0
    for i in range(0,len(flux)):
        if(not np.isnan(flux[i])):
            tempx.append(phase[i])
            tempy.append(flux[i])
        else: count+=1
    tempx=np.array(tempx)
    tempy=np.array(tempy)
    low=tempx[np.argmin(tempx)]
    high=tempx[np.argmax(tempx)]
    medval=np.median(tempy)
    sigma=np.std(tempy)
    thres=medval-3*sigma

    r_phase=[]
    r_flux=[]
    for i in range(0,len(tempy)):
        if(flux[i]<thres):
            r_phase.append(tempx[i])
            r_flux.append(tempy[i])

    df=pd.DataFrame(list(zip(r_phase, r_flux)),columns =['phase', 'flux'])
    binsize=tperiod*24/tdur
    bins=np.linspace(low,high,int(binsize))
    groups = df.groupby(np.digitize(df['phase'], bins))
    groupdat=groups.agg(['min', 'count'],axis=1)
    removestuff=[]
    for i in range(0,len(groupdat)):
        if(groupdat['flux','count'].iloc[i]>5):
            temp=groupdat['flux','min'].iloc[i]
            index=np.where(tempy==temp)
            removestuff.append(phase[index[0][0]])

    removestuff.append(0)
    for i in range(0,len(flux)):
        for el in removestuff:
            if (phase[i]>el-tdur*0.3 and phase[i]<el+tdur*0.3):
                flux[i]=np.NaN
                count+=1
                break
        
    clean=np.array([val for val in flux if not np.isnan(val)])
    if(len(clean)==0): return(pd.DataFrame(columns =['phase', 'flux']),pd.DataFrame(columns =['phase', 'flux']))
    mean=np.mean(clean)
    sigma=np.std(clean)
    noise=np.random.normal(mean,sigma,size=count+10)
    j=0
    for i in range(0,len(flux)):
        if (np.isnan(flux[i])):
            flux[i]=noise[j]
            j=j+1

    bins=np.linspace(low,high,GLOBAL_VIEW)

    df_n=pd.DataFrame(list(zip(phase, flux)),columns =['phase', 'flux'])
    groups = df_n.groupby(np.digitize(df_n['phase'], bins))
    df_gl=groups.median()
    
    tot=pd.Series(np.arange(0,GLOBAL_VIEW))
    left=tot.index.difference(df_gl.index)
    for el in left:
        if (el==0 or el==GLOBAL_VIEW): continue
        i=1
        while el-i in left or el+i in left:
            if (el-i==0 or el+i==GLOBAL_VIEW): break
            i=i+1
        if (el-i==0 or el+i==GLOBAL_VIEW): continue
        df_gl.loc[el]=[(df_gl.loc[el-i]['phase']+df_gl.loc[el+i]['phase'])/2,(df_gl.loc[el-i]['flux']+df_gl.loc[el+i]['flux'])/2]
    df_gl=df_gl.sort_index(axis=0)
    df_gl['phase']=df_gl['phase']/tperiod
    df_lc=df_gl.iloc[int(GLOBAL_VIEW/2-100):int(GLOBAL_VIEW/2+100)]
    return(df_lc,df_gl)
    
#simply slice bins into a fixed binsize and obtain the cut LC
def get_transits_from_raw(binsize,pathin,pathout):
    dataset=os.scandir(pathin)
    entries=list(dataset)
    np.random.shuffle(entries)
    x=0
    for el in entries:
        hdu = fits.open(pathin+el.name)
        flux=hdu[1].data['LC_DETREND']
        x=x+1
        if(x==2000): break
        cuts=int(len(flux)/binsize)
        lightcurve=[]
        for val in range(0,cuts-1):
            red_flux=flux[val*binsize:binsize*(val+1)]
            count_nan=np.isnan(red_flux).sum()
            if(count_nan/binsize > 0.2): continue

            for i in range(0,len(red_flux)):
                if(np.isnan(red_flux[i])):
                    t=1
                    try:
                        while(np.isnan(red_flux[i-t]) or np.isnan(red_flux[i+t])):    
                            if(i-t <0):
                                red_flux[i]=red_flux[i+t]
                                break
                            if(i+t > binsize): 
                                red_flux[i]=red_flux[i-t]
                                break
                            t+=1
                        red_flux[i]=(red_flux[i-t]+red_flux[i+t])/2
                    except:
                        red_flux[i]=0
            
            med=np.median(red_flux)
            std=np.std(red_flux)
            count_tr=(red_flux < med-2*std).sum()
            if(count_tr>5): lightcurve.append(red_flux)
        if(len(lightcurve)==0): continue  
        print(x,np.array(lightcurve).shape,el.name[4:13])
        np.savetxt(pathout+el.name[4:13]+'.dat',lightcurve,delimiter=' ')

#make cuts about phase zero and get 200... or whatever points around it.
def get_transits_from_raw_v2(binsize,pathin,pathout):
    dataset=os.scandir(pathin)
    entries=list(dataset)
    np.random.shuffle(entries)
    x=0
    for el in entries:
        hdu = fits.open(pathin+el.name)
        flux=hdu[1].data['LC_WHITE']
        phase=hdu[1].data['PHASE']
        #if(x==10): break
        ind_arr=[i for i in range(0,len(phase)-1) if phase[i]*phase[i-1]<0]
        lightcurve=[]
        for ind in ind_arr:
            red_flux=flux[ind-int(binsize/2):ind+int(binsize/2)]
            red_flux=np.array(red_flux)
            count_nan=np.isnan(red_flux).sum()
            if(count_nan/binsize > 0.2): continue

            for i in range(0,len(red_flux)):
                if(np.isnan(red_flux[i])):
                    t=1
                    try:
                        while(np.isnan(red_flux[i-t]) or np.isnan(red_flux[i+t])):    
                            if(i-t <0):
                                red_flux[i]=red_flux[i+t]
                                break
                            if(i+t > binsize): 
                                red_flux[i]=red_flux[i-t]
                                break
                            t+=1
                        red_flux[i]=(red_flux[i-t]+red_flux[i+t])/2
                    except:
                        red_flux[i]=0
            
            if(len(red_flux)==binsize): 
                med=np.median(red_flux)
                std=np.std(red_flux)
                count_tr=(red_flux < med-2*std).sum()
                if(count_tr>7): lightcurve.append(red_flux)
    
        if(len(lightcurve)==0): 
            print("miss:",el.name[4:13])
            continue  
        print("hit:",x,np.array(lightcurve).shape,el.name[4:13])
        x=x+1 
        np.savetxt(pathout+el.name[4:13]+'.dat',lightcurve,delimiter=' ')

#a more robust means to get data... right now the problem is incomplete light curves ... how do you expect to get data properly if you dont 
#have the appropriate bins to work with ? Current perspective ... just grab a bunch of transits and detect FPS from there... but the
#problem may not be that simple
def get_transits_from_raw_v3(binsize,pathin,pathout):
    dataset=os.scandir(pathin)
    entries=list(dataset)
    np.random.shuffle(entries)
    x=0
    for el in entries:
        hdu = fits.open(pathin+el.name)
        try:
            impact=np.argmax(np.array([hdu[i].header['TDEPTH'] for i in range(1,len(hdu)-1) if hdu[i].header['TDEPTH']>0]))
        except:
            #impact=1
            print("miss 1:",el.name[4:13])
            continue
        flux=hdu[impact+1].data['LC_DETREND']
        phase=hdu[impact+1].data['PHASE']
        #if(x==25): break
        ind_arr=[i for i in range(0,len(phase)-1) if (phase[i]*phase[i-1]<0 and np.abs(phase[i]*phase[i-1])<0.0001)]
        lightcurve=[]
        for ind in ind_arr:
            red_flux=flux[ind-int(binsize/2):ind+int(binsize/2)]
            red_flux=np.array(red_flux)
            count_nan=np.isnan(red_flux).sum()
            if(count_nan/binsize > 0.2): continue

            for i in range(0,len(red_flux)):
                if(np.isnan(red_flux[i])):
                    t=1
                    try:
                        while(np.isnan(red_flux[i-t]) or np.isnan(red_flux[i+t])):    
                            if(i-t <0):
                                red_flux[i]=red_flux[i+t]
                                break
                            if(i+t > binsize): 
                                red_flux[i]=red_flux[i-t]
                                break
                            t+=1
                        red_flux[i]=(red_flux[i-t]+red_flux[i+t])/2
                    except:
                        red_flux[i]=0
            
            if(len(red_flux)==binsize): 
                med=np.median(red_flux)
                std=np.std(red_flux)
                cut=int(binsize/9)
                count_tr=[(red_flux[int(k):int(k+cut)] < med-1.5*std).sum() for k in np.linspace(0,binsize-cut,cut)]
                if(np.any(np.array(count_tr)>7)): lightcurve.append(red_flux)
            
            if(len(lightcurve)==20): break
    
        if(len(lightcurve)==0): 
            print("miss:",el.name[4:13])
            continue  
        print("hit:",x,np.array(lightcurve).shape,el.name[4:13],impact+1,hdu[impact+1].header['TDEPTH'])
        x=x+1 
        np.savetxt(pathout+el.name[4:13]+'_'+str(impact+1)+'.dat',lightcurve,delimiter=' ')


#use this to obtain three consecutive transit events from a target file... dont know how effective it is tho..
def get_threesome_raw(pathin,pathout):
    dataset=os.scandir(pathin)
    entries=list(dataset)
    np.random.shuffle(entries)
    x=0
    for el in entries:
        hdu = fits.open(pathin+el.name)
        flux=hdu[1].data['LC_DETREND']
        phase=hdu[1].data['PHASE']
        trdur=hdu[1].header['TDUR']
        tpd=hdu[1].header['TPERIOD']
        npixdur=int(trdur*2)
        npix=int(tpd*24)
        if(npix<1):
            print("miss 1:",el.name[4:13])
            continue

        clean=np.array([val for val in flux if not np.isnan(val)])
        med=np.median(clean)
        std=np.std(clean)
        #if(x==10): break
        ind_arr=[i for i in range(0,len(phase)-1) if phase[i]*phase[i-1]<0]

        val1=(flux[int(ind_arr[0]-npixdur*6):int(ind_arr[0]+npixdur*6)]<med-2*std).sum()
        val2=(flux[int(ind_arr[1]-npixdur*6):int(ind_arr[1]+npixdur*6)]<med-2*std).sum()
        if(val1>val2): flag=0
        else: flag=1
        cut=min(npixdur*6,npix*0.75)
        lightcurve=[]
        for i in range(flag,len(ind_arr),2):
            
            red_flux=flux[int(ind_arr[i]-cut):int(ind_arr[i]+cut)]
            red_flux=np.array(red_flux)
            count_nan=np.isnan(red_flux).sum()
            if(len(red_flux)==0):
                lightcurve=[]
                continue
            if(count_nan/len(red_flux) > 0.2): 
                lightcurve=[]
                continue

            for i in range(0,len(red_flux)):
                if(np.isnan(red_flux[i])):
                    t=1
                    try:
                        while(np.isnan(red_flux[i-t]) or np.isnan(red_flux[i+t])):    
                            if(i-t <0):
                                red_flux[i]=red_flux[i+t]
                                break
                            if(i+t > npix*2): 
                                red_flux[i]=red_flux[i-t]
                                break
                            t+=1
                        red_flux[i]=(red_flux[i-t]+red_flux[i+t])/2
                    except:
                        red_flux[i]=0
            
                #count_tr=(red_flux[int(mp-npixdur*4):int(mp+npixdur*4)] < med-std).sum()
            if(red_flux[np.argmin(red_flux)]<med-2*std): lightcurve.append(red_flux)
                #if(count_tr>npixdur): lightcurve.append(red_flux)
            else: lightcurve=[]
            
            if(len(lightcurve)==3): break
    
        if(len(lightcurve)<3): 
            print("miss 2:",el.name[4:13])
            continue  
        print("hit:",x,np.array(lightcurve).shape,el.name[4:13])
        x=x+1 
        try:
            np.savetxt(pathout+el.name[4:13]+'.dat',lightcurve,delimiter=' ')
        except:
            continue
           
#handy functionality to loop over whatever functions are needed to loop over.
def extract(func,pathin,pathout,size):
    dataset=os.scandir(pathin)
    entries=list(dataset)
    np.random.shuffle(entries)
    x=0
    for el in entries:
        hdu = fits.open(pathin+el.name)
        n=len(hdu)
        x=x+1
        if(x==size): break
        for i in range(1,n-1):
            if(hdu[i].header['TDUR']==None or hdu[i].header['TPERIOD']==None): continue
            phase=hdu[i].data['PHASE']
            period=hdu[i].header['TPERIOD']
            flux=hdu[i].data['LC_DETREND'] 
            df_lc,df_gl=func(phase,flux,hdu[i].header['TDUR'],period)
            df_lc.to_csv(pathout+'/local/'+el.name[4:13]+'_'+str(i)+'_l',sep=' ',index=False)
            df_gl.to_csv(pathout+'/global/'+el.name[4:13]+'_'+str(i)+'_g',sep=' ',index=False)
            print(x,len(df_gl),len(df_lc),el.name[4:13])



#here just call out the extract function with whatever values are needed...
#extract(remove_rebin,FILEPATH_DATA,'nonpl_red',13)
#extract(rebin,FILEPATH_FPS,'temp_dir',4000)
#get_threesome_raw(FILEPATH_FPS,"data_prelim_stitch/") 
get_transits_from_raw_v3(500,FILEPATH_DATA,'data_red_raw_dirty500/')
   
