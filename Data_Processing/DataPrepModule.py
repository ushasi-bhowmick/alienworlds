import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import count
from numpy.core.numeric import count_nonzero
from numpy.lib.function_base import median
import pandas as pd
import os
from scipy.interpolate import interp1d
from astropy.io import fits,ascii

#final version of the bin code... after loads of modification
#problem? does not work for the false positives, if we at all wanna remove the transit signature from them... if I figure that our, I'll put it
#in here as a function.
#shallue and vandenberg have normalised their LCs x-axis to 1. Good idea for the neural network, VERY BAD IDEA FOR THE FALSE POSITIVES.
#thats why their false positives detections were so messed up...
#the first function - rebin works for binning data and fps while the second one will remove the transit signature and then rebin.

np.random.seed(12345)

#feel free to change these globlal variables as per requirement.
GLOBAL_VIEW=2000
LOCAL_VIEW=200
FILEPATH_FPS="E:\Masters_Project_Data\\alienworlds_fps\\"
FILEPATH_DATA="E:\Masters_Project_Data\\alienworlds_data\\"

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

def improve_local_view(pathin,pathout,globsize,locsize,count_max,datname):
    prob_entry_list=[]
    dataset=os.scandir(pathin)
    entries=list(dataset)
    np.random.shuffle(entries)
    count=0
    for el in entries:
        hdu = fits.open(pathin+el.name)
        n=len(hdu)
        count=count+1
        if(count==count_max): break
        for i in range(1,n-1):
            if(hdu[i].header['TDUR']==None or hdu[i].header['TPERIOD']==None): continue
            if(hdu[i].header['TDUR']<0.1 or hdu[i].header['TPERIOD']<0.001): 
                print('miss 1:',el.name[4:13],i)
                prob_entry_list.append((el.name,i))
                continue
            phase=hdu[i].data['PHASE']
            tr_pd=hdu[i].header['TPERIOD']
            tr_dur=hdu[i].header['TDUR']/24
            flux=hdu[i].data['LC_DETREND'] 

            temp=np.array([[(phase[i]/tr_pd),flux[i]] for i in range(0,len(flux)) if(not np.isnan(flux[i]))])
            x=temp[:,0]
            y=temp[:,1]
    
            #create bins needed according to local and global view
            bins=np.linspace(-0.25,0.75,globsize+1,endpoint=True)

            #y[(1 < x) & (x < 5)]
            #median out the contents of a group
            temp=[[bins[i-1],np.median(np.array(y[(bins[i-1] < x) & (x < bins[i])]))] for i in range(1,len(bins))
                if len(y[(bins[i-1] < x) & (x < bins[i])])]
            df_gl_x=np.array(temp)[:,0]
            df_gl_y=np.array(temp)[:,1]
    
            print('int:',len(df_gl_x),len(df_gl_y))
            func=interp1d(df_gl_x,df_gl_y,kind='quadratic',fill_value='extrapolate')
            df_gl_x=bins[:-1]
            df_gl_y=func(bins[:-1])
            print('fin:',len(df_gl_x),len(df_gl_y))

            #look above or look below?
            med=np.median(df_gl_y)
            std=np.std(df_gl_y)
            count_down=(df_gl_y < med-2.5*std).sum()
            count_up=(df_gl_y > med+2.5*std).sum()
            if(count_up>count_down):    cut_phase=df_gl_x[np.argmax(df_gl_y)]
            else:   cut_phase=df_gl_x[np.argmin(df_gl_y)]

            #trying out what i hope is a more robust approach
            cut=min(tr_dur*2,np.abs(tr_pd*(0.75-cut_phase)),np.abs(tr_pd*(cut_phase-0.25)))
            bins_lc=np.linspace(cut_phase-cut/tr_pd,cut_phase+cut/tr_pd,locsize+1,endpoint=True)

            #print('check',len(bins),len(bins_lc))
            
            print(cut_phase,cut/tr_pd)
            temp=[(x[i],y[i]) for i in range(0,len(x)) if(x[i]<cut_phase+cut/tr_pd and x[i]>cut_phase-cut/tr_pd)]
            if(len(np.array(temp).shape)<2):
                print('miss 2:',el.name[4:13],i)
                prob_entry_list.append((el.name,i))
                continue
            xl=np.array(temp)[:,0]
            yl=np.array(temp)[:,1]

            temp=[[bins_lc[i-1],np.median(np.array(yl[(bins_lc[i-1] < xl) & (xl < bins_lc[i])]))] for i in range(1,len(bins_lc))
                if len(yl[(bins_lc[i-1] < xl) & (xl < bins_lc[i])])]
            df_lc_x=np.array(temp)[:,0]
            df_lc_y=np.array(temp)[:,1]
            print('int:',len(df_lc_x),len(df_lc_y))

            if(len(df_lc_y)<locsize/4):
                print('miss 2:',el.name[4:13],i)
                prob_entry_list.append((el.name,i))
                continue
            if(df_lc_x[0]>bins_lc[0]+cut*0.5/tr_pd or df_lc_x[-1]<bins_lc[-1]-cut*0.5/tr_pd):
                print('miss 3:',el.name[4:13],i)
                prob_entry_list.append((el.name,i))
                continue

            funcl=interp1d(df_lc_x,df_lc_y,kind='quadratic',fill_value='extrapolate',bounds_error=False)
            df_lc_x=bins_lc[:-1]
            df_lc_y=funcl(bins_lc[:-1])
            print('fin:',len(df_lc_x),len(df_lc_y))

            op_lc = pd.DataFrame(list(zip(df_lc_x, df_lc_y)),columns =['phase', 'flux'])
            op_gl = pd.DataFrame(list(zip(df_gl_x, df_gl_y)),columns =['phase', 'flux'])
            problist = pd.DataFrame(list(prob_entry_list),columns =['file', 'hdu'])

            print('hit:',count,el.name[4:13],i,len(op_lc),len(op_gl))

            op_lc.to_csv(pathout+'/local/'+el.name[4:13]+'_'+str(i)+'_l',sep=' ',index=False)
            op_gl.to_csv(pathout+'/global/'+el.name[4:13]+'_'+str(i)+'_g',sep=' ',index=False)
    problist.to_csv(pathout+'/'+datname,sep=' ',index=False)



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

    noise=np.random.normal(mean,0.5*sigma,size=count)

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
    av_entry=ascii.read('autovetter_label.tab')
    av_pl=np.array(av_entry['tce_plnt_num'])
    ref_kepid=[('0000'+str(el)[:9])[-9:] for el in av_entry['kepid']]
    ref_label=av_entry['av_training_set']
    entries=list(dataset)
    np.random.shuffle(entries)
    x=0
    for el in entries:
        #if(x==10): break
        hdu = fits.open(pathin+el.name)
        try: res = hdu[len(hdu)-1].data['RESIDUAL_LC']
        except: continue
        for imp in range(1,len(hdu)-1):
            flux=hdu[imp].data['LC_DETREND']
            phase=hdu[imp].data['PHASE']
            ind_arr=[i for i in range(0,len(phase)-1) if (phase[i]*phase[i-1]<0 and np.abs(phase[i]*phase[i-1])<0.0001)]
            x=x+1
            

            try:
                loc=np.where(np.array(ref_kepid)==el.name[4:13])
                loc_f=[m for m in loc[0] if str(av_pl[m])==str(1)]
            except ValueError as ve:
                print("miss ind:",el.name[4:13])
                continue
            if(len(loc_f)==0): continue
            if(ref_label[loc_f[0]]=='UNK'): continue

            cuts=int(len(flux)/binsize)
            lightcurve=[]
            labelling=[]
            for val in range(0,cuts-1):
                red_flux=flux[val*binsize:binsize*(val+1)]
                red_res=res[val*binsize:binsize*(val+1)]
                count_nan=np.isnan(red_flux).sum()
                if(count_nan/binsize > 0.2): continue

                trackrog=np.where(np.isnan(red_flux))
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
                #count_tr=(red_flux < med-2*std).sum()
                #if(count_tr>5): lightcurve.append(red_flux)
                if(np.any(np.array([x>val*binsize and x<binsize*(val+1) for x in ind_arr]))):
                    if(red_flux[np.argmin(red_flux)]<med-2.5*std): 
                        lightcurve.append(red_flux)
                        if(ref_label[loc_f[0]]=='PC'): labelling.append([1,0,0])
                        else: labelling.append([0,1,0]) 
                    #else:
                    #    labelling.append([0,0,1]) 
                    #    lightcurve.append(red_flux) 
                else:
                    for m in trackrog:
                        red_res[m]=0
                    if((np.isnan(red_res)).sum()<4):
                        #print('check') 
                        labelling.append([0,0,1]) 
                        lightcurve.append(red_flux)
            if(len(lightcurve)==0): continue  

            print(x,np.array(lightcurve).shape,np.array(labelling).shape,el.name[4:13],imp)
            np.savetxt(pathout+'xlabel/'+el.name[4:13]+'_'+str(imp)+'.dat',lightcurve,delimiter=' ')
            np.savetxt(pathout+'ylabel/'+el.name[4:13]+'_'+str(imp)+'.dat',labelling,delimiter=' ')

#make cuts about phase zero and get 200... or whatever points around it.
def get_transits_from_raw_v2(binsize,pathin,pathout):
    dataset=os.scandir(pathin)
    entries=list(dataset)
    np.random.shuffle(entries)
    x=0
    for el in entries:
        x=x+1
        hdu = fits.open(pathin+el.name)
        #if(x==10): break
        for imp in range(1,len(hdu)-1):
            flux=hdu[imp].data['LC_DETREND']
            phase=hdu[imp].data['PHASE']
            tdur=hdu[imp].header['TDUR']
            if(tdur*2>binsize): 
                print('too big:',el.name[4:13],imp)
                continue
            ind_arr=[i for i in range(0,len(phase)-1) if (phase[i]*phase[i-1]<0 and np.abs(phase[i]*phase[i-1])<0.0001)]
            np.random.shuffle(ind_arr)
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
                    #cut=int(binsize/3)
                    #count_tr=[(red_flux[int(k):int(k+cut)] < med-1.5*std).sum() for k in np.linspace(0,binsize-cut,cut)]
                    #if(np.any(np.array(count_tr)>5)): lightcurve.append(red_flux)
                    if(red_flux[np.argmin(red_flux)]<med-2*std): lightcurve.append(red_flux)

                
                if(len(lightcurve)==50): break
        
            if(len(lightcurve)==0): 
                print("miss:",el.name[4:13],imp,hdu[imp].header['NTRANS'])
                continue  
            print("hit:",x,np.array(lightcurve).shape,el.name[4:13],imp,hdu[imp].header['NTRANS'])
            np.savetxt(pathout+el.name[4:13]+'_'+str(imp)+'.dat',lightcurve,delimiter=' ')

#a more robust means to get data... right now the problem is incomplete light curves ... how do you expect to get data properly if you dont 
#have the appropriate bins to work with ? Current perspective ... just grab a bunch of transits and detect FPS from there... but the
#problem may not be that simple
def get_transits_from_raw_v3(binsize,pathin,pathout):
    dataset=os.scandir(pathin)
    entries=list(dataset)
    np.random.shuffle(entries)
    x=0
    for el in entries:
        x=x+1
        hdu = fits.open(pathin+el.name)
        try:
            impact=np.argmax(np.array([hdu[i].header['TDEPTH'] for i in range(1,len(hdu)-1) if hdu[i].header['TDEPTH']>0]))
        except:
            #impact=1
            print("miss 1:",el.name[4:13])
            continue
        flux=hdu[impact+1].data['LC_DETREND']
        phase=hdu[impact+1].data['PHASE']
        if(x==25): break
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
                cut=int(binsize/3)
                count_tr=[(red_flux[int(k):int(k+cut)] < med-2*std).sum() for k in np.linspace(0,binsize-cut,cut)]
                if(np.any(np.array(count_tr)>7)): lightcurve.append(red_flux)
            
            if(len(lightcurve)==3): break
    
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
           
def get_shorter_ones(binsize,pathin,pathout):
    dataset=os.scandir(pathin)
    entries=list(dataset)
    np.random.shuffle(entries)
    time=binsize/(2*24) #days
    x=0
    for el in entries:
        hdu = fits.open(pathin+el.name)
        try:
            #durarr=np.array([i for i in range(1,len(hdu)-1) if hdu[i].header['TPERIOD']< time/2])
            #if(len(durarr)==0): 
            #    print("miss 1:",el.name[4:13],hdu[1].header['TPERIOD'])
            #    continue
            
            #impact=durarr[np.argmax(np.array([hdu[i].header['TDEPTH'] for i in durarr]))]
            impact=np.argmax(np.array([hdu[i].header['TDEPTH'] for i in range(1,len(hdu)-1) if hdu[i].header['TDEPTH']>0]))
            impact=impact+1
        except:
            #impact=1
            print("miss 2:",el.name[4:13])
            continue
        flux=hdu[impact].data['LC_DETREND']
        phase=hdu[impact].data['PHASE']
        #if(x==25): break
        ind_arr=[i for i in range(0,len(phase)-1) if (phase[i]*phase[i-1]<0 and np.abs(phase[i]*phase[i-1])<0.0001)]
        lightcurve=[]
        for ind in ind_arr:
            red_flux=flux[ind-int(binsize/2):ind+int(binsize/2)]
            red_flux=np.array(red_flux)
            count_nan=np.isnan(red_flux).sum()
            if(count_nan/binsize > 0.1): continue

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
                cut=int(binsize/19)
                count_tr=[(red_flux[int(k):int(k+cut)] < med-2*std).sum() for k in np.linspace(0,binsize-cut,cut)]
                if(np.any(np.array(count_tr)>7)): lightcurve.append(red_flux)
            
            if(len(lightcurve)==3): break
    
        if(len(lightcurve)==0): 
            print("miss 3:",el.name[4:13])
            continue  
        print("hit:",x,np.array(lightcurve).shape,el.name[4:13],impact)
        x=x+1 
        np.savetxt(pathout+el.name[4:13]+'_'+str(impact)+'.dat',lightcurve,delimiter=' ')

def transit_only(pathin,pathout):
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
        if(npix<1 or npixdur<1):
            print("miss 1:",el.name[4:13])
            continue

    
        #if(x==10): break
        ind_arr=[i for i in range(0,len(phase)-1) if (phase[i]*phase[i-1]<0 and np.abs(phase[i]*phase[i-1])<0.0001)]

        cut=min(npixdur*2,npix*0.75)
        lightcurve=[]
        for i in range(0,len(ind_arr)):
            red_flux=flux[int(ind_arr[i]-cut):int(ind_arr[i]+cut)]
            red_flux=np.array(red_flux)
            count_nan=np.isnan(red_flux).sum()
            if(len(red_flux)==0):   continue
            if(count_nan/len(red_flux) > 0.1): continue

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
            med=np.median(red_flux)
            std=np.std(red_flux)
            if(red_flux[np.argmin(red_flux)]<med-1.5*std): lightcurve.append(red_flux)
                #if(count_tr>npixdur): lightcurve.append(red_flux)
            if(len(lightcurve)==6): break

        if(len(lightcurve)==0): 
            print("miss 2:",el.name[4:13])
            continue
        checks=[1 for m in range(1,len(lightcurve)) if(len(lightcurve[m])!=len(lightcurve[1]))]
        if(len(checks)>0): continue
        print("hit:",x,np.array(lightcurve).shape,el.name[4:13])
        x=x+1 
        try:
            np.savetxt(pathout+el.name[4:13]+'.dat',lightcurve,delimiter=' ')
        except:
            continue

def rebin_the_raw(binsize,rebin,pathin,pathout):
    entries=os.listdir(pathin)
    np.random.shuffle(entries)
    x=0
    for el in entries:
        x=x+1
        #if(x==25): break
        hdu = fits.open(pathin+el)
        try:
            impact=np.argmax(np.array([hdu[i].header['TDEPTH'] for i in range(1,len(hdu)-1) if hdu[i].header['TDEPTH']>0]))
        except:
            #impact=1
            print("miss 1:",el[4:13])
            continue
        try: tdur=int(hdu[impact+1].header['TDUR']*2)
        except: tdur=0
        if(tdur<binsize/rebin): 
            print('wash out:',el[4:13],tdur,binsize/rebin)
            continue

        flux=hdu[impact+1].data['LC_DETREND']
        phase=hdu[impact+1].data['PHASE']
        ind_arr=[i for i in range(0,len(phase)-1) if (phase[i]*phase[i-1]<0 and np.abs(phase[i]*phase[i-1])<0.001)]
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
                interpx=np.arange(0,len(red_flux),1)
                newx=np.linspace(0,interpx[-1],rebin,endpoint=True)
                func=interp1d(interpx,red_flux,kind='quadratic')
                newy=func(newx)
                #med=np.median(red_flux)
                #std=np.std(red_flux)
                #cut=int(binsize/19)
                #count_tr=[(red_flux[int(k):int(k+cut)] < med-2*std).sum() for k in np.linspace(0,binsize-cut,cut)]
                if(np.any(np.isnan(red_flux))):
                    print('fail interp:',el[4:13])
                    continue
                lightcurve.append(newy)
            
            if(len(lightcurve)==1): break
    
        if(len(lightcurve)==0): 
            print("miss:",el[4:13])
            continue  
        print("hit:",x,np.array(lightcurve).shape,el[4:13],impact+1,hdu[impact+1].header['TDEPTH'])
        np.savetxt(pathout+el[4:13]+'_'+str(impact+1)+'.dat',lightcurve,delimiter=' ')

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
#get_shorter_ones(6000,FILEPATH_DATA,'data_red_shortdur_6000/')
#rebin_the_raw(3000,1000,FILEPATH_FPS,'raw_rebin2000/')
#transit_only(FILEPATH_DATA,'data_stitch_prelim/')
#improve_local_view(FILEPATH_FPS,'new_loc_glob',2000,200,4000,'probdat_fps')
#improve_local_view(FILEPATH_DATA,'new_loc_glob',2000,200,4000,'probdat_pl')
get_transits_from_raw(500,FILEPATH_DATA,'data_red_raw_dirty500/')
#get_transits_from_raw(500,FILEPATH_FPS,'data_red_raw_dirty500/')