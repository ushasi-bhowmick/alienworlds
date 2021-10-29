import numpy as np
from astropy.io import ascii, fits
import pandas as pd
import os

from pandas.core.dtypes.missing import isna

#this is my personal devised NN ... may or may not work but I have some hopes up from it.

#this function takes out a chunk from the LC... coz the whole thing may be too computationally expensive to deal with.
#the chunk is 1/3 of the whole thing that amounts to about 24000 input samples... should be enough to run a conv or autoencoder NN
#we check the transit periods and sort the data into good ones and problematic ones. The good ones are those where the selected chunk most 
#definitely has a transit... te problematic ones may need manual intervention to select targets.

#we need to do something about the NaN values better to interpolate between the nan values
def take_out_from_dv():
    FILEPATH_RAW_PL="F:\Masters_Project_Data\\alienworlds_data\\"
    FILEPATH_RAW_FPS="F:\Masters_Project_Data\\alienworlds_fps\\"
    FILEPATH_RAW_OTH="F:\Masters_Project_Data\\alienworlds_others\\"
    entry_pl=os.scandir(FILEPATH_RAW_PL)
    entry_fps=os.scandir(FILEPATH_RAW_FPS)

    for el in entry_pl:
        hdu=fits.open(FILEPATH_RAW_PL+el.name)
        num=len(hdu)
        obsdur=hdu[1].data["TIME"][-1]-hdu[1].data["TIME"][0]
        dang_list=[hdu[i].header['NTRANS']*hdu[i].header['TPERIOD'] for i in range(1,num-1)]
        datl=len(hdu[1].data["TIME"])

        x=hdu[1].data["TIME"][int(datl*7/15):int(datl*8/15)]
        y=hdu[1].data["LC_WHITE"][int(datl*7/15):int(datl*8/15)]
        yr=hdu[1].data["LC_WHITE"][int(datl*7/15):int(datl*8/15)]

        if(np.any(dang_list<obsdur/2)): 
            print(el.name[4:13],dang_list,obsdur/2)
            df=pd.DataFrame(list(zip(x, y)),columns =['time', 'flux'])
            df.to_csv("ushasi_NN_data/short_prob_ones/"+el.name[4:13],sep=' ',index=False)
            continue

        nancount=np.isnan(y).sum()
        if(nancount/len(y)>0.2): 
            print("nans:",nancount/len(y))
            df=pd.DataFrame(list(zip(x, y)),columns =['time', 'flux'])
            df.to_csv("ushasi_NN_data/short_prob_ones/"+el.name[4:13],sep=' ',index=False)
            continue

        for i in range(0,len(y)):
            if(np.isnan(y[i])):
                t=1
                try:
                    while(np.isnan(y[i-t]) or np.isnan(y[i+t])):    
                        if(i-t <0):
                            y[i]=y[i+t]
                            break
                        if(i+t >len(x)): 
                            y[i]=y[i-t]
                            break
                        t+=1
                    y[i]=(y[i-t]+y[i+t])/2
                except:
                    y[i]=0

        print("clean",len(x),np.isnan(y).sum(),np.isnan(yr).sum())
        df=pd.DataFrame(list(zip(x, y)),columns =['time', 'flux'])
        #print(el.name[4:13],dang_list,obsdur/3,len(df))
        df.to_csv("ushasi_NN_data/short_good_ones/"+el.name[4:13],sep=' ',index=False)   
        

#now the next question is assigning labels and making a working NN training sample out of this. To do this we use the filtered
#robovetter catalog... Now some TCE targets will inadvertently invite planet candidates into them... we'll label them as planets
#coz we can't create another class out of it
def get_labels_and_make_TS():
    FILEPATH="ushasi_NN_data/short_good_ones/"
    entry=os.listdir(FILEPATH)
    cat=ascii.read('robovetter_label.dat')
    kepid=cat['kepid']
    kepid=[('00000'+str(el))[-9:] for el in kepid]
    label=cat['label']
    Xtrain=[]
    Ytrain=[]

    for el in entry:
        df=pd.read_csv(FILEPATH+el,sep=" ")
        loc=np.where(np.array(kepid)==el)
        this_label=np.array([label[m] for m in loc[0]])
        if(np.all(this_label=='CANDIDATE')): continue
        temparr=[0,0]
        #if(np.any(this_label=='CONFIRMED') or np.any(this_label=='CANDIDATE')): temparr[0]=1
        #if(np.any(this_label=='FPS')): temparr[1]=1
        if(this_label[0]=='CONFIRMED' or this_label[0]=='CANDIDATE'): temparr[0]=1
        if(this_label[0]=='FPS'): temparr[1]=1

        if(np.all(temparr==1)): continue
        print(len(df),temparr,this_label)
        Xtrain.append(np.array(df['flux']))
        Ytrain.append(np.array(temparr))


    np.savetxt('training_data/Xtrain_ush_short_nolap.csv', np.transpose(np.array(Xtrain)), delimiter=',')
    np.savetxt('training_data/Ytrain_ush_short_nolap.csv', np.array(Ytrain), delimiter=',')

get_labels_and_make_TS()
#take_out_from_dv()