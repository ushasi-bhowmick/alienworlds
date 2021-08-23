import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

FILEPATH_DATA='nonpl_data_red/global/'
FILEPATH_FPS='nonpl_fps_red/global/'


entries_data=os.scandir(FILEPATH_DATA)
#print(len(list(entries_data)))

for el in entries_data:
    df=pd.read_csv(FILEPATH_DATA+el.name,sep=" ")
    
    mid=np.median(np.array(df['flux']))
    std=np.std(np.array(df['flux']))
    count=0
    for row in df['flux']:
        if(row<mid-2.5*std or row>mid+3*std):
            row=np.NaN
            count+=1

    clean=np.array([val for val in df['flux'] if not np.isnan(val)])
    if(len(clean)==0): continue
    mean=np.mean(clean)
    sigma=np.std(clean)
    noise=np.random.normal(mean,sigma,size=count)
    j=0
    for row in df['flux']:
        if(count==0): break
        if(np.isnan(row)):
            row=noise[j]
            j+=1
    print(len(df))
    
    df.to_csv('nonpl_data_red2/global/'+el.name[0:11],sep=' ',index=False)