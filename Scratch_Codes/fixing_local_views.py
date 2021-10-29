import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

PATH='data_red/local/'
entry=os.scandir(PATH)

for el in entry:
    df=pd.read_csv(PATH+el.name,sep=" ")
    if(len(df)<200):
        phasedif=df['phase'].iloc[1]-df['phase'].iloc[0]
        phaselast=df['phase'].iloc[len(df)-1]
        temp=[df['phase'].iloc[i]-df['phase'].iloc[i-1] for i in range(1,len(df))]
        print(temp)
        count=200-len(df)
        mean=df['flux'].mean()
        sig=df['flux'].std()
        phase_arr=[phaselast+i*phasedif for i in range(1,count+1)]
        #ran_arr=np.random.uniform(mean,sig,size=count)
        #df2 = pd.DataFrame({"phase":phase_arr,"flux":ran_arr})
        #df=df.append(df2, ignore_index = True)
        #print(phase_arr[-1])
        #print(len(df))
        #df.to_csv('data_red/'+el.name,sep=" ")