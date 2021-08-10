#"a spectrum is equal to a thousand pictures"... does this apply to lightcurves too?
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

#I'm whimsical and i know it...
color_arr=['#5b3b8c','#4B9CD3','#800020','#df6985','#ffa600','#00a86b','grey','red','black','#003153','#d1e231']

#here we're plotting sample light curves from the binned data that we made...
FILEPATH='nonpl_red/local/'

entries=os.scandir(FILEPATH)
entries=list(entries)
np.random.shuffle(entries)
i=0
j=0
tabs=0
plt.style.use('seaborn-bright')
#remove the sharey if needed... the representations can be weird sometimes.
fig,ax=plt.subplots(5,5,figsize=(20,20),sharex=True,sharey=True)

plt.suptitle('Example lightcurves')
for el in entries:
    tabs+=1
    df=pd.read_csv(FILEPATH+el.name, sep=" ")
    ax[i][j].set_title(el.name[0:11],size=9)
    ax[i][j].plot(df['phase'],df['flux'],marker='.',ls='None',color=color_arr[i+2-j])
    i=i+1
    if(i==0): ax[i][j].set_ylabel('flux')
    if(j==5): ax[i][j].set_xlabel('phase')

    if(i==5):
        i=0
        j+=1
    if(tabs==25): break

plt.show()
