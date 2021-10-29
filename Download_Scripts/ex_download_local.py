#Perhaps we've never been visited by aliens because they have looked upon Earth and decided there's no sign of intelligent life.
from astropy.io import fits, ascii
from astropy.table import Table, Column, MaskedColumn
from astropy import units as u
import numpy as np
import os
from matplotlib import pyplot as plt

color_arr=['#003f5c','#58508d','#bc5090','#ff6361','#ffa600','black','grey','green','magenta','yellow','blue']
#like '%Kepler%'
data_pl= ascii.read("ex_kepler_id.dat")
data_tce= ascii.read("ex_TCE_sum.tab")
data_koi=ascii.read("ex_koi_id.tab")

pl_kepid=np.array(data_pl['Kepler_ID'])
tce_kepid=np.array(data_tce['kepid'])
koi_kepid=np.array(data_koi['kepid'])
koi_desp=np.array(data_koi['koi_disposition'])

#we need two strings that give us the required kepler ID abbreviations to write the wget script
def get_strings(ids):
    str_s=[]
    str_b=[]
    for el in ids:
        no=len(str(el))
        val=str(el)
        for i in range(0,9-no):
            val='0'+val
        str_b.append(val)
        str_s.append(val[:4])
    return(str_b,str_s)

#remove common elements from two arrays... like separate out the confirmed planets from the TCE list, etc etc.
def filter_out(pl,tce):
    tce_new=[]
    for el in tce:
        flag=0
        for i in pl: 
            if(el==i): 
                flag=1
                break
        if(flag==0): tce_new.append(el)
    return(tce_new)
               
tce_kepid=filter_out(np.unique(pl_kepid),np.unique(tce_kepid))
tce_kepid=filter_out(np.unique(koi_kepid),np.unique(tce_kepid))

#generate a false positives list to extract information from...
koi_kepidf=[]
for i in range(0,len(koi_kepid)):
    if(koi_desp[i]=='FALSE POSITIVE'):
        koi_kepidf.append(koi_kepid[i])
        
print(len(np.unique(koi_kepidf)))
print(len(np.unique(koi_kepid)))
print(len(np.unique(pl_kepid)))
print(len(np.unique(tce_kepid)))

#get the required strings to download from any dataset needed
pl_str_b,pl_str_s=get_strings(pl_kepid)
tce_str_b,tce_str_s=get_strings(tce_kepid)
koi_str_b,koi_str_s=get_strings(koi_kepidf)

#function to download... right now configured to download dv files....
def download(i,f,str_b,str_s):
    for i in range(i,f):
        print(i)
        if(str_b[i]==str_b[i-1]): continue
        os.system("wget ftp://archive.stsci.edu/pub/kepler/dv_files/"+str_s[i]+"/"+str_b[i]+"/*.fits")
        #os.system('wget -r -nd --no-parent -erobots=off -A.fits https://archive.stsci.edu/pub/kepler/dv_files/'+str_s[i]+'/'+str_b[i])

#this means I'm downloading false positives... for confirmed planets use pl_str_b,pl_str_s
download(1500,2500,koi_str_b,koi_str_s)
    
#wget "ftp://archive.stsci.edu/pub/kepler/dv_files/0007/000757450/*.fits"
#wget "https://archive.stsci.edu/pub/kepler/dv_files/0007/000757450/" -P /tmp -A "*.fits"
#wget -r -nd --no-parent -erobots=off -A.fits https://archive.stsci.edu/pub/kepler/dv_files/0017/001724719/
