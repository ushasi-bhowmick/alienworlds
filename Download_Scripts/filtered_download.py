import numpy as np 
import os
from astropy.io import ascii
import pandas as pd

data_pl= ascii.read("../../Catalogs/ex_kepler_id.dat")
data_tce= ascii.read("../../Catalogs/ex_TCE_sum.tab")
data_koi=ascii.read("../../Catalogs/ex_koi_id.tab")
pl_entries=os.listdir("E:\Masters_Project_Data\\alienworlds_pl\\")
fps_entries=os.listdir("E:\Masters_Project_Data\\alienworlds_fps\\")
oth_entries=os.listdir("E:\Masters_Project_Data\\alienworlds_others\\")
temp_entries=os.listdir("E:\Masters_Project_Data\\alienworlds_data\\")

pl_entries=np.unique(np.asarray([el[4:13] for el in pl_entries]))
fps_entries=np.unique(np.asarray([el[4:13] for el in fps_entries]))
oth_entries=np.unique(np.asarray([el[4:13] for el in oth_entries]))
temp_entries=np.unique(np.asarray([el[4:13] for el in temp_entries]))

print('koitable',len(np.unique(data_koi)),len(pl_entries))
#remove candidates
idarr=[]
for i in range(0,len(data_koi['kepid'])):
    if(data_koi['koi_disposition'][i] == 'CANDIDATE'):
        idarr.append(data_koi['kepid'][i])

print('plfps',len(np.unique(idarr)),len(idarr))

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


#print(np.unique(data_koi['koi_disposition']))
koi_id_b,koi_id_s=get_strings(np.unique(idarr))

#koi_id_b_not=[el for el in koi_id_b if(not np.any(pl_entries==el) and not np.any(fps_entries==el))]
rem_pl=np.setdiff1d(koi_id_b, pl_entries)
print('rempl',len(rem_pl))
rem_pl=np.setdiff1d(rem_pl, fps_entries)
print('remfps',len(rem_pl))
rem_pl=np.setdiff1d(rem_pl, oth_entries)
print('remrem',len(rem_pl))
rem_pl=np.setdiff1d(rem_pl, temp_entries)
rem_pl_s=[el[:4] for el in rem_pl]

print("new:",len(rem_pl))

def download(i,f,str_b,str_s):
    for i in range(i,f):
        print(i)
        if(str_b[i]==str_b[i-1]): continue
        #os.system("wget ftp://archive.stsci.edu/pub/kepler/dv_files/"+str_s[i]+"/"+str_b[i]+"/*.fits")
        #os.system('wget -r -nd --no-parent -erobots=off -A.fits https://archive.stsci.edu/pub/kepler/dv_files/'+str_s[i]+'/'+str_b[i])
        os.system('wget -r -nd --no-parent --wait 1 --no-clobber -erobots=off -A.fits https://archive.stsci.edu/pub/kepler/dv_files/'+str_s[i]+'/'+str_b[i]+'/')

download(0,436,rem_pl,rem_pl_s)
