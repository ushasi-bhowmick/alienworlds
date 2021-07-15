#Perhaps we've never been visited by aliens because they have looked upon Earth and decided there's no sign of intelligent life.
from astropy.io import fits, ascii
from astropy.table import Table, Column, MaskedColumn
from astropy import units as u
import numpy as np
import os
from matplotlib import pyplot as plt

color_arr=['#003f5c','#58508d','#bc5090','#ff6361','#ffa600','black','grey','green','magenta','yellow','blue']
#like '%Kepler%'
data2= ascii.read("ex_kepler_id.dat")

kepid=np.array(data2['Kepler_ID'])


print(len(kepid))
#we need two strings that give us the required kepler ID abbreviations to write the wget script
str_s=[]
str_b=[]
for el in kepid:
    no=len(str(el))
    val=str(el)
    for i in range(0,9-no):
        val='0'+val
    str_b.append(val)
    str_s.append(val[:4])

#2344 lines of data in kepler table
#trial run...

for i in range(0,len(str_b)):
    if(str_b[i]==str_b[i-1]): continue
    os.system("wget ftp://archive.stsci.edu/pub/kepler/dv_files/"+str_s[i]+"/"+str_b[i]+"/*.fits")

#print(len(np.unique(str_b))) => 1652 points
    
#wget "ftp://archive.stsci.edu/pub/kepler/dv_files/0007/000757450/*.fits"
#wget "https://archive.stsci.edu/pub/kepler/dv_files/0007/000757450/" -P /tmp -A "*.fits"
#wget -r -nd --no-parent -erobots=off -A.fits https://archive.stsci.edu/pub/kepler/dv_files/0017/001724719/

