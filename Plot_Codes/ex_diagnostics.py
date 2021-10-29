#“I'm sure the universe is full of intelligent life. It's just been too intelligent to come here.”
#― Arthur C. Clarke
from astropy.io import fits, ascii
from astropy.table import Table, Column, MaskedColumn
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt

color_arr=['#5b3b8c','#4B9CD3','#800020','#df6985','#ffa600','#00a86b','grey','red','black','#003153','#d1e231']

data = ascii.read("ex_archive_sum.tab")  
orb_p=np.array(data['pl_orbper'])
orb_sma=np.array(data['pl_orbsmax'])
pl_r=np.array(data['pl_rade'])
year=np.array(data['disc_year'])
method=np.array(data['discoverymethod'])
ra=np.array(data['ra'])
dec=np.array(data['dec'])
planets=data['pl_name']

#remove empty stuff from arrays
def filter_out(arr_miss, arr_fill, method_arr):
    temp1=[]
    temp2=[]
    temp3=[]
    for i in range(0,len(arr_miss)):
        if(arr_miss[i]>0 and arr_fill[i]>0):
            temp1.append(arr_miss[i])
            temp2.append(arr_fill[i])
            temp3.append(method_arr[i])
    return(temp1, temp2, temp3)

#figure out the distinct transit methods from whats given in the table
def classify(arr):
    classes=[]
    for el in arr:
        flag=0
        for stuff in classes:
            if(stuff==el): 
                flag=1
                break
        if(flag==0): classes.append(el)
    return classes

#help with the color coding
def zip_stuff(arr1,arr2,method, unique_methods):
    net=[]
    for el in unique_methods:
        temp1=[]
        temp2=[]
        for i in range(0, len(arr1)):
            if(el==method[i]):
                temp1.append(arr1[i])
                temp2.append(arr2[i])
        net.append([temp1,temp2])
                
    print(len(net[1][1]))
    return(net)

#color code the histogram
def zip_hist(arr1,method, unique_methods):
    net=[]
    for el in unique_methods:
        temp1=[]
        for i in range(0, len(arr1)):
            if(el==method[i]):
                temp1.append(arr1[i])
        net.append(temp1)
    return(net)
    

#call the function that finds out the various classification methods        
unique_methods=classify(method)

#first plot, planet orbital period vs planet radius
pl_r,orb_pn,m_arr=filter_out(pl_r,orb_p,method)
net=zip_stuff(pl_r,orb_pn,method,unique_methods)

fig1,ax1=plt.subplots(1,1,figsize=(12,6))
for i in range(0,len(unique_methods)):
    if(i==3): ax1.scatter(net[i][1],net[i][0],marker='.',color=color_arr[i],label=unique_methods[i], zorder=0)
    else: ax1.scatter(net[i][1],net[i][0],marker='.',color=color_arr[i],label=unique_methods[i], zorder=1)
ax1.legend()
ax1.set_xlabel('orbital period(days)')
ax1.set_ylabel('planet radii(earth radius)')
ax1.set_title('Period vs Radii for exoplanets')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(0.01,10000)
plt.savefig('ex_period_radii.png')

#plot 2 the histogram of various detection methods
max_year=year[np.argmax(year)]
min_year=year[np.argmin(year)]
hist_year=range(min_year,max_year+1)
net2=zip_hist(year,method,unique_methods)
fig2,ax2=plt.subplots(1,1,figsize=(6,6))
ax2.set_xlabel('year')
ax2.set_ylabel('number of planets')
ax2.set_title('Histogram of exoplanet detection')
ax2.hist(net2,bins=hist_year, stacked=True, density=True, label=unique_methods, color=color_arr)
ax2.legend()
plt.savefig('ex_exoplanet_hiistogram.png')

#plot 3 the RA DEC distribution: we can see the kepler FOV in the plot, so thats cool
fig3,ax3=plt.subplots(1,1,figsize=(10,6))
net3=zip_stuff(ra,dec,method,unique_methods)
ax3.scatter(ra,dec,marker='.')
for i in range(0,len(unique_methods)):
    ax3.scatter(net3[i][0],net3[i][1],marker='.',color=color_arr[i],label=unique_methods[i], zorder=1)
ax3.set_xlabel('RA(deg)')
ax3.set_ylabel('DEC(deg)')
ax3.set_xlim(-200,380)
ax3.set_title('Distribution of Planet-Rich stars')
ax3.legend()
plt.savefig('ex_ra_dec.png')

#plot 4 .... basically keplers second law, but still interesting to plot.
orb_sma,orb_pn2, m_arr=filter_out(orb_sma,orb_p, method)
net4=zip_stuff(orb_sma,orb_pn2,method,unique_methods)
fig4,ax4=plt.subplots(1,1,figsize=(10,6))
for i in range(0,len(unique_methods)):
    if(i==3): ax4.scatter(net4[i][0],net4[i][1],marker='.',color=color_arr[i],label=unique_methods[i], zorder=0)
    else: ax4.scatter(net4[i][0],net4[i][1],marker='.',color=color_arr[i],label=unique_methods[i], zorder=1)
ax4.legend()
ax4.set_xlabel('orbital semi-major axis(AU)')
ax4.set_ylabel('orbital period(days)')
ax4.set_title('Period vs distance for exoplanets')
ax4.set_xscale('log')
ax4.set_yscale('log')
plt.savefig('ex_period_distance.png')

plt.show()
