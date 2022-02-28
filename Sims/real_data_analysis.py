from random import gauss
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import os
from astropy.io import ascii

import GetLightcurves as gc

entries = os.listdir('../../processed_directories/find_circles/')
rv_entry=ascii.read('../../Catalogs/robovetter_label.dat')
rv_pl=np.array(rv_entry['tce_plnt_num'])
rv_label = rv_entry['label']
rv_kepid=[('0000'+str(el)[:9])[-9:] for el in rv_entry['kepid']]
av_entry=ascii.read('../../Catalogs/autovetter_label.tab')
av_pl=np.array(av_entry['tce_plnt_num'])
av_label = av_entry['av_training_set']
av_kepid=[('0000'+str(el)[:9])[-9:] for el in av_entry['kepid']]
av_time=av_entry['tce_duration']
av_p=av_entry['tce_period']
av_timeerr=av_entry['tce_duration_err']


'''fig, axs = plt.subplots(4,4,figsize=(10,10))

for el,ax in zip(entries[32:32+16],axs.ravel()):
    df = pd.read_csv('../../processed_directories/find_circles/'+el)
    ax.plot(df['phase_l'],df['flux_l'])
    ax.plot(df['phase_l'],df['flux_l']-df['model_l'])
    ax.plot(df['phase_l'],df['model_l'])

plt.show()'''

def gausses(x, A1, m1, s1, A2, m2, s2):
    y = A1*np.exp(-(x-m1)**2/(2*s1**2)) + A2*np.exp(-(x-m2)**2/(2*s2**2))
    return(y)

def lorz(x,A1,x0,g, A2, x02, g2):
    y = A1 / (1000*((x-x0)**2+(g/2)**2)) + A2 / (1000*((x-x02)**2+(g2/2)**2))
    return(y)

#getting a fit: both gaussian and lorentzian seem like a good idea so go for both?


'''phase, tr2d, tr3d = np.loadtxt('2d3d_0.1R_circ.csv', delimiter=' ', unpack=True)
res = tr2d-tr3d
plt.plot(phase, res)

print(min(tr3d))

dur2d = [phase[i] for i in range(0,len(phase)) if(tr3d[i]<1)]
dur3d = [phase[i] for i in range(0,len(phase)) if(tr3d[i]<1.001*min(tr3d))]
print('td: ',dur3d[0]-dur3d[-1], dur2d[0]-dur2d[-1])

plt.plot(phase, tr2d)
plt.plot(phase, tr3d)
plt.scatter(dur3d[0],0.99)
plt.scatter(dur3d[-1],0.99)
plt.scatter(dur2d[0],1)
plt.scatter(dur2d[-1],1)

#plt.plot(phase, lorz(phase,0.001,-0.5, 0.05,0.001, 0.5, 0.05))
popt1, pcov1 = curve_fit(gausses, phase, res, bounds=([0,phase[0],0,0,0,0], [max(res), 0, np.inf,max(res), phase[-1], np.inf]))
popt2, pcov2 = curve_fit(lorz, phase, res, bounds=([0,phase[0],0,0,0,0], [max(res), 0, np.inf,max(res), phase[-1], np.inf]))
plt.plot(phase, lorz(phase, *popt2))
plt.plot(phase, gausses(phase, *popt1))

print('ptd: ', popt1[1]-popt1[4], popt2[1]-popt2[4])
plt.show()'''

#now we take each local view lightcurve residual and try to fit lorentzian
for el in entries:
    df = pd.read_csv('../../processed_directories/find_circles/'+el)

    try: 
        loc=np.where(np.asarray(rv_kepid)==el[:9])[0]
        rv_loc_f = [x for x in loc if(str(rv_pl[x])==el[10:11])]
        #print(rv_loc_f, el)
    except: 
        print("not in catalog")
        continue

    if(len(rv_loc_f)==0): 
        print("not in catalog")
        continue
   
    res = np.array(df['flux_l'] - df['model_l'])
    res = np.array([ x for x in res if(not(np.isnan(x)))])
    phase = np.array(df['phase_l'])[:len(res)]
    print(len(res))
    try: popt1, pcov1 = curve_fit(gausses, phase, res, bounds=([0,phase[0],0,0,0,0], [max(res), 0, np.inf,max(res), phase[-1], np.inf]))
    except: 
        print("no gauss fit")
        continue
    try: popt2, pcov2 = curve_fit(lorz, phase, res, bounds=([0,phase[0],0,0,0,0], [max(res), 0, np.inf,max(res), phase[-1], np.inf]))
    except: 
        print("no lorz fit")
        continue

    print(popt1.shape, pcov1.shape)
    print(el[:9],'ptd: ', np.abs(popt1[1]-popt1[4]), np.abs(popt2[1]-popt2[4]))
     
    df1 = pd.DataFrame(zip(phase,df['flux_l'][:200],df['model_l'][:200], res, gausses(phase, *popt1), lorz(phase, *popt2)),
        columns=['phase', 'flux', 'model','residue', 'gaussian', 'lorentzian'])

    store = pd.HDFStore('../../processed_directories/fit_circles/'+el[:11])
    store.put('data', df1)
    store.get_storer('data').attrs.metadata = {'label':rv_label[rv_loc_f[0]],'gauss':popt1, 
        'lorz':popt2,'gdur':np.abs(popt1[1]-popt1[4]) ,'ldur':np.abs(popt2[1]-popt2[4]),'g_cov':pcov1.reshape(-1), 'l_cov':pcov2.reshape(-1)}
    store.close()
    




