from random import gauss
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import os
from astropy.io import ascii

import GetLightcurves as gc

entries = os.listdir('../../processed_directories/go_circles/find_circles/')
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
'''for el in entries:
    df = pd.read_csv('../../processed_directories/go_circles/find_circles/'+el)

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
   
    k = np.ones(10)/10
    flux = np.convolve(np.array([ x for x in df['flux_l'] if(not(np.isnan(x)))]),k,mode='same')
    model = np.convolve(np.array([ x for x in df['model_l'] if(not(np.isnan(x)))]),k,mode='same')

    res = flux - model
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

    store = pd.HDFStore('../../processed_directories/go_circles/fit_circles_smooth/'+el[:11])
    store.put('data', df1)
    store.get_storer('data').attrs.metadata = {'label':rv_label[rv_loc_f[0]],'gauss':popt1, 
        'lorz':popt2,'gdur':np.abs(popt1[1]-popt1[4]) ,'ldur':np.abs(popt2[1]-popt2[4]),'g_cov':pcov1.reshape(-1), 'l_cov':pcov2.reshape(-1)}
    store.close()'''

#made the directory of fits... just a quick check to see the fits...
entries = os.listdir('../../processed_directories/go_circles/fit_circles/')

'''fig, axs = plt.subplots(3,3,figsize=(10,10))


for el,ax in zip(entries[:16],axs.ravel()):
    store = pd.HDFStore('../../processed_directories/analyse_circles/'+el)
    df = store['data']
    #ax.plot(df['phase'],df['flux'])
    #ax.plot(df['phase'],df['model'])
    ax.plot(df['phase'],df['residue'])
    ax.plot(df['phase'],df['gaussian'])
    ax.plot(df['phase'],df['lorentzian'])
    store.close()

plt.show()'''

'''for el in entries:
    store = pd.HDFStore('../../processed_directories/go_circles/fit_circles/'+el)
    try: metadata = store.get_storer('data').attrs.metadata
    except: 
        store.close()
        continue
    
    dur1 = metadata['gdur']
    arr1 = metadata['gauss']
    dur2 = metadata['ldur']
    arr2 = metadata['lorz']
    #print('copy '+'../../processed_directories/fit_circles/'+el+' ../../processed_directories/analyse_circles')
    if((dur1 < 1 and dur1 > 0.5) or (dur2 < 1 and dur2 > 0.5)): 
        if((arr1[1]>-0.5 and arr1[4]<0.5) or (arr2[1]>-0.5 and arr2[4]<0.5)):
            #print(arr1[1], arr1[4],arr2[1], arr2[4])
            store2 = pd.HDFStore('../../processed_directories/go_circles/analyse_circles_2/'+el[:11])
        
            data = store['data']
            print(dur1, dur2)

            store2.put('data', data)
            store2.get_storer('data').attrs.metadata = metadata
            store2.close()
        
    store.close()'''

entries = os.listdir('../../processed_directories/go_circles/analyse_circles_smooth/')
plt.style.use('seaborn-bright')    

for el in entries[300:]:
    fig, ax = plt.subplots(2,1,figsize=(6,6))
    store = pd.HDFStore('../../processed_directories/go_circles/analyse_circles_smooth/'+el)
    df = store['data']
    mn = np.mean(df['residue']**2)
    #k = np.ones(2)/2
    ax[0].plot(df['phase'],df['flux'], label='flux')
    #ax[0].plot(df['phase'],np.convolve(df['flux'], k,mode='same'))
    ax[0].plot(df['phase'],df['model'], label='model')
    #ax[1].plot(df['phase'],np.convolve(df['residue'], k,mode='same'))
    ax[1].plot(df['phase'],df['residue'], label='residue')
    ax[1].plot(df['phase'],df['gaussian'], label='Gaussian fit')
    ax[1].plot(df['phase'],df['lorentzian'], label='Lorentzian fit')
    ax[0].set_ylabel('Flux')
    ax[1].set_xlabel('Phase')
    ax[1].set_ylabel('Flux - Model')
    ax[1].fill_between(df['phase'], np.sqrt(mn)*np.ones(len(df['phase'])), -np.sqrt(mn)*np.ones(len(df['phase'])), alpha=0.2)
    ax[1].legend()
    ax[0].legend()
    ax[0].set_title(el[:9]+'\nPl:'+el[10:])
    plt.savefig('../../processed_directories/go_circles/plots_smooth/'+el+'.png')
    plt.close()
    store.close()


