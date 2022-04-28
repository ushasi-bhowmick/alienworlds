import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import aliensims as dysim
import os
import time
import pandas as pd
from transit import occultnonlin, occultquad
from multiprocessing import Process, Pool

#yo! kumaran recieving? over and out.
# 
testg = 0
start_time = time.time()
fl = np.pi/3
sides = 3

def new_plar(ph,p,u1,u2,rorb,imp):
    znp = np.sqrt(np.abs(rorb*np.sin(ph*np.pi))**2+imp**2)
    a= occultquad(znp,p,[u1,u2])  
    return(a -1) 

def test_multi_loops_3d(x):
    global testg
    global fl
    np.random.seed(1234*x)
    sim_3d = dysim.Simulator (100, 50000, 500, fl, limb_u1=0.6, limb_u2=0.0)
    meg_3d = dysim.Megastructure(200, True, 10)
    sim_3d.add_megs(meg_3d)
    sim_3d.set_frame_length()
    sim_3d.simulate_transit()
    fl=sim_3d.frame_length
    if(x==0): print("Count:", meg_3d.set, x, np.pi/sim_3d.frame_length)
    return(sim_3d.lc)

def test_multi_loops_2d(x):
    global testg
    np.random.seed(3456*x)
    global fl
    sim_2d = dysim.Simulator (100, 20000, 500, fl, limb_u1=0.0, limb_u2=0.0)
    meg_2d = dysim.Megastructure(200, True, 20, isrot=True)
    sim_2d.add_megs(meg_2d)
    sim_2d.set_frame_length()
    if(x==0): print("Count:", meg_2d.set, x, np.pi/sim_2d.frame_length)
    sim_2d.simulate_transit()
    return(sim_2d.lc)

def shape_test(x):
    global sides
    np.random.seed(3456*x)
    global fl
    sim_2d = dysim.Simulator (100, 20000, 500, fl, limb_u1=0.0, limb_u2=0.0)
    meg_2d = dysim.Megastructure(200, isrot=True)
    meg_2d.regular_polygons_2d(20,sides)
    sim_2d.add_megs(meg_2d)
    sim_2d.set_frame_length()
    if(x==0): print("Count:", meg_2d.set, x, np.pi/sim_2d.frame_length)
    sim_2d.simulate_transit()
    return(sim_2d.lc)


def twoD_vs_threeD():
    global fl
    global testg
    # start 4 worker processes
    with Pool(processes=40) as pool:
        lc2dsum = np.asarray(pool.map(test_multi_loops_2d, range(160)))
        lc2d = np.mean(lc2dsum, axis = 0)
        lc2dstd = np.sqrt(np.mean((lc2dsum-lc2d)**2, axis=0))
        print("--- %s min ---" % ((time.time() - start_time)/60))

        lc3dsum = np.asarray(pool.map(test_multi_loops_3d, range(160)))
        print("--- %s min ---" % ((time.time() - start_time)/60))

        lc3d = np.mean(lc3dsum, axis = 0)
        lc3dstd = np.sqrt(np.mean((lc3dsum-lc3d)**2, axis=0))

        mn = (np.asarray(lc3d-lc2d)**2).sum()/len(lc3d)
        print(np.sqrt(mn))

        plt.style.use('seaborn-bright')
        fig, ax = plt.subplots(2,1, figsize = (7,10), sharex=True)

    frm = np.linspace(-fl,fl, len(lc3d))
    print(fl)
    #model = new_plar(frm/np.pi,0.01, 0,0,2,0)+1
    ax[0].plot(frm,lc2d,label = '2d')
    ax[0].plot(frm, lc3d, label = '3d')
    #ax[0].plot(frm, model, label='model' )
    #print('noise:', np.std(model-lc3d), 'snr', (1-np.min(lc3d))/np.std(model-lc3d))
    ax[0].legend()
    ax[1].plot(frm, np.asarray(lc3d-lc2d), label="mean:"+str(round(np.sqrt(mn),6)))
    ax[1].fill_between(frm, np.sqrt(mn)*np.ones(len(lc3d)), -np.sqrt(mn)*np.ones(len(lc3d)), alpha=0.2)
    ax[1].set_xlabel('Phase')
    ax[1].set_ylabel('Flux')
    ax[0].set_ylabel('Flux')
    ax[0].set_title("$R_{pl}$ = 0.1 $R_{st}$, Orbit: = 2 $R_{st}$, u1: 0.6, u2:0.0, e: 0.0")
    ax[1].set_title('Residual')
    ax[1].legend()
    plt.suptitle('2D vs 3D transiting objects')
    df = pd.DataFrame(zip(frm, lc2d, lc2dstd, lc3d, lc3dstd), columns=['frame','2d','2dstd','3d','3dstd'])
    df.to_csv('2d3d_0.1R_limb_circ.csv', index='False', sep=',')
    #np.savetxt('2d3d_0.1R_circ.csv', np.transpose(np.array([frm, lc2d, lc2dstd, lc3d, lc3dstd])),delimiter=',', header='frame, 2d, 2dstd, 3d, 3dstd')
    plt.savefig('2d3d_0.1R_limb_circ.png')
    #plt.show()

def multishape():
    global fl
    global sides
    df = pd.DataFrame()
    

    plt.style.use('seaborn-bright')
    fig, ax = plt.subplots(1,1, figsize = (7,10), sharex=True)
    frm = np.linspace(-fl,fl, 500)
    df['frame']=frm

    for s in range(3,15):
        sides = s
    # start 4 worker processes
        with Pool(processes=40) as pool:
            lc2dsum = np.asarray(pool.map(shape_test, range(120)))
            lc2d = np.mean(lc2dsum, axis = 0)
            lc2dstd = np.sqrt(np.mean((lc2dsum-lc2d)**2, axis=0))
            print("--- %s min ---" % ((time.time() - start_time)/60))

            df['sd_'+str(s)]=lc2d
            df['std_sd_'+str(s)]=lc2dstd
        
        ax.plot(frm,lc2d, label=str(s))

    with Pool(processes=40) as pool:
        lc2dsum = np.asarray(pool.map(test_multi_loops_2d, range(120)))
        lc2d = np.mean(lc2dsum, axis = 0)
        lc2dstd = np.sqrt(np.mean((lc2dsum-lc2d)**2, axis=0))
        print("--- %s min ---" % ((time.time() - start_time)/60))

        df['sd_inf']=lc2d
        df['std_sd_inf']=lc2dstd

        ax.plot(frm,lc2d, label='inf')

    ax.legend()
    ax.set_xlabel('Phase')
    ax.set_ylabel('Flux')
    plt.suptitle('Multi Shape')
    df.to_csv('multishape.csv', index='False', sep=',')
    plt.savefig('multishape.png')

multishape()