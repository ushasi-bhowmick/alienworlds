import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import aliensims as dysim
import time
import pandas as pd
from multiprocessing import Process, Pool

#gonna run this on hpc... will take weeks probably... hope not to drag it so far tho

#automatic iteration over:
#u1, u2, b
#manual iteration over:
#Rpl
#Rorb

#resolution setters
frame_res=50
mcmc_pts=500
frame_l=np.pi/3

#variable parameters
Rstar=100
Rpl=10
u1=0.1
u2=0.1
Rorb=300
b=0
ecc=0
per_off=0


start_time = time.time()

def test_multi_loops_3d(x):
    np.random.seed(1234*x)
    sim_3d = dysim.Simulator (100, 50000, 500, np.pi/3, limb_u1=0.0, limb_u2=0.0)
    meg_3d = dysim.Megastructure(200, True, 20, ecc=0.0)
    sim_3d.add_megs(meg_3d)
    sim_3d.simulate_transit()
    print("Count:", meg_3d.set, x)
    return(sim_3d.lc)

def test_multi_loops_2d(x):
    global Rpl
    global u1
    global u2
    global b
    np.random.seed(3456*x)
    sim_2d = dysim.Simulator (Rstar, mcmc_pts, frame_res, frame_l, limb_u1=u1, limb_u2=u2)
    meg_2d = dysim.Megastructure(Rorb, True, Rpl, isrot=True, ecc=ecc, per_off=per_off, incl=b)
    sim_2d.add_megs(meg_2d)
    if(x==0): print("Count:", meg_2d.set, ' u1:',u1,' u2:',u2, ' b:',b*180/np.pi)
    sim_2d.simulate_transit()
    return(sim_2d.lc)

frm = np.linspace(-frame_l, frame_l, frame_res)
for bss in [0,0.2,1]:
    b=np.arcsin(bss*Rstar/Rorb)
    
    for u1ss in [0,0.2,0.4]:
        plt.style.use('seaborn-bright')
        fig, ax = plt.subplots(1,1, figsize = (5,5))

        for u2ss in [0,0.2,0.4]:
            u1=u1ss
            u2=u2ss
            # start 4 worker processes
            with Pool(processes=4) as pool:
                lc2dsum = np.asarray(pool.map(test_multi_loops_2d, range(4)))
                lc2d = np.mean(lc2dsum, axis = 0)
                lc2dstd = np.sqrt(np.mean((lc2dsum-lc2d)**2, axis=0))
                print("--- %s min ---" % ((time.time() - start_time)/60))

            try: df = pd.read_csv('../Computation_Directory/2d_b'+str(bss)+'.csv', sep=',')
            except:
                df = pd.DataFrame(zip(frm, lc2d, lc2dstd), columns=['frame','u1_'+str(u1ss)+
                    'u2_'+str(u2ss),'std_u1_'+str(u1ss)+'u2_'+str(u2ss)])
                df.to_csv('../Computation_Directory/2d_b'+str(bss)+'.csv',index=False, sep=',')
                continue
            df['u1_'+str(u1ss)+'u2_'+str(u2ss)]=lc2d
            df['std_u1_'+str(u1ss)+'u2_'+str(u2ss)]=lc2dstd
            df.to_csv('../Computation_Directory/2d_b'+str(bss)+'.csv',index=False, sep=',')

            ax.plot(frm,lc2d,label = 'u2_'+str(u2ss))

        ax.legend()
        ax.set_xlabel('Phase')
        ax.set_ylabel('Flux')
        ax.set_title("$R_{pl}$ = 0.2 $R_{st}$, Orbit: = 2 $R_{st}$, e: 0.0, b:"+str(bss))
        plt.savefig('../Computation_Directory/pl_b'+str(bss)+'u1'+str(u1ss)+'.png')
        plt.close()

        
    
        #plt.show()
