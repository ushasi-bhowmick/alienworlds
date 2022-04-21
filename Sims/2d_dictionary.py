from codecs import replace_errors
import random
from traceback import print_tb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import aliensims as dysim
import time
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process, Pool, Manager

#gonna run this on hpc... will take weeks probably... hope not to drag it so far tho

#automatic iteration over:
#u1, u2, b
#manual iteration over:
#Rpl
#Rorb
rpl_arr=np.around(np.linspace(0.01,0.5, 10) ,2)
print(rpl_arr)
rorb_arr=np.around(np.logspace(0.31,3,10), 2)

#resolution setters
frame_res=300
mcmc_pts=3000

man = Manager()
 
frame_l=man.list()

#variable parameters
# Rpl: 1, 5, 10, 30, 50
# Rorb: 2, 4, 16, 64, 128
Rpl=100*rpl_arr[3]

Rorb=200
u1=0.1
u2=0.1
Rstar=100
b=0
ecc=0
per_off=0 

if not os.path.exists('../Computation_Directory/Rpl_'+str(np.around(Rpl,2))):
    os.mkdir('../Computation_Directory/Rpl_'+str(np.around(Rpl,2)))
    print("Directory Created ")

start_time = time.time()

def test_multi_loops_2d(x):  
    global Rpl
    
    global u1
    global u2
    global b
    np.random.seed(3456*x)
    sim_2d = dysim.Simulator (Rstar, mcmc_pts, frame_res, limb_u1=u1, limb_u2=u2)
    meg_2d = dysim.Megastructure(Rorb, True, Rpl, isrot=True, ecc=ecc, per_off=per_off, incl=b)
    sim_2d.add_megs(meg_2d)
    sim_2d.set_frame_length()
    frame_l.append(sim_2d.frame_length)
    #if(x==0): print("Count:", meg_2d.set, ' u1:',u1,' u2:',u2, ' b:',np.around(b*180/np.pi,2), 'rorb:',Rorb)
    sim_2d.simulate_transit()  
    return(sim_2d.lc) 

for r in rorb_arr[1:]:
    Rorb=r*100
    global frm
    for u1ss in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        plt.style.use('seaborn-bright') 
        fig, ax = plt.subplots(1,1, figsize = (10,10))
        for bss in [0,0.2,0.4,0.6,0.8]:
            b=np.arcsin(bss*Rstar/Rorb)
            
            for u2ss in [0,0.2,0.4]: 
                u1=u1ss
                u2=u2ss  
                frame_l[:]=[]
                # start 4 worker processes
                with Pool(processes=40) as pool:
                    lc2dsum = np.asarray(pool.map(test_multi_loops_2d, range(80)))
                    lc2d = np.mean(lc2dsum, axis = 0)
                    lc2dstd = np.sqrt(np.mean((lc2dsum-lc2d)**2, axis=0))
                    #print("--- %s min ---" %((time.time() - start_time)/60))

                
                frm = np.linspace(-frame_l[0], frame_l[0], frame_res)
                ax.plot(frm,lc2d,label = 'u2_'+str(u2ss)+'b_'+str(bss))
                
                try: df = pd.read_csv('../Computation_Directory/Rpl_'+str(np.around(Rpl,2))+'/2d_rorb_'+str(r)+'.csv', sep=',')
                except:
                    df = pd.DataFrame(zip(frm, lc2d, lc2dstd), columns=['frame','u1_'+str(u1ss)+
                        '_u2_'+str(u2ss)+'_b_'+str(bss),'std_u1_'+str(u1ss)+'_u2_'+str(u2ss)+'_b_'+str(bss)])
                    df.to_csv('../Computation_Directory/Rpl_'+str(np.around(Rpl,2))+'/2d_rorb_'+str(r)+'.csv',index=False, sep=',')
                    continue
                df['u1_'+str(u1ss)+'_u2_'+str(u2ss)+'_b_'+str(bss)]=lc2d
                df['std_u1_'+str(u1ss)+'_u2_'+str(u2ss)+'_b_'+str(bss)]=lc2dstd
                df.to_csv('../Computation_Directory/Rpl_'+str(np.around(Rpl,2))+'/2d_rorb_'+str(r)+'.csv',index=False, sep=',')

            print("Count:", ' u1:',u1, ' b:',np.around(b*180/np.pi,2), 'rorb:',Rorb)
            print("--- %s min ---" %((time.time() - start_time)/60))    
  
        ax.legend()
        ax.set_xlabel('Phase')
        ax.set_ylabel('Flux')
        ax.set_title("$R_{pl}$ ="+str(np.around(Rpl,2))+" $R_{st}$, Orbit: = "+str(r)+" $R_{st}$, e: 0.0, u1:"+str(u1ss))
        plt.savefig('../Computation_Directory/Rpl_'+str(np.around(Rpl,2))+'/u1_'+str(u1ss)+'_rorb_'+str(r)+'.png')
        plt.close()

        
    
        #plt.show()
