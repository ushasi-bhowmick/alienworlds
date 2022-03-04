import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import aliensims as dy
import time
from multiprocessing import Process, Pool

start_time = time.time()

#vertical swarms: polar clark belt
#lets aim to cover the star more or less...
n = 21
phi = 2*np.pi/n

Rst = 100
Rorb =200
a = 2*np.pi*Rorb/n
rinp = np.pi*Rorb/(n*np.sqrt(2))
print(rinp)
sim1 = dy.Simulator(Rst, 1000, 100, np.pi+1)

for j in range(-int(n/4),int(n/4)+1):
    for i in range(0,n):
        #print((j*phi+phi/2)/np.pi)
        a1=2*np.pi*Rorb*np.cos(j*phi+phi/2)/n
        a2=2*np.pi*Rorb*np.cos(j*phi-phi/2)/n
        a=2*np.pi*Rorb*np.cos(j*phi)/n
        #print(a1,a2)
        #coords=np.array([[a/2,a/2,0],[-a/2,a/2,0],[-a/2,-a/2,0],[a/2,-a/2,0]])
        coords=np.array([[a1/2,a/2,0],[-a1/2,a/2,0],[-a2/2,-a/2,0],[a2/2,-a/2,0]])
        meg = dy.Megastructure(Rorb*np.cos(j*phi), False, isrot=True,elevation=Rorb*np.sin(j*phi), Plcoords=coords, ph_offset=i*phi)
        
        meg.Plcoords=meg.rotate([1,0,0],-j*phi)
        meg.Plcoords=meg.rotate([0,1,0],i*phi)
        sim1.add_megs(meg)
#print(len(sim1.megs))

sim1.simulate_transit()

print("--- %s seconds ---" % (time.time() - start_time))

TA = dy.Transit_Animate(sim1.road, sim1.megs, sim1.lc, sim1.frames)
TA.go()
#plt.plot(sim1.lc)
#plt.plot(sim2.lc)
#plt.show()