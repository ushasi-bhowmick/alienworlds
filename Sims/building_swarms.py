import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import aliensims as dy
import time
from multiprocessing import Process, Pool

#vertical swarms: polar clark belt
#lets aim to cover the star more or less...
n = 15
Rst = 100
Rorb =200
rinp = np.pi*Rorb/(n*np.sqrt(2))
print(rinp)
sim1 = dy.Simulator(Rst, 1000, 400, np.pi)

'''for i in range(1,int(n/2)+1):
    meg = dy.Megastructure(Rorb*np.cos(i*np.pi/n), False, isrot=True, elevation=Rorb*np.sin(i*np.pi/n))
    meg.Plcoords = meg.regular_polygons_2d(rinp, 4)
    meg.Plcoords=meg.rotate([0,0,1],np.pi/4)
    meg.Plcoords=meg.rotate([1,0,0],-i*np.pi/n)
    sim1.add_megs(meg)

for i in range(1,int(n/2)+1):
    meg = dy.Megastructure(Rorb*np.cos(i*np.pi/n), False, isrot=True, elevation=-Rorb*np.sin(i*np.pi/n))
    meg.Plcoords = meg.regular_polygons_2d(rinp, 4)
    meg.Plcoords=meg.rotate([0,0,1],np.pi/4)
    meg.Plcoords=meg.rotate([1,0,0],i*np.pi/n)
    sim1.add_megs(meg)

for i in range(1,int(n/2)+1):
    meg = dy.Megastructure(Rorb*np.cos(i*np.pi/n), False, isrot=True, elevation=Rorb*np.sin(i*np.pi/n), ph_offset=np.pi)
    meg.Plcoords = meg.regular_polygons_2d(rinp, 4)
    meg.Plcoords=meg.rotate([0,0,1],np.pi/4)
    meg.Plcoords=meg.rotate([1,0,0],i*np.pi/n)
    sim1.add_megs(meg)

for i in range(1,int(n/2)+1):
    meg = dy.Megastructure(Rorb*np.cos(i*np.pi/n), False, isrot=True, elevation=-Rorb*np.sin(i*np.pi/n), ph_offset=np.pi)
    meg.Plcoords = meg.regular_polygons_2d(rinp, 4)
    meg.Plcoords=meg.rotate([0,0,1],np.pi/4)
    meg.Plcoords=meg.rotate([1,0,0],-i*np.pi/n)
    sim1.add_megs(meg)'''

for j in range(-int(n/2),int(n/2)+1):
    newn = int(np.pi*Rorb*np.cos(j*np.pi/n)/(rinp*np.sqrt(2)))
    if(j==0): newn=n
    print(newn)
    for i in range(0,2*int(n)):
        meg = dy.Megastructure(Rorb*np.cos(j*np.pi/n), False, isrot=True,elevation=Rorb*np.sin(j*np.pi/n), ph_offset=i*np.pi/newn)
        meg.Plcoords = meg.regular_polygons_2d(rinp, 4)
        meg.Plcoords=meg.rotate([0,0,1],np.pi/4)
        meg.Plcoords=meg.rotate([1,0,0],-j*np.pi/n)
        meg.Plcoords=meg.rotate([0,1,0],i*np.pi/newn)
        sim1.add_megs(meg)

'''newn = int(np.pi*Rorb*np.cos(np.pi/n)/(rinp*np.sqrt(2)))
print(newn)
for i in range(0,2*int(newn)):
    meg = dy.Megastructure(Rorb*np.cos(np.pi/n), False, isrot=True,elevation=Rorb*np.sin(np.pi/n), ph_offset=i*np.pi/newn)
    meg.Plcoords = meg.regular_polygons_2d(rinp, 4)
    meg.Plcoords=meg.rotate([0,0,1],np.pi/4)
    meg.Plcoords=meg.rotate([1,0,0],-np.pi/n)
    meg.Plcoords=meg.rotate([0,1,0],i*np.pi/newn)
    sim1.add_megs(meg)

newn = int(np.pi*Rorb*np.cos(np.pi/n)/(rinp*np.sqrt(2)))
print(newn)
for i in range(0,2*int(newn)):
    meg = dy.Megastructure(Rorb*np.cos(np.pi/n), False, isrot=True,elevation=-Rorb*np.sin(np.pi/n), ph_offset=i*np.pi/newn)
    meg.Plcoords = meg.regular_polygons_2d(rinp, 4)
    meg.Plcoords=meg.rotate([0,0,1],np.pi/4)
    meg.Plcoords=meg.rotate([1,0,0],np.pi/n)
    meg.Plcoords=meg.rotate([0,1,0],i*np.pi/newn)
    sim1.add_megs(meg)

newn = int(np.pi*Rorb*np.cos(2*np.pi/n)/(rinp*np.sqrt(2)))
print(newn)
for i in range(0,2*int(newn)):
    meg = dy.Megastructure(Rorb*np.cos(2*np.pi/n), False, isrot=True,elevation=Rorb*np.sin(2*np.pi/n), ph_offset=i*np.pi/newn)
    meg.Plcoords = meg.regular_polygons_2d(rinp, 4)
    meg.Plcoords=meg.rotate([0,0,1],np.pi/4)
    meg.Plcoords=meg.rotate([1,0,0],-2*np.pi/n)
    meg.Plcoords=meg.rotate([0,1,0],i*np.pi/newn)
    sim1.add_megs(meg)

newn = int(np.pi*Rorb*np.cos(3*np.pi/n)/(rinp*np.sqrt(2)))
print(newn)
for i in range(0,2*int(newn)):
    meg = dy.Megastructure(Rorb*np.cos(3*np.pi/n), False, isrot=True,elevation=Rorb*np.sin(3*np.pi/n), ph_offset=i*np.pi/newn)
    meg.Plcoords = meg.regular_polygons_2d(rinp, 4)
    meg.Plcoords=meg.rotate([0,0,1],np.pi/4)
    meg.Plcoords=meg.rotate([1,0,0],-3*np.pi/n)
    meg.Plcoords=meg.rotate([0,1,0],i*np.pi/newn)
    sim1.add_megs(meg)'''

sim1.simulate_transit()

#
TA = dy.Transit_Animate(sim1.road, sim1.megs, sim1.lc, sim1.frames)
TA.go()
#plt.plot(sim1.lc)
#plt.plot(sim2.lc)
#plt.show()