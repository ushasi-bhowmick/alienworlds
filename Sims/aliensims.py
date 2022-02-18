import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy

#lets make classes in python... make it into C++ style but python flavor module that can be used anytime anywhere
#1st class to make a transiting object

#need to add limb darkening.... urgent!!!
#further avenues for exploration: kepler orbits, tilted orbits. for now we're sticking to circles.
#variable stars... eclipsing binaries... exomoons... etc. (after graduating probably)

class Path:
    def __init__(self, megnum, Rstar):
        self.traj = [[] for i in range(megnum)]
        self.Rstar = Rstar

    def add_frame(self,megs):
        for i in range(len(megs)):
            self.traj[i].append({'x':megs[i].Plcoords[:,0], 'y': megs[i].Plcoords[:,1], 'z':megs[i].Plcoords[:,2]})
            


class Megastructure:

    frame = 0

    def __init__(self, Rorb=1.0, iscircle = False, Rcircle = 1.0, isrot = False, ph_offset = 0.0, 
        o_vel = 1.0, elevation = 0.0, Plcoords=[]):
        self.iscircle = iscircle
        self.isrot =isrot

        self.Rorbit = Rorb #semi major axis of kepler orbit
        self.Rcircle = Rcircle
        self.ph_offset = ph_offset
        self.o_vel = o_vel
        self.elevation = elevation
        self.centre = np.zeros(3)

        #kepler orbits
        self.ecc = 0.0
        self.periapsis_offset = np.pi/2

        self.circ_res = 200

        if iscircle: 
            th = np.linspace(0, 2*np.pi, self.circ_res)
            self.Plcoords = np.transpose(np.asarray([Rcircle*np.cos(th), Rcircle*np.sin(th), np.zeros(self.circ_res)]))
        else: self.Plcoords = Plcoords #[(x,y,z),(x,y,z)...]

    def set_shape(self, Plcoords, iscircle, Rcircle):
        self.Plcoords = Plcoords
        self.iscircle = iscircle
        self.Rcircle = Rcircle
        if iscircle: 
            th = np.linspace(0, 2*np.pi, self.circ_res)
            self.Plcoords = np.transpose(np.asarray([Rcircle*np.cos(th), Rcircle*np.sin(th), np.zeros(self.circ_res)]))
        else: self.Plcoords = Plcoords #[(x,y,z),(x,y,z)...]

    def set_trajectory(self, Rorb, elevation, o_vel, phase_offset, isrot):
        self.Rorbit = Rorb
        self.elevation = elevation
        self.o_vel = o_vel
        self.phase_offset = phase_offset
        self.isrot = isrot

    def rotate(self, axis, L):
        
        if (axis=='x'): mat = np.array([[1,0,0],[0,np.cos(L),np.sin(L)],[0,-np.sin(L),np.cos(L)]])
        if (axis=='y'): mat = np.array([[np.cos(L),0,np.sin(L)],[0,1,0],[-np.sin(L),0,np.cos(L)]])
        if (axis=='z'): mat = np.array([[np.cos(L),np.sin(L),0],[-np.sin(L),np.cos(L),0],[0,0,1]])

        temp = np.asarray([np.matmul(mat,el) for el in self.Plcoords])
        
        return(temp)

    def translate(self, frm):
        kep_corr = self.Rorbit*(1-self.ecc**2)/(1+self.ecc*np.cos(self.o_vel*frm+self.ph_offset-self.periapsis_offset))
        xt = kep_corr*np.sin(self.o_vel*frm+self.ph_offset)
        zt = kep_corr*np.cos(self.o_vel*frm+self.ph_offset)
        yt = self.elevation
        temp = np.asarray([[xt+el[0], yt+el[1], zt+el[2]] for el in self.Plcoords])
        return(temp,np.asarray([self.centre[0]+xt, self.centre[1]+yt, self.centre[2]+zt]))


    def glp(self,frm):
        el = self.o_vel*frm + self.ph_offset 
        rem = np.floor(el*0.5/np.pi)
        return(el - rem*2*np.pi)



#2nd class to simulate a bunch of transits... taking in data from the first class.
class Simulator:
    def __init__(self, Rstar, no_pt, frame_no, frame_length = np.pi):
        self.megs = []

        self.lc = []

        self.Rstar = Rstar
        self.no_pt = no_pt
        self.frame_length = frame_length
        self.frames = self.frame_length*np.linspace(-1, 1, frame_no)
        self.tmegs = []

        self.ran_rad=self.Rstar*np.sqrt(np.random.rand(self.no_pt))
        self.ran_th=2*np.pi*np.random.rand(self.no_pt)

        self.road = Path(0, self.Rstar)

    def add_megs(self,meg):
        self.megs.append(meg)

    def reinitialize(self):
        self.ran_rad=self.Rstar*np.sqrt(np.random.rand(self.no_pt))
        self.ran_th=2*np.pi*np.random.rand(self.no_pt)
        self.lc=[]
        self.road = Path(len(self.megs), self.Rstar)


    def in_or_out(self,refx,refy, meg):
        shx = meg.Plcoords[:,0]
        shy = meg.Plcoords[:,1]
        #step 1: eliminate stuff outside the bounding box
        if(refx<min(shx) or refx>max(shx) or refy>max(shy) or refy<min(shy)): return(0)
        #step 2: ray tracing horizontal
        shy = np.append(shy,shy[0])
        shx = np.append(shx,shx[0])
        intsecty = (np.asarray([(shy[i]-refy)*(shy[i+1]-refy) if(shx[i]>refx) else 0 
            for i in range(0,len(shy)-1)])<0).sum()
        if(intsecty%2 !=0): return(1)
        else: return(0)

    def in_or_out_of_circle(self, refx, refy, meg):
        distarr = np.sqrt((refx-meg.centre[0])**2+ (refy-meg.centre[1])**2)<meg.Rcircle
        return(np.array(distarr, dtype = 'int'))

    #instead of coordinates these will work on elements of class megastructure
    def monte_carlo_multi(self, frame): 
        dists=[]

        for meg in self.tmegs:
            if(meg.glp(frame)>np.pi/2 and meg.glp(frame)<3*np.pi/2): dists.append(np.zeros(self.no_pt))

            else: 
                if(meg.iscircle and not meg.isrot):
                    distarr = self.in_or_out_of_circle(self.ran_rad*np.cos(self.ran_th), 
                        self.ran_rad*np.sin(self.ran_th),meg)
                else:
                    distarr=np.asarray([self.in_or_out(self.ran_rad[j]*np.cos(self.ran_th[j]),
                        self.ran_rad[j]*np.sin(self.ran_th[j]), meg) for j in range(self.no_pt)])
                dists.append(distarr)

        frac = np.sum(np.sum(np.asarray(dists), axis=0)>0)/self.no_pt
        return(frac)

    def simulate_transit(self):
        self.tmegs = copy.deepcopy(self.megs)
        self.lc = []
        self.road = Path(len(self.megs), self.Rstar)

        for frame in self.frames:
            for i in range(len(self.megs)):
                if self.megs[i].isrot: tcoordsh = self.megs[i].rotate('y', frame)
                else: tcoordsh = self.megs[i].Plcoords
                self.tmegs[i].Plcoords = tcoordsh
                self.tmegs[i].centre = self.megs[i].centre

                tcoordsh, cntr = self.tmegs[i].translate(frame)
                self.tmegs[i].centre = cntr
                self.tmegs[i].Plcoords = tcoordsh
            
            area = self.monte_carlo_multi(frame)
            self.road.add_frame(self.tmegs)
            self.lc.append(1-area)


#3rd class is the animation library containing modules to make the animation
class Transit_Animate:
    def __init__(self, road, megs, lc, phase):
        self.gopath = road
        self.lc = lc
        self.megs = megs
        self.phase = phase
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(7,7))
        self.ax1 = plt.subplot2grid((2, 2), (0, 0))
        self.ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        self.ax3 = plt.subplot2grid((2, 2), (0, 1))
        self.maxorb = 0

        self.th = np.linspace(0,2*np.pi, 200)
        self.ln, = plt.plot([], [], 'ro')

    def init_frame(self):
        self.ln.set_data([], [])
        self.ax2.set_xlabel('Phase')
        self.ax1.set_anchor('W')
        #self.ax1.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
        self.ax3.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
        self.ax2.set_ylabel('Flux')
        theta = np.arange(-2 * np.pi, 2 * np.pi+np.pi/2, step=(np.pi / 2))
        self.ax2.set_xticks(theta)
        self.ax2.set_xticklabels(['-2π', '-3π/2', '-π', '-π/2', '0', 'π/2', 'π', '3π/2', '2π'])
        self.ax2.set_xlim(min(self.phase),max(self.phase))
        self.ax2.set_ylim(min(self.lc)*0.99,1.01)
        plt.suptitle('Transit Simulations')

        t_rpls = [max(np.abs(el.Plcoords.reshape(-1)))/self.gopath.Rstar if(not el.iscircle) 
            else el.Rcircle/self.gopath.Rstar for el in self.megs]
        temp = np.array([[x['x'] for x in el] for el in self.gopath.traj])
        self.maxorb = max(np.abs(temp.reshape(-1)))
        t_orbs = [el.Rorbit for el in self.megs]
        t_offs = [round(el.ph_offset/np.pi,3) for el in self.megs]
        t_vels = [el.o_vel for el in self.megs]

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5, pad=1)
        txt = "$R_{pl}:$"+str(t_rpls)+"$R_{star}$\nOrbit:"+str(t_orbs)+"\nOffset:"+str(t_offs)+"$\pi$\nVelocity:"+str(t_vels)

        self.ax3.text(0.5, 0.5, txt, fontsize=9,transform=self.ax3.transAxes,  horizontalalignment='center',
            verticalalignment='center', linespacing=2, bbox=props, color='white')

        return self.ln,

    def update(self,frame):
        self.ax1.clear()
        zst=1
        zpl=[0 if np.all(el[frame]['z']<0) else 2 for el in self.gopath.traj]
        self.ax1.set_aspect(1)
        self.ax1.fill(self.gopath.Rstar*np.cos(self.th), self.gopath.Rstar*np.sin(self.th), zorder = zst, color='#fff44f')

        self.ax1.set_xlim(-self.maxorb*1.2,self.maxorb*1.2)
        self.ax1.set_ylim(-self.maxorb*1.2,self.maxorb*1.2)

        i=0
        for el in self.gopath.traj:
            self.ax1.fill(el[frame]['x'],el[frame]['y'], zorder=zpl[i], color='black', edgecolor='gray')
            i+=1
        self.ax2.scatter(self.phase[frame], self.lc[frame], color='red', marker='.')
        return self.ln,

    def go(self):
        ani = animation.FuncAnimation(self.fig, self.update, frames=np.arange(0,len(self.phase)), interval=1,init_func=self.init_frame)
        #writergif = animation.PillowWriter(fps=20) 
        plt.show()
        #ani.save('animation_dm_turn_5.gif', writer=writergif)


#4rth class for a plotting and saving data library

sim1 = Simulator(100, 1000, 100, np.pi)

th = np.linspace(0, 2*np.pi, 120)
Plcoord = np.array([[10*np.cos(el), 10*np.sin(el), 0] for el in th])

meg_2d = Megastructure(200, True, 10, isrot=True)
meg_3d = Megastructure(300, True, 10, elevation=0)
sim1.add_megs(meg_3d)
#sim1.add_megs(meg_2d)

sim1.simulate_transit()

#
TA = Transit_Animate(sim1.road, sim1.megs, sim1.lc, sim1.frames)
TA.go()
#plt.plot(sim1.lc)
#plt.plot(sim2.lc)
#plt.show()'''
