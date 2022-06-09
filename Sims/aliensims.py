
from xml.etree.ElementInclude import include
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import root_scalar
import copy
import time
from gfg import in_or_out

"""TRANSIT SIMULATION MODULE
Simulated lightcurves generated for anything and everything going around a star of your choice.
Just set parameters and go!

EXAMPLE:
    Just follow a couple of easy steps.
    Set up a Simulator

        :: sim = Simulator(star_radius, resolution, number_of_frames, length of frame)

    Set up a Megastructure

        :: meg = Megastructure(coordinates)
    
    Add one or more megastructures to the simulator

        ::sim.add_megs(meg)

    And go!

        ::sim.simulate_transit()

DESCRIPTION:
    Megastructure Module:
        This module stores the megastructure. For circles, and basic shapes, I've got it covered, but if you
        got some cool alien structures in mind, feel free to plug in the coordinates. Also set some basic 
        parameters about the trajectory, and locations and velocities in case you got multiple megastructures
        to simulate together.

    Simulator Module:
        This module contains parameters for the so-called host star, and a number of nitty-gritties about the 
        simulator. The simulator is a really basic monte-carlo simulation which numerically calculates the area
        of overlap between transiting object and host.

    Animation Module:
        Just a quick logic to create animated view of transiting lightcurves for better visuals. Feel
        free to create your own visuals, using the Path object returned by simulator, which has everything 
        you need about the trajectory and coordinates. Cheers!

ATTRIBUTES:
    Megastructure module:

    Simulator module:

    Animation module:
"""

# lets make classes in python... make it into C++ style but python flavor module that can be used anytime anywhere
# 1st class to make a transiting object

# further avenues for exploration: tidal distortions, gravity darkening
# tilted orbits made looking at impact parameter concerns. like tilted along direction of line of site...
# 
#variable stars... eclipsing binaries... exomoons...rings etc. (after graduating probably)

class Path:
    def __init__(self, megnum, Rstar):
        self.traj = [[] for i in range(megnum)]
        self.Rstar = Rstar
        self.centres = [[] for _ in range(megnum)]
        self.MCscatter_x = []
        self.MCscatter_y = []

    def add_random_no(self,x,y):
        self.MCscatter_x=x
        self.MCscatter_y=y

    def add_frame(self,megs):
        for i in range(len(megs)):
            self.traj[i].append({'x':megs[i].Plcoords[:,0], 'y': megs[i].Plcoords[:,1], 'z':megs[i].Plcoords[:,2]})
            self.centres[i].append(megs[i].centre)

#add some nesting to incorporate various features
#need to add: circles, polygons, holed geometries, stars.

class Megastructure:

    set = 0

    def __init__(self, Rorb=1.0, iscircle = False, Rcircle = 1.0, isrot = False, ph_offset = 0.0, 
        o_vel = 1.0, elevation = 0.0, Plcoords=[], ecc=0.0, per_off=0.0, incl=0.0):

        """Megastructure Module

        Create a megastructure to transit around a star. 
        :param Rorb: Radius of the orbit. For eccentric orbits put semi-major axis
        :param iscircle: If transiting object is circular/ spherical (simpler logic thats why)
        :param Rcircle: Radius of circle in case its circular
        :param isrot: Is the object rotating? 
        :param ph_offset: Phase offset at init time of simulation (useful for multi-transit stuff)
        :param o_vel: relative orbital velocity(for multiple structures)
        :param elevation: height above the centre line of the orbit
        :param Plcoords: coordinates for complex geometries [(x,y,z),(x,y,z)...]
        :param ecc: eccentricity if kepler orbit (0 for circular)
        :param per_off: periapsis offset. Default closest approach is behind the star
        :param incl: inclination of the orbit with respect to the line of site... like impact parameter

        Others.
        :param rot_vel: rotation velocity for rotating objects (1 means same as orbital velocity)
        :param rot_axis: axis of rotation of rotating objects
        :param centre: centre of the shape, a must-do for circular stuff
        :param circ_res: resolution with which we draw the circle
        """

        Megastructure.set+=1
        self.iscircle = iscircle
        self.isrot = isrot
        self.rot_axis = [0,1,0]
        self.rot_vel = 1

        self.Rorbit = Rorb #semi major axis of kepler orbit
        self.Rcircle = Rcircle
        self.ph_offset = ph_offset
        self.o_vel = o_vel
        self.elevation = elevation
        self.incl = incl
        self.centre = np.zeros(3)

        #kepler orbits
        self.ecc = ecc
        self.periapsis_offset = per_off

        self.circ_res = 200

        if iscircle: 
            th = np.linspace(0, 2*np.pi, self.circ_res)
            self.Plcoords = np.transpose(np.asarray([Rcircle*np.cos(th), Rcircle*np.sin(th), np.zeros(self.circ_res)]))
        else: self.Plcoords = Plcoords #[(x,y,z),(x,y,z)...]

    def regular_polygons_2d(self, rad, no_of_sides):
        """Generate regular polygons like triangles, hexagons...

        :param rad: radius of enclosing circle of polygon
        :param no_of_sides: how many sided polygon is needed?
        
        """
        ang = 2*np.pi/no_of_sides
        coord = []
        for i in range(no_of_sides):
            x = rad*np.cos(ang*i)
            y = rad*np.sin(ang*i)
            coord.append([x,y,0])
        self.Plcoords = np.array(coord)
        return(np.array(coord))

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
        """Rotate the Plcoords about an axis of rotation by some angle

        :param axis: axis of rotation [x,y,z]
        :param L: the angle with which it should be rotated
        
        """
        n1 = axis[0]
        n2 = axis[1]
        n3 = axis[2]
        mat = np.array([[np.cos(L)+n1**2*(1-np.cos(L)), n1*n2*(1-np.cos(L))-n3*np.sin(L), n1*n3*(1-np.cos(L))+n2*np.sin(L)],
                        [n1*n2*(1-np.cos(L))+n3*np.sin(L), np.cos(L)+n2**2*(1-np.cos(L)), n2*n3*(1-np.cos(L))-n1*np.sin(L)],
                        [n1*n3*(1-np.cos(L))-n2*np.sin(L), n2*n3*(1-np.cos(L))+n1*np.sin(L), np.cos(L)+n3**2*(1-np.cos(L))]])

        temp = np.asarray([np.matmul(mat,el) for el in self.Plcoords])

        return(temp)

    def translate(self, frm):
        """ translate Plcoords by some distance determined by the angle
        """
        kep_corr = self.Rorbit*(1-self.ecc**2)/(1+self.ecc*np.cos(self.o_vel*frm+self.ph_offset-self.periapsis_offset))
        xt = kep_corr*np.sin(self.o_vel*frm+self.ph_offset)
        zt = kep_corr*np.cos(self.incl)*np.cos(self.o_vel*frm+self.ph_offset)
        yt = self.elevation+kep_corr*np.sin(self.incl)*np.cos(self.o_vel*frm+self.ph_offset)
        temp = np.asarray([[xt+el[0], yt+el[1], zt+el[2]] for el in self.Plcoords])
        return(temp,np.asarray([self.centre[0]+xt, self.centre[1]+yt, self.centre[2]+zt]))


    def glp(self,frm):
        el = self.o_vel*frm + self.ph_offset 
        rem = np.floor(el*0.5/np.pi)
        return(el - rem*2*np.pi)



#2nd class to simulate a bunch of transits... taking in data from the first class.
class Simulator:
    def __init__(self, Rstar, no_pt, frame_no, frame_length = np.pi, limb_u1=0.0, limb_u2=0.0):

        """Simulator Module

        A Simulator object means a star system with multiple megastructures used to simulate a transit
        :param Rstar: radius of star
        :param no_pt: number of points to put in monte-carlo sim
        :param frame_no: how many points should the trajectory contain
        :param frame_length: length of trajectory in terms of phase, default (-pi,pi)
        :param limb_u1: limb darkening coefficient 1
        :param limb_u2: limb darkening coefficient 2

        Others
        :param megs: list of all the megastructures imported
        :param lc: the final lightcurve from one simulation run
        :param road: history of the object
        
        """
        self.megs = []

        self.lc = []

        self.Rstar = Rstar
        self.no_pt = no_pt
        self.frame_length = frame_length
        self.frame_no = frame_no
        self.frames = self.frame_length*np.linspace(-1, 1, frame_no)
        self.tmegs = []

        #limb darkening coefficient
        self.lmb_u = limb_u1
        self.lmb_v = limb_u2

        self.ran_rad=[]
        self.ran_th=[]

        self.road = Path(0, self.Rstar)

        self.initialize()

    def Prob(self, x, z, u, v):
        a=1
        #R=1
        #k=(1-u)*R+u*np.pi*R/4
        #y = ((1-u)*x + u*x*np.sqrt(R**2-x**2)/2*R + u*R*np.arcsin(x/R)/2)/k -z
        k = (1-u-5*v/3)*a + (u+2*v)*np.pi*a/4
        y = ((1-u-2*v)*x + v*x**3/(3*a**2) + ((u+2*v)/a)*(x*np.sqrt(a**2 - x**2)/2 + a**2*np.arcsin(x/a)/2))/k - z
        return(y)

    def add_megs(self,meg):

        """Add a megastructure object to the simulator
        """

        self.megs.append(meg)

    def set_frame_length(self):

        """Let us decide an optimum frame length for the code!
        """

        Rorb=[meg.Rorbit for meg in self.megs]
        n = np.floor(np.pi*max(Rorb)/(2*self.Rstar))
        self.frame_length = np.pi/n
        self.frames = self.frame_length*np.linspace(-1, 1, self.frame_no)
        #print("Frame Length:"+str(n))

    def initialize(self):

        """Empty the history and lightcurve variables and redo the random numbers
        """

        #self.ran_rad=self.Rstar*np.power(np.random.rand(self.no_pt),(1/2))
        self.ran_th=2*np.pi*np.random.rand(self.no_pt)
        self.lc=[]
        ran_x = [] 
        for el in np.sqrt(np.random.rand(self.no_pt)):
            sol = root_scalar(self.Prob,args=(el,self.lmb_u, self.lmb_v),bracket=[0,1])
            ran_x.append(sol.root)
        self.ran_rad = self.Rstar*np.array(ran_x)
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
        
        #start_time=time.time()
        for meg in self.tmegs:
            if(meg.glp(frame)>np.pi/2 and meg.glp(frame)<3*np.pi/2): dists.append(np.zeros(self.no_pt))

            else: 
                if(meg.iscircle and not meg.isrot):
                    distarr = self.in_or_out_of_circle(self.ran_rad*np.sin(self.ran_th), 
                        self.ran_rad*np.cos(self.ran_th),meg)
                else:
                    distarr=np.asarray(in_or_out(self.ran_rad*np.sin(self.ran_th),self.ran_rad*np.cos(self.ran_th),
                        meg.Plcoords[:,0], meg.Plcoords[:,1]))
                    # distarr=np.asarray([self.in_or_out(self.ran_rad[j]*np.sin(self.ran_th[j]),
                    #     self.ran_rad[j]*np.cos(self.ran_th[j]), meg) for j in range(self.no_pt)])
                dists.append(distarr)

        #print("one: ",time.time()-start_time)
        frac = np.sum(np.sum(np.asarray(dists), axis=0)>0)/self.no_pt
        return(frac)

    def simulate_transit(self):
        
        """Run one iteration of the simulation
        """

        self.tmegs = copy.deepcopy(self.megs)
        self.lc = []
        self.road = Path(len(self.megs), self.Rstar)

        self.road.add_random_no(self.ran_rad*np.sin(self.ran_th), self.ran_rad*np.cos(self.ran_th))

        for frame in self.frames:
            for i in range(len(self.megs)):
                if self.megs[i].isrot: 
                    tcoordsh = self.megs[i].rotate(self.megs[i].rot_axis, self.megs[i].rot_vel*frame+self.megs[i].ph_offset)
        
                else: tcoordsh = self.megs[i].Plcoords
                self.tmegs[i].Plcoords = tcoordsh
                self.tmegs[i].centre = self.megs[i].centre
                
                if (self.megs[i].incl and self.megs[i].isrot): 
                    tcoordsh2, cntr = self.tmegs[i].translate(frame)
                    ang = np.arcsin(cntr[1]/self.tmegs[i].Rorbit)
                    #print(ang*180/np.pi)
                    tcoordsh = self.tmegs[i].rotate([1,0,0], -np.abs(ang))
                    self.tmegs[i].Plcoords = tcoordsh

                tcoordsh2, cntr = self.tmegs[i].translate(frame)
                self.tmegs[i].centre = cntr
                self.tmegs[i].Plcoords = tcoordsh2
            
            area = self.monte_carlo_multi(frame)
            self.road.add_frame(self.tmegs)
            self.lc.append(1-area)

        return(self.road, self.frames, self.lc)


#3rd class is the animation library containing modules to make the animation
class Transit_Animate:
    def __init__(self, road, megs, lc, phase):
        self.gopath = road
        self.lc = lc
        self.megs = megs
        self.phase = phase
        plt.style.use('seaborn-bright')
        plt.rcParams["font.family"] = "serif"
        self.fig = plt.figure(figsize=(7,7))
        self.fig.patch.set_facecolor('#CCCCCC')
        self.ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=2)
        self.ax2 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        self.ax3 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
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
        theta = np.arange(-2 * np.pi, 2 * np.pi+np.pi/4, step=(np.pi / 4))
        self.ax2.set_xticks(theta)
        self.ax2.set_xticklabels(['-2π', '-7π/4', '-3π/2', '-5π/4', '-π', '-3π/4', '-π/2', '-π/4', '0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '3π/2', '7π/4', '2π'])
        self.ax2.set_xlim(min(self.phase),max(self.phase))
        self.ax2.set_ylim(min(self.lc)*0.99,1.01)
        plt.suptitle('Transit Simulations')

        self.ax1.spines['top'].set_visible(False)
        self.ax1.spines['right'].set_visible(False)
        self.ax1.spines['bottom'].set_visible(False)
        self.ax1.spines['left'].set_visible(False)

        self.ax3.spines['top'].set_visible(False)
        self.ax3.spines['right'].set_visible(False)
        self.ax3.spines['bottom'].set_visible(False)
        self.ax3.spines['left'].set_visible(False)

        t_rpls = [max(np.abs(el.Plcoords.reshape(-1)))/self.gopath.Rstar if(not el.iscircle) 
            else el.Rcircle/self.gopath.Rstar for el in self.megs]
        temp = np.array([[x['x'] for x in el] for el in self.gopath.traj])
        self.maxorb = max(np.abs(temp.reshape(-1)))
        t_orbs = [el.Rorbit/self.gopath.Rstar for el in self.megs]
        t_offs = [round(el.ph_offset/np.pi,3) for el in self.megs]
        t_vels = [el.o_vel for el in self.megs]
        t_incl = [round(el.incl*180/np.pi,2) for el in self.megs]

        props = dict(boxstyle='round', facecolor='#dddddd', alpha=0.5, pad=1)
        txt = "$R_{pl}:$\n"+str(np.round(t_rpls,3)[:3])+"$R_{star}$\nOrbit:\n"+str(np.round(t_orbs,
            3)[:3])+"$R_{star}$\nOffset:\n"+str(t_offs[:3])+"$\pi$\nVelocity:\n"+str(t_vels[:3]
            )+"$\pi$\nIncl:\n"+str(t_incl[:3])

        self.ax3.text(0.5, 0.5, txt, fontsize=9,transform=self.ax3.transAxes,  horizontalalignment='center',
            verticalalignment='center', linespacing=2, bbox=props, color='black')

        return self.ln,

    def update(self,frame):
        self.ax1.clear()
        zst=1
        zpl=[0 if np.all(el[frame]['z']<0) else 2 for el in self.gopath.traj]
        self.ax1.set_aspect(1)
        self.ax1.fill(self.gopath.Rstar*np.cos(self.th), self.gopath.Rstar*np.sin(self.th), zorder = zst, color='#ffae42')

        self.ax1.set_xlim(-self.maxorb*1.2,self.maxorb*1.2)
        self.ax1.set_ylim(-self.maxorb*1.2,self.maxorb*1.2)
        self.ax1.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
        i=0
        for el in self.gopath.traj:
            self.ax1.fill(el[frame]['x'],el[frame]['y'], zorder=zpl[i], color='black', edgecolor='gray')
            i+=1
        # for el in self.gopath.centres:
        #     el=np.asarray(el)
        #     a=el[:,0]*(el[:,2]>0)
        #     b=el[:,1]*(el[:,2]>0)
        #     self.ax1.scatter(a,b,marker='.',s=1, zorder=5, color='red')
        #     a=el[:,0]*(el[:,2]<0)
        #     b=el[:,1]*(el[:,2]<0)
        #     self.ax1.scatter(a,b,marker='.',s=1, zorder=0, color='red')
        self.ax2.scatter(self.phase[frame], self.lc[frame], color='red', marker='.')
        return self.ln,

    def go(self,ifsave=False,filepath=""):
        ani = animation.FuncAnimation(self.fig, self.update, frames=np.arange(0,len(self.phase)), interval=1,init_func=self.init_frame)
        if(ifsave):
            print('here', filepath)
            writergif = animation.PillowWriter(fps=20) 
            ani.save(filepath, writer=writergif, savefig_kwargs=dict(facecolor='#CCCCCC'))
            plt.show()
        else: plt.show()


# 4rth class for a plotting and saving data library

# sim1 = Simulator(100, 5000, 100, np.pi/3)
# # # sim2 = Simulator(100, 1000, 100, np.pi)

# meg_2d = Megastructure(300, True, 40, isrot=True, incl=20*np.pi/180, ph_offset=0, elevation=0, ecc=0, per_off=np.pi/2)
# # # meg_2d2 = Megastructure(120, True, 20, isrot=True, incl=0, ph_offset=0, elevation=0)

# sim1.add_megs(meg_2d)
# sim1.simulate_transit()
# # # sim2.add_megs(meg_2d2)
# # # sim2.simulate_transit()


# TA = Transit_Animate(sim1.road, sim1.megs, sim1.lc, sim1.frames)
# TA.go()
# plt.plot(sim1.lc)
# # # plt.plot(sim2.lc)
# plt.show()
