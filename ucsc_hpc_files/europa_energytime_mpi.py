# Obtain dynamical tides in an ocean with varying forcing frequency in eccentricity tides.
#
# Guideline:
# https://computationalmechanics.in/parallelizing-for-loop-in-python-with-mpi/
#
# Guidelines on Reduce: 
# https://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/
#
# Simple run: 
# mpirun -np 6 python3 titan_omvar_mpi.py
#
# 03/2023
# Ben Idini

import numpy as np
from mpi4py import MPI
import time
import pdb
import dill
import math
import scipy.integrate as integrate
import scipy.special as special

from interiorize.solvers import cheby
from interiorize.hough import dynamical
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

G   = 6.67e-8
label = 'g6_tau9_mpi'
# Titan
R       = 1561e5    # Mean radius (cm)
rhom    = 3.04     # Mean density (g/cc)
ms      = 4.8e25    # mass (g)
Ts      = 3.5*24 # orbital period (hours)
e       = 0.009     # Eccentricity

# Jupiter 
Ms = 1.898e30
Rs = 6.99e9

# Model parameters
rhow    = 1.
tau     = 1e8       # frictional dissipation
H	= 150e5
N2	= 1e-6*0

# Chebyshev solver
N = 80         # number of Chebyshev polynomials
Lmax = 80
M = 2

# Initial calculations
L       = np.arange(abs(M), Lmax+1)
Rp      = R             # body radius
eta	= (R-H)/R
Rc	= R*eta
a	= np.pi*eta
rhoc	= 3*ms/(4*np.pi*Rc**3) - rhow*((R/Rc)**3 - 1)
b       = np.pi
t_step = 500

# Varying om calculations (orbital frequency)omeg

def sma():
    k2 = .6
    m1 = 8.931e25 #Mass of Io in grams
    Rj = 6.99e9
    Q = 10e4
    G = 6.67e-8
    M = 1.898e30 #Mass of Jupiter in grams
    Om = 2*np.pi/(10*3600) #Orbital frequency of Jupiter in hours
    om1 = 2*np.pi/(1.77*24*3600) #Orbital frequency of Io in hours
    m2 = 4.7998e25 #Mass of Europa in grams
    m3 = 1.4819e26 #Mass of Ganymede in grams
    t = np.linspace(1, 4e9*3e7, t_step)
    t_not = 0
    a_not = 4*Rj 

    C1 = ((3*k2*m1*Rj**5)/Q)*((G/M)**(1/2))
    C2_top = (Om/om1) - 1
    C2_bottom = (Om/om1) * (1 + ((2**(1/3) * m2 )/ m1) + ((2**(2/3) * m3 )/ m1)) - 1 - ((4**(-1/3) * m2)/m1) - ((16**(-1/3)*m3)/m1)
    C = C1 * (C2_top/C2_bottom)

    a1 = ((13/2)*(C*t+((2/13)*(a_not**(13/2)))))**(2/13)

    a2 = 4**(1/3)*a1

    a3 = 16**(1/3)*a1

    return a2

semi = sma()

bottom = (((4*np.pi)**2)/(G*Ms))*(sma()**3)

omega = ((2*np.pi)/(np.sqrt(bottom)))

oms_vec = omega

Om_vec = oms_vec  # rotational frequency in synchronous rotation
ome_vec = oms_vec # eccentricity tidal frequency
k2_vec = np.zeros(len(oms_vec))
E_vec = np.zeros(len(oms_vec))

# Varying 

# Main calculation
def main_calc():
    
    cheb = cheby(npoints=N, loend=a, upend=b)

    dyn = dynamical(cheb, ome_vec[ind], Om_vec[ind]*0, ms, Ms, semi[ind], Rp,
                 rho=rhow, rhoc=rhoc, Rc=Rc,
                 l=L, m=M, tau=tau, x1=a, x2=b,
                 tides='ee', e=e, N2=N2)
    
    dyn.solve(kind='bvp-core')

    return dyn.k[0], sum(dyn.E)

# MPI 
num_per_rank = t_step//size

oms_lower_bound = rank*num_per_rank 
oms_upper_bound = (rank + 1)*num_per_rank

print("Processor ", rank, ": iterations ", oms_lower_bound," to ", oms_upper_bound - 1, flush=True)

comm.Barrier() # start parallel processes

start_time = time.time()

for ind in range(oms_lower_bound, oms_upper_bound):

    k2_vec[ind], E_vec[ind] = main_calc() 
    
    print("Iteration ", ind, " done in processor ", rank)

if rank == 0:
    k2_global = np.zeros(len(oms_vec))
    E_global = np.zeros(len(oms_vec))

else:
    k2_global = None
    E_global = None

comm.Barrier() # end prallel processes

comm.Reduce( k2_vec , k2_global, op=MPI.SUM, root=0)

comm.Reduce( E_vec , E_global, op=MPI.SUM, root=0)

stop_time = time.time()

if rank ==0:
    # Complete processes that did not evenly fit the number of processors
    for ind in range(size*num_per_rank, t_step):

        k2_global[ind], E_global[ind] = main_calc()

    print("time spent with ", size, " threads in minutes")
    print("-----", int((stop_time - start_time)/60 ), "-----")

    dill.dump([oms_vec, k2_global, E_global], file=open('./homotime{}_tau{}_N2{}_H{}.pickle'.format(t_step, tau, N2, int(H/1e5)), 'wb') )

'''
oms_size   = 12500 
oms_vec      = np.linspace(1, 3.312, oms_size)*2*np.pi/Ts/3600

#Definitions 

t_step = 500 #timestep 
a0 = 7*Rs #Beginning SMA is 7 RJ
a1 = 10*Rs #Ending SMA is 10 RJ 
t1 = 1.482e17 #4.7 billion years in seconds 
t = np.linspace(1, 1.482e17, t_step) # 0 - 4.7 billion years with 1000 steps 
k2J = .6
Q = 10e4

#SMA Calculation
C = ((3*k2J)/Q)*ms*(Rs**5)*((G/Ms)**.5)
k = (a0**(13/2))*(2/13)
sma = ((13/2)*(C*t+k))**(2/13)

#Calculate omega 

#bottom = (((4*np.pi)**2)/(G*Ms))*((t*((a1-a0)/t1)+(a0))**3)

#a_test = t*((a1-a0)/t1)+(a0)

a1_dot = ((13/2)*(((C*t)/2)+k))**(-11/13)

frac = a1_dot/sma
#a2 = math.exp(np.trapz(a1_dot, dx = t1))

def integrand(t):
    return 1/((13/2)*(((C*t)/2)+k))

a2 = ((integrate.quad(integrand, 0, t1)))
'''
