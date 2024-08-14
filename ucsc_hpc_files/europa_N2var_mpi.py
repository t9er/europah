# Obtain dynamical tides in an ocean with varying stratification (B.V. frequency).
#
# Guideline:
# https://computationalmechanics.in/parallelizing-for-loop-in-python-with-mpi/
#
# Guidelines on Reduce: 
# https://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/
#
# Simple run: 
# mpirun -np 6 python3 titan_N2var_mpi.py
#
# 03/2023
# Ben Idini

import numpy as np
from mpi4py import MPI
import time
import pdb
import dill

from interiorize.solvers import cheby
from interiorize.hough import dynamical


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

G   = 6.67e-8
label = 'g6_tau9_mpi'
# Titan
R       = 1561e5    # Mean radius (cm)
rhom    = 3.04     # Mean density (g/cc)
MOI     = 0.346     # Moment of inertia
e       = 0.0009    # Eccentricity
ms      = 4.8e25    # mass (g)
Ts      = 3.5*24 # orbital period (hours)

# Saturn
Ms = 1.898e30
Rs = 6.99e9

# Model parameters
rhow    = 1.
tau     = 1e9       # frictional dissipation
H	= 150e5

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
sma     = (G*(Ms+ms)*(Ts*3600)**2/4/np.pi**2)**(1/3)     # satisfy Kepler's third law
oms     = 2*np.pi/Ts/3600           # orbital frequency
Om      = oms                # rotational frequency in synchronous rotation
ome     = oms     # eccentricity tidal frequency

# Varying N2 calculations
N2_size   = 10 
N2_vec      = np.linspace(1.375e-8, 1.525e-8, N2_size)
k2_vec = np.zeros(len(N2_vec))
E_vec = np.zeros(len(N2_vec))

# Main calculation
def main_calc():
    
    cheb = cheby(npoints=N, loend=a, upend=b)

    dyn = dynamical(cheb, ome, Om, ms, Ms, sma, Rp,
                 rho=rhow, rhoc=rhoc, Rc=Rc,
                 l=L, m=M, tau=tau, x1=a, x2=b,
                 tides='ee', e=e, N2=N2_vec[ind])
    
    dyn.solve(kind='bvp-core')

    return dyn.k[0], sum(dyn.E)

# MPI 
num_per_rank = N2_size//size

N2_lower_bound = rank*num_per_rank 
N2_upper_bound = (rank + 1)*num_per_rank

print("Processor ", rank, ": iterations ", N2_lower_bound," to ", N2_upper_bound - 1, flush=True)

comm.Barrier() # start parallel processes

start_time = time.time()

for ind in range(N2_lower_bound, N2_upper_bound):

    k2_vec[ind], E_vec[ind] = main_calc() 
    
    print("Iteration ", ind, " done in processor ", rank)

if rank == 0:
    k2_global = np.zeros(len(N2_vec))
    E_global = np.zeros(len(N2_vec))

else:
    k2_global = None
    E_global = None

comm.Barrier() # end prallel processes

comm.Reduce( k2_vec , k2_global, op=MPI.SUM, root=0)

comm.Reduce( E_vec , E_global, op=MPI.SUM, root=0)

stop_time = time.time()

if rank ==0:
    # Complete processes that did not evenly fit the number of processors
    for ind in range(size*num_per_rank, N2_size):

        k2_global[ind], E_global[ind] = main_calc()

    print("time spent with ", size, " threads in minutes")
    print("-----", int((stop_time - start_time)/60 ), "-----")

    dill.dump([N2_vec, k2_global, E_global], file=open('./k2Q_N2vec_L{}_N{}{}_H{}.pickle'.format(np.max(L), N, label, int(H/1e5)), 'wb') )
