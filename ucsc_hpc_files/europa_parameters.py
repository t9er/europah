# Solve the dynamical tides in a Poincare/Hough problem (uniform density, full Coriolis). 
#
# Ben Idini, Nov 2022.

from pdf2image import convert_from_path, convert_from_bytes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pdfFile = PdfPages("output.pdf")

graphChoice = input("Would you like to view dissipation or energy graphs? (dissipation, energy)")
parameter = input("What parameter would you like to change? (N2, H, tau)")
print("N2 suggestion: 1-2, H suggestion: 50-200, tau suggestion: 5-11")

lowerB = float(input("Lower bound"))
upperB = float(input("Upper bound"))
step = float(input("Step size"))

if parameter == 'N2':
    H = float(input("Set H"))* 1e5
    tau = 10**float(input("Set Tau"))
elif parameter == 'H':
    N2 = float(input("Set N2"))* 1e-9
    tau = 10**float(input("Set Tau"))
elif parameter == 'tau':
    N2 = float(input("Set N2"))* 1e-9
    H = float(input("Set H"))* 1e5
else:
    print('Incorrect Parameter')
    

for itr in np.arange(lowerB, upperB, step):
    import math
    from scipy.special import spherical_jn as jn
    from scipy.special import sph_harm as SH

    from interiorize.solvers import cheby
    from interiorize.poincare import dynamical as inertial
    from interiorize.hough import dynamical

    from matplotlib import rcParams
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.projections import get_projection_class
    from matplotlib.ticker import MaxNLocator



    import pdb
    import dill

    ## INPUT PARAMETERS
    G   = 6.67e-8                           # gravitational universal constant
    # Europa
    R       = 1561e5    # Mean radius (cm)
    rhom    = 3.04     # Mean density (g/cc)
    MOI     = 0.346     # Moment of inertia
    e       = 0.0009    # Eccentricity
    ms      = 4.8e25    # mass (g)
    Ts      = 3.5*24 # orbital period (hours)
    # Jupiter 
    Ms = 1.898e30
    Rs = 6.99e9

    # MODEL PARAMETERS
    rhow    = 1.    

    if parameter == 'N2':
        N2 = itr * 1e-9
    elif parameter == 'H':
        H = itr * 1e5
    elif parameter == 'tau':
        tau = 10**(itr)

    # Chebyshev solver
    N = 80         # number of Chebyshev polynomialsi
    Lmax = 80
    M = 2
    save = False
    label = 'tau9_gmode_'

    # Initial calculations
    L       = np.arange(abs(M), Lmax+1)
    eta     = (R-H)/R
    Rc      = R*eta        # core radius
    Rp      = R             # body radius
    a       = np.pi*eta
    b       = np.pi                              # planet's surface   
    rhoc    = 3*ms/(4*np.pi*Rc**3) - rhow*((R/Rc)**3 -1)  # fit satellite's mass
    sma     = (G*(Ms+ms)*(Ts*3600)**2/4/np.pi**2)**(1/3)     # satisfy Kepler's third law                   
    oms     = 2*np.pi/Ts/3600           # orbital frequency
    Om      = oms                # rotational frequency in synchronous rotation
    om      = M*(oms - Om)     # conventional tidal frequency
    ome     = oms     # eccentricity tidal frequency

    ###################################################

    # PLOT DEFINITIONS
    rcParams['font.sans-serif'] = "Arial"
    rcParams['font.family'] = "sans-serif"
    rcParams['mathtext.fontset'] = 'cm'

    plot_dir = '/Users/benja/Documents/projects/titan_tides/models/'

    def my_plt_opt():

        plt.minorticks_on()
        plt.tick_params(which='both', direction='in', top=True, right=True)
        for tick in plt.xticks()[1] + plt.yticks()[1]:
            tick.set_fontname("DejaVu Serif")

        plt.tight_layout()

        return

    ## PLOT RESULTS

    # Chebyshev spectral coefficients.
    # NOTE: gravity at the surface is the sum of the spectral coefficients. If they reach a round-off plateau, we are adding crap to the solution.


    # Radial displacement shells
    def field(p, theta, varphi=0):
        """
        Get the total field of a spherical harmonic decomposition while summing over all degree. The result is a cross section at a given azimuthal angle.
        p: radial function.
        theta: colatitude.
        varphi: azimuthal angle
        """

        field_grd = np.zeros((len(theta), len(p[0])), dtype=complex)
        
        for i in np.arange(len(L)):
            p_grd, th_grd = np.meshgrid(p[i], theta)
            field_grd += p_grd*SH(M, L[i], varphi, th_grd)

        return field_grd, th_grd 

    # generate grids for plotting.
    def my_plot2():
        print('start contourf plot')
        x = cheb.xi/np.pi
        th = np.linspace(0, np.pi, 1000)
        disp_grd, th_grd = field(dyn.y1, th)
        x_grd = np.meshgrid(x, th)[0]
        disp_grd_m = np.real(disp_grd)/100
        vmin = np.min(disp_grd_m)
        vmax = np.max(disp_grd_m)

        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        #cb = ax.contourf(th_grd, x_grd, abs(disp_grd_m), levels=500,vmin=0,vmax=30,  cmap='inferno')#'YlOrBr'
        log_disp = np.log10(abs(disp_grd_m))
        log_disp[log_disp<=0] = 0
        cb = ax.contourf(th_grd, x_grd, log_disp, levels=600, vmin=0,  cmap='inferno', extend='min')#'YlOrBr'
        #cb = ax.contourf(th_grd, x_grd, abs(disp_grd_m), levels=600, cmap='inferno')#'YlOrBr'
    #    ax.contour(th_grd, x_grd, disp_grd_m, levels=10, colors='white', linewidths=1)#'YlOrBr'
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)
        ax.axis("off")
        ax.set_theta_zero_location('N')
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_ylim([0,1])
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        colorb = fig.colorbar(cb, pad=-.1, ax=ax)
        temp = str(itr)
        colorb.set_label(r"$\log |\xi_r|$ where" + parameter + "=" + temp )
        #colorb.set_label(r'm')
        colorb.ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='upper'))
        my_plt_opt()

        if graphChoice == 'dissipation':
            pdfFile.savefig(fig)

        
   
    def my_plot3():
        x = cheb.xi/np.pi
        th = np.linspace(0, np.pi, 1000)
        disp_grd, th_grd = field(totalEnergy, th)
        x_grd = np.meshgrid(x, th)[0]
        disp_grd_m = np.real(disp_grd)/100
        vmin = np.min(disp_grd_m)
        vmax = np.max(disp_grd_m)

        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        #cb = ax.contourf(th_grd, x_grd, abs(disp_grd_m), levels=500,vmin=0,vmax=30,  cmap='inferno')#'YlOrBr'
        log_disp = np.log10(abs(disp_grd_m))
        log_disp[log_disp<=0] = 0
        cb = ax.contourf(th_grd, x_grd, log_disp, levels=600, vmin=0,  cmap='inferno', extend='min')#'YlOrBr'
        #cb = ax.contourf(th_grd, x_grd, abs(disp_grd_m), levels=600, cmap='inferno')#'YlOrBr'
    #    ax.contour(th_grd, x_grd, disp_grd_m, levels=10, colors='white', linewidths=1)#'YlOrBr'
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)
        ax.axis("off")
        ax.set_theta_zero_location('N')
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_ylim([0,1])
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        colorb = fig.colorbar(cb, pad=-.1, ax=ax)
        colorb.set_label(r'$\log |\xi_r|$')
        #colorb.set_label(r'm')
        colorb.ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='upper'))
        my_plt_opt()
        if graphChoice == 'energy':
            pdfFile.savefig(fig)
    ###################################################


    cheb = cheby(npoints=N, loend=a, upend=b)

    # Solve the tidal motion of the stratified ocean
    dyn = dynamical(cheb, ome, Om, ms, Ms, sma, Rp, 
                rho=rhow, rhoc=rhoc, Rc=Rc,
                l=L, m=M, tau=tau, x1=a, x2=b,
                tides='ee', e=e, N2=N2)

    print('model has a hydrostatic k2 = {:.4f}'.format(dyn.k_hydro(2)))

    dyn.solve(kind='bvp-core') # flow with ocean bottom

    print('model has a dynamic k2 = ',dyn.k[0])

    ###################################################
    l = 2
    y1 = np.array(dyn.y1)
    y2 = np.array(dyn.y2)
    y3 = np.array(dyn.y3)

    energy1 = rhom*tau*(Ts**2)
    energy2 = (y1**2)
    energy3 = l*(l+1)
    energy4 = (y2**2)
    energy5 = (y3**2)

    totalEnergy = (energy1 * (energy3 * (energy4 + energy5)))/(4*math.pi)

    print("total energy is equal to")
    print(totalEnergy)
    ###################################################
   
    if graphChoice == 'dissipation':
     my_plot2()
    if graphChoice == 'energy':
        my_plot3()


pdfFile.close()

images = convert_from_path('/Users/tyler/output.pdf')

for i in range(len(images)):
   
      # Save pages as images in the pdf
    images[i].save('page'+ str(i) +'.jpg', 'JPEG')