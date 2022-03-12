from numba import njit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from sys import exit

import warnings
from time import time
warnings.filterwarnings('error')

# Van Genuchten constitutive relations
@njit
def Cw(h):
    return -α*n*m*(1 + (α*h)**n)**(-(m+1)) * (α*h)**(n-1)

@njit
def K(h):
    return (Ks*(1-(α*h)**(m*n) * (1 + (α*h)**n)**(-m))**2)/(1 + (α*h)**n)**(m*l)

@njit
def θ(h):
    return θr + (θs - θr)/(1 + (α*h)**n)**m

@njit
def invθ(θ): # h
    return (1/α)*(((θs - θr)/(θ - θr))**(1/m) - 1)**(1/n)

# Solver settings
# ---------------
sol_do_calibration_workflow = False
sol_compute_residuals = True
sol_compute_numerical_probes = True
sol_datafile = './field_measurements/DF27_period1_reduced.csv' # 'test_data.csv'

sol_use_RWU = False
RWU_start_hour = 8
RWU_end_hour = 23
sol_use_free_drainage_bc = True # Soil physics with Hydrus ~pp190

# Simulation parameters
# ---------------------
θr = 0.01
θs = 0.539 # Value for vertical fracture fillings karst (Yang)
θdry = 0.01 + 0.0001
gbc = 0.1 # Constant gradient at lower boundary (L/L)
m  = 0.99
n  = 1/(1 - m)
#m = 1-(1/n)
l = 1
Ks = 1e-6
α = -1#-4.9e-4
Z = 200/1000
τ0 = 0
τ = 3600*24 # total time of simulation *24 is a whole day
t_checkpoints = [n*τ/8 for n in range(8+1)]
dt = 0.75
dz = Z/100
Nz = int(Z/dz) # grid cell count

# Surface boundary conditions parameters
a1 = 0.0003
a2 = 0.002

# Root water uptake
# -----------------
Ψx = -1e6
β = 0.01
Rmin = 1e8
d = 1e6
Brwu = 0.74
C = 1e-6

# Numerical probes
# ----------------
np_probe_every_s = 60 # s
np_probe_depth = 25 # dz = 25mm
np_surface_head = []
np_residual_median = []
np_residual_max = []
np_RWU = []
np_dΨx = []

print(f'>> dt={round(dt,4)}s, dz={round(dz,4)}m, Nz={Nz}')

# Initial conditions
h0 = np.zeros(Nz+2)
h0[0] = invθ(0.4)
h0[1:]= invθ(θdry)

@njit
def log_mean_p(X):
    new = np.zeros_like(X[1:-1])
    for i,(x,xp) in enumerate(zip(X[1:-1], X[2:])):
        if abs(np.log(x) - np.log(xp)) > 0:
            new[i] = (x - xp)/(np.log(x) - np.log(xp))
        else:
            new[i] = X[i]
    return new

@njit
def log_mean_m(X):
    new = np.zeros_like(X[1:-1])
    for i,(x,xm) in enumerate(zip(X[1:-1], X[:-2])):
        if abs(np.log(xm) - np.log(x)) > 0:
            new[i] = (xm - x)/(np.log(xm) - np.log(x))
        else:
            new[i] = X[i]
    return new

def compute_RWU(h, Ψx, timestamp):
    if timestamp != None:
        timestamp = int(f'{timestamp}'.split('T')[1].split(':')[0])

    if ((timestamp == None) or
        ((timestamp != None) and (timestamp >= RWU_start_hour and timestamp <= RWU_end_hour))):
        _K = K(h[1:-1])
        Rp = Rmin*np.exp((-Ψx/d)**Brwu)
        λ = 1/(sum([β**(i*dz/Z) for (i,_) in enumerate(h[1:-1])]))
        Rr = np.array([Rp/(λ*β**(i*dz/Z)) for (i,_) in enumerate(h[1:-1])])
        Rsr = np.array([dz/(λ*β**(i*dz/Z)*_K[i]) for (i,_) in enumerate(h[1:-1])])
        RWU = (h[1:-1] - Ψx)/(Rr + Rsr)
        return RWU
    else:
        return np.zeros_like(h[1:-1])

def create_const_vector(h, t, Ψx, Cw, timestamp):
    A = log_mean_p(K(h))*dt/(Cw[1:-1]*dz**2)
    B = log_mean_m(K(h))*dt/(Cw[1:-1]*dz**2)
    b = np.zeros_like(h)
    RWU = compute_RWU(h, Ψx, timestamp)
    b[1:-1] = h[1:-1] + (A - B)*dz - (RWU*(1/Cw[1:-1]) if sol_use_RWU else 0)
    b[0] = h[0]
    b[-1] = h[-1]
    return b

def create_coeff_matrix(h, Cw):
    A = log_mean_p(K(h))*dt/(Cw[1:-1]*dz**2)
    B = log_mean_m(K(h))*dt/(Cw[1:-1]*dz**2)
    diag_main = np.zeros(Nz+2)
    diag_sup = np.zeros(Nz+2)
    diag_low = np.zeros(Nz+2)
    for i in range(1,Nz+1):
        diag_main[i]  = 1 + A[i-1] + B[i-1]
        diag_sup[i]   = -A[i-1]
        diag_low[i-1] = -B[i-1]
    diag_main[0]  = 1
    diag_main[-1] = 1
    return diags(diagonals=[diag_main, diag_low, diag_sup],
                 offsets=[0,-1,1],
                 shape=(Nz+2, Nz+2),
                 format='csr')

def lower_boundary_condition(h):
    if sol_use_free_drainage_bc:
        return h[-2] + dz*(gbc - 1) # Neuman
    else:
        return h[-1] # Dirichlet

def upper_boundary_condition(h, rain=None):
    if rain == None:
        return h[0]
    else:
        θ0 = θ(h[0])
        if rain > 0:
            assert True, ''
        return invθ(max(θdry, θ0 - dt*a1 + dt*a2*rain*(1/θ0)))

def try_run_solver(h, t, Ψx, Cw, timestamp=None):
    try:
        return spsolve(create_coeff_matrix(h, Cw), create_const_vector(h, t, Ψx, Cw, timestamp))
    except Exception as e:
        print(f'>> Solution didn\'t converge (t={t})')
        print(f'>> {e}')
        exit(0)

if sol_do_calibration_workflow:
    h = h0.copy()
    all_h = []
    t = 0
    start_time = time()
    current_probe_sp = 0
    __tchecks = list(reversed(t_checkpoints))
    current_checkpoint = __tchecks.pop()
    while t <= τ:
        # Misc. bookeeping
        if t >= current_checkpoint:
            all_h.append(h.copy())
            current_checkpoint = __tchecks.pop() if len(__tchecks) > 0 else 1e99

        hm = h.copy()
        hm[0] = upper_boundary_condition(hm)
        hm[-1] = lower_boundary_condition(hm)

        hi = try_run_solver(hm, t, Ψx, Cw(h))
        hi[0] = upper_boundary_condition(hi)
        hi[-1] = lower_boundary_condition(hi)
        h = try_run_solver(hm, t, Ψx, Cw((hi + hm)/2))

        # |Ax - b| == residual
        if sol_compute_residuals:
            residual = np.abs(np.asarray(np.matmul(create_coeff_matrix(h, Cw(h)).todense(), h)).reshape(-1) - create_const_vector(h, t, Ψx, Cw(h), None))

        if sol_compute_numerical_probes:
            if t >= current_probe_sp:
                np_surface_head.append(h[np_probe_depth])
                if sol_compute_residuals:
                    np_residual_median.append(np.median(residual))
                    np_residual_max.append(np.max(residual))
                current_probe_sp += np_probe_every_s

        print(f'Progress: {round(100*t/τ)}%   ', end='\r')
        t += dt
    print('')
    end_time = time()

    print(f'>> Computation took {round(end_time - start_time)} s')

    # Safe to remove: the following 2 lines simplify working on the chart without having to execute the full solver.
    pd.to_pickle((h, all_h, np_surface_head, np_residual_median, np_residual_max), './data.pkl')
    (h, all_h, np_surface_head, np_residual_median, np_residual_max) = pd.read_pickle('./data.pkl')

    plt.style.use('bmh')

    fig = plt.figure(figsize=(12,11))
    gs1 = GridSpec(4, 3, width_ratios=[0.3, 1, 1], height_ratios=[1, 0.4, 0.4, 0.4])
    ax1 = fig.add_subplot(gs1[0, 0])
    ax2 = fig.add_subplot(gs1[0, 1])
    ax3 = fig.add_subplot(gs1[0, 2])
    ax4 = fig.add_subplot(gs1[1, :])
    ax5 = fig.add_subplot(gs1[2, :])
    ax6 = fig.add_subplot(gs1[3, :])

    z = np.linspace(0,Z,Nz)
    Rp = Rmin*np.exp((-Ψx/d)**Brwu)
    λ = 1/(sum([β**(i*dz/Z) for (i,p) in enumerate(h[1:-1])]))
    root_density = np.array([λ*β**(i*dz/Z) for (i,p) in enumerate(h[1:-1])])

    ax1.set_ylim([Z, 0])
    ax1.set_xscale('log')
    ax1.set_xlabel('RDF')
    ax1.set_ylabel('z (m)')
    ax1.plot(root_density, z, color='red', linewidth=0.5)

    for i,h in enumerate(all_h):
        ax2.plot(np.abs(h[1:-1]), z, label=f'{round(t_checkpoints[i]/3600,1)} hr', linewidth=0.5)
        ax3.plot(θ(h[1:-1]), z, label=f'{round(t_checkpoints[i]/3600,1)} hr', linewidth=0.5)

    ax2.set_ylim([Z, 0])
    ax2.set_xscale('log')
    ax2.set_xlabel('|h| (m)')
    ax2.set_ylabel('z (m)')
    ax2.legend()

    ax3.set_ylim([Z, 0])
    ax3.set_xlabel('θ $(\\frac{cm^3}{cm^3})$')
    ax3.set_ylabel('z (m)')
    ax3.plot([θs, θs], [z[0], z[-1]], label='Saturation', color='blue', linestyle='--', linewidth=0.5)
    ax3.legend()

    ax4.set_xlabel(f'Time (probing interval = {np_probe_every_s} s)')
    ax4.set_ylabel('Soil moisture $(\\frac{cm^3}{cm^3})$')
    ax4.set_xlim([0, len(np_surface_head)-1])
    ax4.set_ylim([0, 1])
    ax4.plot(θ(np.array(np_surface_head)), label='Model', linewidth=0.5)
    ax4.legend()

    ax5.set_xlabel(f'Time (probing interval = {np_probe_every_s} s)')
    ax5.set_yscale('log')
    ax5.set_ylabel('|Pressure| $(Pa)$')
    ax5.set_xlim([0, len(np_surface_head)-1])
    ax5.plot(np.abs(np.array(np_surface_head)), label='Model', linewidth=0.5)
    ax5.legend()

    ax6.set_yscale('log')
    ax6.set_xlabel(f'Time (probing interval = {np_probe_every_s} s)')
    ax6.set_ylabel('Residual')
    ax6.set_xlim([0, len(np_residual_max)-1])
    ax6.plot(np_residual_max, label='Max.', linewidth=0.5)
    ax6.plot(np_residual_median, label='Median', linewidth=0.5)
    ax6.legend()

    plt.tight_layout()
    plt.savefig('richards_calibration.png')

else:
    data = pd.read_csv(sol_datafile, parse_dates=['Unnamed: 0'])
    from_date = data['Unnamed: 0'].min()
    to_date = data['Unnamed: 0'].max()
    ds_domain = data['Unnamed: 0'].values.copy()
    ds_rain = data['Rain(mm)'].values.copy()
    ds_soil_moisture = data['Soil moisture (cm3/cm3)'].values.copy()
    ds_sapflow = data['Total Flow(cm3/h)'].values.copy()
    ds_Ψx = data['Water Potential (MPa)'].values*1e6

    h = h0.copy()
    int_rwu = 0
    last_Ψx = ds_Ψx[0]
    print(f'>> Running full model with input file {sol_datafile}')
    print(f'>> Simulating {round(ds_rain.shape[0]/2)} hours-clock')
    start_time = time()
    for i,(timestamp, rain, m_Ψx) in enumerate(zip(ds_domain, ds_rain, ds_Ψx)):
        print(f'[{round(100*i/ds_rain.shape[0])}%] Timestamp: {timestamp}, Rain (mm): {rain}')
        t = 0
        τ = 3600/2
        current_probe_sp = 0
        int_rwu = 0
        while t <= τ:
            hm = h.copy()
            hm[0] = upper_boundary_condition(hm, rain)
            hm[-1] = lower_boundary_condition(hm)

            hi = try_run_solver(hm, t, m_Ψx, Cw(h))
            hi[0] = upper_boundary_condition(hi, rain)
            hi[-1] = lower_boundary_condition(hi)
            h = try_run_solver(hm, t, m_Ψx, Cw((hi + hm)/2))

            if t >= current_probe_sp:
                np_surface_head.append(h[np_probe_depth])
                current_probe_sp += np_probe_every_s

            int_rwu += np.abs(np.sum(compute_RWU(h, m_Ψx, timestamp)))
            t += dt
        np_RWU.append(int_rwu)
        np_dΨx.append(last_Ψx - m_Ψx)
        last_Ψx = m_Ψx

    end_time = time()
    print(f'>> Computation took {round((end_time - start_time)/round(ds_rain.shape[0]), 2)} seconds per 30 minutes-clock')
    print(f'>> Total simulation time: {round(end_time - start_time)} s')

    pd.to_pickle((np_surface_head, np_RWU), './data.pkl')
    (np_surface_head, np_RWU) = pd.read_pickle('./data.pkl')

    extended_domain = pd.to_datetime(np.linspace(from_date.value, to_date.value, len(np_surface_head)))

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(13,6), sharex=True)
    fig.suptitle(f'Infiltration model - {from_date} to {to_date}')

    ax1.set_ylim([θr, θs])
    # ax1.set_ylim([0, 0.2])
    # ax1.set_yscale('log')
    ax1.set_ylabel('$\\theta (\\frac{cm^3}{cm^3})$')
    ax1.plot(ds_domain, ds_soil_moisture, linewidth=0.5, label='Measurement')
    ax1.plot(extended_domain, θ(np.array(np_surface_head)), linewidth=0.5, label='Model')
    ax1.grid()
    ax1.legend()

    # ax2.set_yscale('log')
    ax2.set_ylabel('Volumetric flow $(\\frac{cm^3}{h})$')
    ax2.set_xlim([from_date, to_date])
    # ax2.set_ylim([0, 10000])
    ax2.plot(ds_domain, ds_sapflow, linewidth=0.5, label='Sapflow (Measurement)')
    ax2.plot(ds_domain, np.array(np_RWU), linewidth=0.5, label='RWU (Model)')
    # ax2.plot(ds_domain, np.array(np_RWU) - np.array(np_dΨx)*C, linewidth=0.5, label='Sapflow (Model)')
    ax2.grid()
    ax2.legend()

    ax3.set_yscale('log')
    ax3.set_ylabel('|Pressure| $(Pa)$')
    ax3.set_xlim([from_date, to_date])
    ax3.plot(ds_domain, np.abs(ds_Ψx), linewidth=0.5, label='Ψx (Measurement)')
    ax3.plot(extended_domain, np.abs(np.array(np_surface_head)), linewidth=0.5, label='h (Model)')
    ax3.grid()
    ax3.legend()

    plt.tight_layout()
    plt.savefig(f'richards_{sol_datafile}.png')
