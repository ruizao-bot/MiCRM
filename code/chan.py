############################# PRODUCING ACTUAL SIMULATION GRAPHS #############################
import numpy as np
import os
import sys

sys.path.append(os.path.expanduser("~/Documents/MiCRM/code"))
import param
from numpy.random import default_rng
from scipy.integrate import solve_ivp
from numpy import linspace 
import matplotlib.pyplot as plt
import chan_function

import CUE
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
"""
want to first generate parameters for a particular randomly-assembled community
and then simulate 31 different temperatures for this same community (both MiCRM and EGLV graphs for each temperature) 
compare how temperature affects the deviation between MiCRM and EGLV graphs
"""


rng = default_rng(111)

N = 7
M = 5
L = np.full(N, 0.3) # leakage (this is not temp dependent, as per original MiCRM)

x0 = np.concatenate([np.full(N, 0.1), np.full(M, 1)]) # initial conditions for consumers and resources

# Temperature‐dependence parameters
num_temps = 31 # number of temperatures
rho_t = np.array([0.0, 0.0])   # minimal trade‐off
Tr = 273.15 + 10 # reference temperature (10 °C)
Ed = 3.5 



####### TEST OUT DIFFERENT TEMPERATURES #######


structural = generate_params(
    N, M,
    f_u=def_u,      # relative preferences only
    f_m=def_m,      # placeholder
    f_rho=def_rho,
    f_omega=def_omega,
    f_l=def_l,
    # *no* T, Tr, Ed, rho_t here
    L=L,
    T=273.15,   # dummy (This satisfies temp_trait’s requirement that kw contain T, rho_t, Tr, Ed) 
    # since we are using the default def_u and def_m here, they ignore kw, so any T will work 
    # this whole thing will just provide relative preferences (u) and a constant m=1    
    rho_t=rho_t,  # dummy
    Tr=Tr,        # dummy
    Ed=Ed         # dummy
)

# the 'structural' parameters are static and don't change with temperature
# e.g. the def_u only generates relative preferences, not absolute ones
# however the uptake rate (u) will change with temperature. the relative preferences won't. 
# other things like rho and omega also don't change with temperature



temp_vals = linspace(273.15, 273.15 + 30, num_temps) # 31 temperatures from 0 to 30 degrees C

results = [] # store results for each temperature. it is a list of dictionaries. each temp would produce its own dictionary. 

for T in temp_vals:

    # temp-dependent scalars

    temp_p, B, E, Tp = temp_trait(N, {
        'T': T, 'Tr': Tr, 'Ed': Ed, 'rho_t': rho_t, 'L': L
    })
    temp_p_u = temp_p[:,0]
    temp_p_m = temp_p[:,1]

    # full parameter dictionary for this temp

    pT = {
        **structural,            # brings in u_pref, l, B, E, Tp, L, N, M, etc.
        'u': structural['u'] * temp_p_u[:,None],  # absolute uptake rates. the preference matrix from structural['u'] is multiplied (scaled) by the temperature-dependent uptake rates
        'm': temp_p_m,                            # mortality rates. the preference matrix from structural['m'] is multiplied (scaled) by the temperature-dependent mortality rates
        'lambda': np.sum(structural['l'], axis=2),
        'T': T
    }   
    

    # set up integration range

    t_max_micrm = 500
    t_max_glv   = 500

    t_eval_micrm = np.linspace(0, t_max_micrm, 1000)
    t_eval_glv   = np.linspace(0, t_max_glv,   1000)


    # solve MiCRM at this temperature
    

    sol = solve_ivp(
        lambda t, y: MiCRM_dxx(y, t, pT),
        t_span=(0, t_max_micrm),
        y0=x0,
        method='BDF',
        t_eval=t_eval_micrm 
    )

    
    # solve EGLV at this temperature
    
    p_lv = eff_LV_params(pT, sol, verbose=False)

    sol_lv = solve_ivp(
        lambda t, y: LV_dx(y, t, p_lv),
        t_span=(0, t_max_glv), 
        y0=sol.y[:N, 0],
        method='BDF', 
        t_eval=t_eval_glv        
        )
    
    ##### deviation calculations #####

    # first collect the equilibrium values for MiCRM and GLV (this is the last value in time series, t1)
    C_MiCRM_eq = sol.y[:N, -1] # equilibrium consumer biomass
    C_LV_eq = sol_lv.y[:N, -1] # equilibrium consumer biomass

       
    # equilibrium abundance deviation 
    ErrEqAb, overlap = err_eq_and_overlap(C_LV_eq, C_MiCRM_eq)

    # trajectory deviation  
    times      = sol.t                   # shape (T,). this defines the time array to be investigated in trajectory deviations 
    C_Mi_traj  = sol.y[:N, :]            # shape (N, T). this is the MiCRM trajectory, to be analysed in trajectory deviations 
    C_LV_traj  = sol_lv.y[:N, :]         # shape (N, T). this is the GLVM trajectory, to be analysed in trajectory deviations 
    err_t, overlap_t = err_time_series(times, C_LV_traj, C_Mi_traj) # first get the time series 
    j_eq = estimate_teq(times, sol, sol_lv, pT, p_lv, tol=1e-6, window=5) # find equilibrium time 
    times_crop = times[: j_eq + 1] # crop time array to equilibrium 
    err_crop   = err_t[: j_eq + 1] # crop trajectory error array to equilibrium 
    Err_traj = integrate_err(times_crop, err_crop) # integrate over [0, t_eq] instead of [0, t_max]

    # diversity deviation
    jaccard = jaccard_index(C_LV_eq, C_MiCRM_eq, thresh=1e-6)
    sh_LV = shannon(C_LV_eq)
    sh_Mi = shannon(C_MiCRM_eq)
    bc = bray_curtis_dissimilarity(C_LV_eq, C_MiCRM_eq)

    # stability (Jacobian) and reactivity (Hermitian) 

    J_glv   = eff_LV_jac(p_lv, sol)
    stab_glv  = leading_eigenvalue(J_glv)
    react_glv = leading_hermitian_eigenvalue(J_glv)

    J_micrm   = MiCRM_jac(pT, sol)
    stab_mic  = leading_eigenvalue(J_micrm)
    react_mic = leading_hermitian_eigenvalue(J_micrm)

    # timescale separation 

    J_full = MiCRM_jac(pT, sol)                 # (N+M)x(N+M)
    diagJ  = np.diag(J_full)
    # consumer return times τ_{C_i} = 1/|J_{ii}| for i=0..N-1
    tau_Cs = 1.0/np.abs(diagJ[:N])
    tau_C  = np.min(tau_Cs)                     # fastest consumer
    # resource return times τ_{R_α} = 1/|J_{(N+α),(N+α)}|
    tau_Rs = 1.0/np.abs(diagJ[N:])
    tau_R  = np.max(tau_Rs)                     # slowest resource
    epsilon = tau_C / tau_R                     # timescale‐separation metric

       
    ##### store results as a dictionary ##### 

    results.append(dict(T=T, 
                        sol=sol, 
                        sol_lv=sol_lv, 
                        ErrEqAb=ErrEqAb, 
                        overlap=overlap,
                        ErrTraj=Err_traj,
                        jaccard=jaccard,
                        sh_LV=sh_LV,
                        sh_Mi=sh_Mi,
                        bray_curtis=bc,
                        stab_glv=stab_glv,
                        stab_mic=stab_mic,
                        react_glv=react_glv,
                        react_mic=react_mic,
                        tau_C=tau_C,
                        tau_R=tau_R,
                        epsilon=epsilon
                        ))

# analyse results for equilibrium abundance

# extract values for plotting 
temps_C = temp_vals - 273.15 # Convert temperature array from K to °C
errs     = [r['ErrEqAb'] for r in results]
overlaps = [r['overlap']  for r in results]
errtraj = [r['ErrTraj']  for r in results]
jaccards = [r['jaccard'] for r in results]
sh_lvs   = [r['sh_LV']   for r in results]
sh_mis   = [r['sh_Mi']   for r in results]
bcs = [r['bray_curtis']   for r in results]
stabs_glv  = [r['stab_glv']  for r in results]
stabs_mic  = [r['stab_mic']  for r in results]
reacts_glv = [r['react_glv'] for r in results]
reacts_mic = [r['react_mic'] for r in results]
abs_diff_stab  = np.abs(np.array(stabs_glv ) - np.array(stabs_mic )) # absolute differences 
abs_diff_react = np.abs(np.array(reacts_glv) - np.array(reacts_mic)) # absolute differences 
taus_C      = [r['tau_C']    for r in results]
taus_R      = [r['tau_R']    for r in results]
epsilons    = [r['epsilon']  for r in results]


