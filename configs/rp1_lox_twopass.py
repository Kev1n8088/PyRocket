#####################################################################
#                           PyRocket								#
# 2D Regenertive Cooling Simulation for Bipropellant Rocket Engines #
#                                                                   #
# Creator:  Joanthan Neeser                                         #
# Date:     15.12.2022                                              #
# Version:  2.3  													#
# License:	GNU GENERAL PUBLIC LICENSE V3							#                                          
#####################################################################

# Input parameters only. Any number of config files like this one can be run with run.py, e.g. copies in configs/
# The gas, coolant and geometry objects are built from these inputs in CaseSetup.py

import numpy as np

import MaterialLib as matlib
import PropLibrary as proplib


# CEA input
fuel            = proplib.rp1               # choose existing fuel or oxidiser from rocketcea or create new fuel or oxidiser blend in PropLibrary
ox              = 'O2'
hot_gas_method  = 'standard-bartz'                         # use either 'cinjarev', 'standard-bartz' or 'modified-bartz'
bartz_knockdown = 0.9                                # knockdown factor for bartz heat transfer. 0.9 typical for smooth hotwall + traditional impinging jet injector, can be as low as 0.5 with interesting recirculation zones. 1.0 for no knockdown

# Operating point of combustion chamber 
eta_c_star      = 1.0                               # c* efficiency
MR              = 1.8
m_dot           = 1.517                                # total mass flow [kg/s]
m_dot_f         = m_dot / (MR + 1)
m_dot_ox        = m_dot - m_dot_f
Pc              = 20.68e5                               # Chamber pressure [Pa]

# Coolant input
cooling_fluid   = ['n-dodecane']                     # needs to be list of str (thermo names); RP1 -> 'n-dodecane' (surrogate), IPA -> 'isopropanol'
fluid_mass_frac = [1.0]    			                 # mass fractions of the cooling fluid (needs to add up to 1)
m_dot_coolant   = m_dot_f                 		 	 # mass flow through the cooling channels
inlet_temp      = 288.0                     		 # inlet temperature [K]
inlet_pressure  = 51e5                               # Cooling channel inlet pressure [Pa]

cooling_method  = 'gnielinski'                       # use either 'gnielinski', 'dittus-boelter' or 'dittus-boelter-simple', pay attention to suitable Re number range
ambient_temp    = 288.15							 # ambient temperature for radiation boundary condition [K] (must be type float)!!

# Film cooling settings for coolant originating at the injector face
m_dot_film = 0.147 * m_dot_coolant                      # set mass flow to 0.0 to deactivate film cooing
injection_velocity = 37								  # [m/s]
injector_diameter  = 0.4e-3							  # [m]

# Chamber geometry, r_1 and r_2 follow RPA nomenclature
D_c       = 88.67e-3                                   # chamber diamter [m]
D_t       = 36.20e-3                                    # throat diameter [m]
D_e       = 63.74e-3                                    # exit diamter [m]
L_cyl     = 120.78e-3                                   # cylindrical chamber length [m]
r_2       = 84.34e-3                                    # converging section inlet radius, R2 in RPA [m]
r_1       = 27.15e-3                                  # throat converging section radius, R1 in RPA [m]
r_n       = 6.91e-3                                  # throat diverging radius [m]
phi_conv  = 30                                       # convergence angle [deg]
phi_div   = 15.52                                       # divergence angle [deg]
phi_e     = 8.00                                       # exit angle [deg]
nozzle_type = 'bezier'                               # use either 'parabola' (length follows from phi_div and phi_e) or 'bezier' (RPA style, uses L_e)
L_e       = 65.59e-3                                 # nozzle length from throat to exit [m], only used for 'bezier'
step_size = 0.003                                    # step size along the chamber contour [m]
material  = matlib.IN718
# cooling channel geometry; h_c, psi, channel_width, t_w_i and helix_angle can be functions of x
# Design: constant helix angle and constant fill factor (channel width grows with the radius), two-pass. Limits: channel width
# and height >= 1 mm and rib width >= 0.8 mm (both normal to the flow), helix angle <= 30 deg
# best design from optimize.py for ipa_lox_lp_twopass, replaces the cooling channel inputs of the config
n         = 52
h_c       = 0.00158333333333                              # channel height [m]
t_w_i     = 0.0008                              # hot wall thickness [m]
helix_pitch_deg_per_mm = None
helix_angle            = 27.5
channel_width = None                          # width grows with the radius: 1.000 mm normal to the flow at the throat
psi       = 0.493404405024
cell_size = 0.15 * t_w_i                          # keep the mesh resolution relative to the wall thickness
start_idx = 0										 # starting index, use 0 for injector side and -1 for nozzle. Injector side: the IPA reaches the throat warmer and 
t_w_o = 1e-3                                     # outer chamber wall thickness [m]

two_pass = True

# 2D Section Simulation settings
cell_size    = 0.15 * t_w_i                          # cell size in 2D section solver, will heavily impact performance
time_step    = (cell_size)**2 / (material.alpha)     # time step in 2D section solver (cell_size)**2 / (material.alpha)
adaptive_up  = 1.4                                   # UP factor for adaptive time step. Increase for faster convergence but less stability
tolerance    = 1e-2                                  # maximum temperature difference between time steps
max_iter     = 300									 # maximum number of iterations before termination
save_fig     = True                                  # save figures to folder in Output Class
print_result = True                                  # print intermittant maximum temperatures, useful for DEBUGGING
run_time     = 'steady_state'                        # use 'steady_state' as default. use time in [s] if transient solution is desired


# add thermocouple locations for model validation (effectively temperature logging points in the 2D solution)
log_TC = False                                       # temperature at thermocouple locations is to be logged
TC_x   = [10e-3, 20e-3, 30e-3]                       # x coordinate of thermocouples 
TC_r   = [36e-3, 36e-3, 36e-3]                       # radial position of thermocouples

# output is written to results/<config file name>/ by run.py
