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
MR              = 1.3
m_dot           = 1.557                                # total mass flow [kg/s]
m_dot_f         = m_dot / (MR + 1)
m_dot_ox        = m_dot - m_dot_f
Pc              = 17.4e5                               # Chamber pressure [Pa]

# Coolant input
cooling_fluid   = ['n-dodecane']                     # needs to be list of str (thermo names); RP1 -> 'n-dodecane' (surrogate), IPA -> 'isopropanol'
fluid_mass_frac = [1.0]    			                 # mass fractions of the cooling fluid (needs to add up to 1)
m_dot_coolant   = m_dot_f                 		 	 # mass flow through the cooling channels
inlet_temp      = 288.0                     		 # inlet temperature [K]
inlet_pressure  = 34.5e5                               # Cooling channel inlet pressure [Pa]. Fuel throttle valve upstream of the jacket, so the jacket
                                                       # outlet sits at the injector manifold: Pc + 10 bar injector dp = 30.1 bar, plus ~2.3 bar jacket dp

cooling_method  = 'gnielinski'                       # use either 'gnielinski', 'dittus-boelter' or 'dittus-boelter-simple', pay attention to suitable Re number range
ambient_temp    = 288.15							 # ambient temperature for radiation boundary condition [K] (must be type float)!!

# Film cooling settings for coolant originating at the injector face
m_dot_film = 0.15 * m_dot_coolant                       # set mass flow to 0.0 to deactivate film cooing. ~15% is the most 200 s allows at 85% c* efficiency
injection_velocity = 10								  # [m/s]
injector_diameter  = 0.4e-3							  # [m]

# Chamber geometry, r_1 and r_2 follow RPA nomenclature
D_c       = 100.93e-3                                   # chamber diamter [m]
D_t       = 41.21e-3                                    # throat diameter [m]
D_e       = 64.57e-3                                    # exit diamter [m]
L_cyl     = 115.30e-3                                   # cylindrical chamber length [m]
r_2       = 96.0e-3                                    # converging section inlet radius, R2 in RPA [m]
r_1       = 30.90e-3                                  # throat converging section radius, R1 in RPA [m]
r_n       = 7.87e-3                                  # throat diverging radius [m]
phi_conv  = 30                                       # convergence angle [deg]
phi_div   = 13.15                                       # divergence angle [deg]
phi_e     = 8.00                                       # exit angle [deg]
nozzle_type = 'bezier'                               # use either 'parabola' (length follows from phi_div and phi_e) or 'bezier' (RPA style, uses L_e)
L_e       = 64.57e-3                                 # nozzle length from throat to exit [m], only used for 'bezier'
step_size = 0.003                                    # step size along the chamber contour [m]
material  = matlib.CuCr1Zr
# cooling channel geometry; h_c, psi, channel_width, t_w_i and helix_angle can be functions of x
# Design: two-pass with inlet and outlet at the injector end (the fuel ends up at the injector), axial channels whose width grows
# with the radius (constant fill factor), 1.2 mm wide at the throat. The warm return channel next to each cold forward channel
# carries the injector end, where the cold IPA runs laminar when throttled. Limits: channel width and height >= 1 mm, rib width
# >= 0.8 mm, coolant pressure drop <= 10 bar. Fast model at eta_c_star = 1.0: 745 K max hot wall, +7 K bulk boiling margin at full
# thrust; throttling needs O/F ~1.15 at 60% and ~1.10 at 50% (fixed-area injector, valve upstream of the jacket)
n         = 56                                       # number of cooling channels [int], of both passes together
t_w_i     = 0.75e-3                                   # inner chamber wall thickness [m] CAN BE FLOAT OR FUNCTION
t_w_o     = 1e-3                                     # outer chamber wall thickness [m]
psi       = n * 1.2e-3 / (2 * np.pi * (D_t / 2 + t_w_i))   # fill factor of the cooling channels (0 - 1) CAN BE FLOAT OR FUNCTION, gives 1.2 mm channels at the throat
#approx 0.38490446889786584

start_idx = 0										 # starting index, use 0 for injector side and -1 for nozzle. Two-pass: inlet and outlet at this end
two_pass = True
h_c       = 0.0025                                   # channel height [m]
helix_pitch_deg_per_mm = None
helix_angle            = 2.0


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
