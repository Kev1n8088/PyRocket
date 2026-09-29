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
bartz_knockdown = 1.0                                # knockdown factor for bartz heat transfer. 0.9 typical for smooth hotwall + traditional impinging jet injector, can be as low as 0.5 with interesting recirculation zones. 1.0 for no knockdown

# Operating point of combustion chamber 
eta_c_star      = 0.8                               # c* efficiency
MR              = 1.4
m_dot           = 1.65                                # total mass flow [kg/s]
m_dot_f         = m_dot / (MR + 1)
m_dot_ox        = m_dot - m_dot_f
Pc              = 27e5                               # Chamber pressure [Pa]

# Coolant input
cooling_fluid   = ['n-dodecane']                     # needs to be list of str (thermo names); RP1 -> 'n-dodecane' (surrogate), IPA -> 'isopropanol'
fluid_mass_frac = [1.0]    			                 # mass fractions of the cooling fluid (needs to add up to 1)
m_dot_coolant   = m_dot_f                 		 	 # mass flow through the cooling channels
inlet_temp      = 288.0                     		 # inlet temperature [K]
inlet_pressure  = 45e5                               # Cooling channel inlet pressure [Pa]

cooling_method  = 'gnielinski'                       # use either 'gnielinski', 'dittus-boelter' or 'dittus-boelter-simple', pay attention to suitable Re number range
ambient_temp    = 288.15							 # ambient temperature for radiation boundary condition [K] (must be type float)!!

# Film cooling settings for coolant originating at the injector face
m_dot_film = 0.1 * m_dot_coolant                      # set mass flow to 0.0 to deactivate film cooing
injection_velocity = 10								  # [m/s]
injector_diameter  = 0.4e-3							  # [m]

# Chamber geometry, r_1 and r_2 follow RPA nomenclature
D_c       = 76e-3                                   # chamber diamter [m]
D_t       = 31.03e-3                                    # throat diameter [m]
D_e       = 58.34e-3                                    # exit diamter [m]
L_cyl     = 127.99e-3                                   # cylindrical chamber length [m]
r_2       = 72.29e-3                                    # converging section inlet radius, R2 in RPA [m]
r_1       = 23.27e-3                                  # throat converging section radius, R1 in RPA [m]
r_n       = 5.93e-3                                  # throat diverging radius [m]
phi_conv  = 30                                       # convergence angle [deg]
phi_div   = 16.29                                       # divergence angle [deg]
phi_e     = 8.00                                       # exit angle [deg]
nozzle_type = 'bezier'                               # use either 'parabola' (length follows from phi_div and phi_e) or 'bezier' (RPA style, uses L_e)
L_e       = 63.42e-3                                 # nozzle length from throat to exit [m], only used for 'bezier'
step_size = 0.003                                    # step size along the chamber contour [m]
material  = matlib.AlSi10Mg
stress_longitudinal = 'shell'                        # hot wall stress margin (RPE method, materials with stress data): temperature difference of the longitudinal
                                                     # thermal stress, 'shell' = hot wall to coolant-side wall, 'wall' = across the hot wall
# cooling channel geometry; h_c, psi, t_w_i can be functions of x
n         = 40                                       # number of cooling channels [int]
h_c       = 1e-3                                     # radial height of cooling channels [m] CAN BE FLOAT OR FUNCTION
# EXAMPLE of function input for psi: psi = lambda x: 1/3 * (1 - 0.0001 * x) 
psi       = 0.55                                      # fill factor of the cooling channels; fraction of the circumferecne covered by the cooling channels (0 - 1) CAN BE FLOAT OR FUNCTION
t_w_i     = 0.8e-3                                   # inner chamber wall thickness [m] CAN BE FLOAT OR FUNCTION
t_w_o     = 1e-3                                     # outer chamber wall thickness [m]
start_idx = -1										 # starting index, use 0 for injector side and -1 for nozzle

# Two-pass cooling: alternating channels, half of them carry the coolant from the end given by start_idx to the other end, where a
# manifold turns it into the other half, which carry it back. Inlet and outlet are both at the start_idx end. n must be even
two_pass        = False
turnaround_loss = 1.5                                # pressure loss of the turnaround manifold, times the dynamic pressure in the channels

# Helical cooling channel configuration
# Set helix_pitch_deg_per_mm to a non-zero value to enable helical channels.
# The pitch is specified as degrees of circumferential revolution per millimetre of axial distance,
# e.g. 72 deg/mm means a full 360-degree turn every 5 mm.
# This overrides helix_angle if both are set.
# helix_angle is used when you know the angle directly instead of the pitch;
# it can be a float [deg] or a callable f(x) returning an array of angles along the contour.
# Set both to their defaults (pitch=None, angle=0.0) to revert to straight axial channels.
helix_pitch_deg_per_mm = None                        # helix pitch [deg/mm], e.g. 72.0; set None for axial channels
helix_angle            = 0.0                         # helix angle [deg] from the meridian (the axis on the cylinder), float or callable f(x) -> array; ignored if pitch is set

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
