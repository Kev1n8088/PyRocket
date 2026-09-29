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
fuel            = proplib.ipa               # choose existing fuel or oxidiser from rocketcea or create new fuel or oxidiser blend in PropLibrary
ox              = 'O2'
hot_gas_method  = 'standard-bartz'                         # use either 'cinjarev', 'standard-bartz' or 'modified-bartz'
bartz_knockdown = 0.9                                # knockdown factor for bartz heat transfer. 0.9 typical for smooth hotwall + traditional impinging jet injector, can be as low as 0.5 with interesting recirculation zones. 1.0 for no knockdown

# Operating point of combustion chamber 
eta_c_star      = 0.8                               # c* efficiency
MR              = 1.4
m_dot           = 1.532                                # total mass flow [kg/s]
m_dot_f         = m_dot / (MR + 1)
m_dot_ox        = m_dot - m_dot_f
Pc              = 20.68e5                               # Chamber pressure [Pa]

# Coolant input
cooling_fluid   = ['isopropanol']                     # needs to be list of str (thermo names); RP1 -> 'n-dodecane' (surrogate), IPA -> 'isopropanol'
fluid_mass_frac = [1.0]    			                 # mass fractions of the cooling fluid (needs to add up to 1)
m_dot_coolant   = m_dot_f                 		 	 # mass flow through the cooling channels
inlet_temp      = 288.0                     		 # inlet temperature [K]
inlet_pressure  = 45e5                               # Cooling channel inlet pressure [Pa]

cooling_method  = 'gnielinski'                       # use either 'gnielinski', 'dittus-boelter' or 'dittus-boelter-simple', pay attention to suitable Re number range
ambient_temp    = 288.15							 # ambient temperature for radiation boundary condition [K] (must be type float)!!

# Film cooling settings for coolant originating at the injector face
m_dot_film = 0.35 * m_dot_coolant                      # set mass flow to 0.0 to deactivate film cooing
injection_velocity = 10								  # [m/s]
injector_diameter  = 0.4e-3							  # [m]

# Chamber geometry, r_1 and r_2 follow RPA nomenclature
D_c       = 88.55e-3                                   # chamber diamter [m]
D_t       = 36.15e-3                                    # throat diameter [m]
D_e       = 64.91e-3                                    # exit diamter [m]
L_cyl     = 121.60e-3                                   # cylindrical chamber length [m]
r_2       = 84.22e-3                                    # converging section inlet radius, R2 in RPA [m]
r_1       = 27.11e-3                                  # throat converging section radius, R1 in RPA [m]
r_n       = 6.90e-3                                  # throat diverging radius [m]
phi_conv  = 30                                       # convergence angle [deg]
phi_div   = 15.14                                       # divergence angle [deg]
phi_e     = 8.00                                       # exit angle [deg]
nozzle_type = 'bezier'                               # use either 'parabola' (length follows from phi_div and phi_e) or 'bezier' (RPA style, uses L_e)
L_e       = 64.91e-3                                 # nozzle length from throat to exit [m], only used for 'bezier'
step_size = 0.003                                    # step size along the chamber contour [m]
material  = matlib.AlSi10Mg
# cooling channel geometry; h_c, psi, channel_width, t_w_i and helix_angle can be functions of x
# Design: narrow fast helical channels over the high heat flux zone around the throat and in the chamber after the end of the 
# liquid film, wider slower channels elsewhere to save pressure drop. Limits: channel width and height >= 1 mm and rib width
# >= 0.8 mm (both normal to the flow), helix angle <= 30 deg, coolant pressure drop <= 10 bar
n         = 44                                       # number of cooling channels [int], the most that fit the throat at 30 deg with 1 mm channels and 0.8 mm ribs
psi       = 0.55                                     # fill factor of the cooling channels (0 - 1) CAN BE FLOAT OR FUNCTION, not used as channel_width is set
t_w_i     = 0.8e-3                                   # inner chamber wall thickness [m] CAN BE FLOAT OR FUNCTION
t_w_o     = 1e-3                                     # outer chamber wall thickness [m]
start_idx = -1										 # starting index, use 0 for injector side and -1 for nozzle. Injector side: the IPA reaches the throat warmer and 
                                                     # far less viscous, which almost doubles the coolant side heat transfer there (572 K vs 617 K max wall temperature)

# contour of this chamber, used to place the cooling zones
from GeomClass import ChamberGeometry as _ChamberGeometry
_contour = _ChamberGeometry(D_c=D_c, D_t=D_t, D_e=D_e, L_cyl=L_cyl, r_2=r_2, r_1=r_1, r_n=r_n, phi_conv=phi_conv, phi_div=phi_div,
                            phi_e=phi_e, step_size=2e-4, nozzle_type=nozzle_type, L_e=L_e)
_contour.contour()
x_throat = _contour.x[np.argmin(_contour.y)]

X_ZONE    = x_throat - 4e-3                          # centre of the throat zone, at the heat flux peak just upstream of the throat [m]
S_UP      = 20e-3                                    # throat zone width upstream [m]
S_DOWN    = 15e-3                                    # throat zone width downstream [m]
ALPHA     = 30.0                                     # helix angle in the throat zone [deg]
X_C0      = 70e-3                                    # chamber zone start, just before the end of the liquid film [m]
X_C1      = 170e-3                                   # chamber zone end, where it joins the throat zone [m]
ALPHA_C   = 15.0                                     # helix angle in the chamber zone [deg]
W_ZONE    = 1.0e-3                                   # channel width normal to the flow in the throat zone [m]
W_C       = 1.2e-3                                   # channel width normal to the flow in the chamber zone [m]
W_FAR     = 2.0e-3                                   # channel width normal to the flow elsewhere [m]
H_C       = 1.0e-3                                   # channel height [m]
RIB_MIN   = 0.8e-3                                   # minimum rib width normal to the flow [m]

def zone(x):
    # smooth weight, 1 in the throat zone and falling off with separate widths up- and downstream
    x = np.asarray(x, dtype=float)
    return np.exp(-((x - X_ZONE) / np.where(x < X_ZONE, S_UP, S_DOWN))**2)

def chamber_zone(x):
    # smooth plateau, 1 between X_C0 and X_C1 with 5 mm edges
    x = np.asarray(x, dtype=float)
    return 0.25 * (1 + np.tanh((x - X_C0) / 5e-3)) * (1 - np.tanh((x - X_C1) / 5e-3))

h_c = H_C                                            # radial height of cooling channels [m] CAN BE FLOAT OR FUNCTION

def channel_width(x):
    # channel width normal to the flow [m]: narrow in the zones, wide elsewhere, limited so the rib keeps its minimum width
    r_i = np.interp(x, _contour.x, _contour.y) + t_w_i
    pitch_normal = 2 * np.pi * r_i * np.cos(np.radians(helix_angle(x))) / n
    w = W_FAR + (W_ZONE - W_FAR) * zone(x)
    w = np.minimum(w, W_FAR + (W_C - W_FAR) * chamber_zone(x))
    return np.minimum(w, pitch_normal - RIB_MIN)

# Helical cooling channel configuration
# Set helix_pitch_deg_per_mm to a non-zero value to enable helical channels.
# The pitch is specified as degrees of circumferential revolution per millimetre of axial distance,
# e.g. 72 deg/mm means a full 360-degree turn every 5 mm.
# This overrides helix_angle if both are set.
# helix_angle is used when you know the angle directly instead of the pitch;
# it can be a float [deg] or a callable f(x) returning an array of angles along the contour.
# Set both to their defaults (pitch=None, angle=0.0) to revert to straight axial channels.
helix_pitch_deg_per_mm = None                        # helix pitch [deg/mm], e.g. 72.0; set None for axial channels

def helix_angle(x):
    # helix angle [deg] from the meridian (the axis on the cylinder): steepest over the throat zone, moderate in the chamber zone
    return np.maximum(ALPHA * zone(x), ALPHA_C * chamber_zone(x))

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
