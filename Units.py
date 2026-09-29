#####################################################################
#                           PyRocket								#
# 2D Regenertive Cooling Simulation for Bipropellant Rocket Engines #
#                                                                   #
# Creator:  Joanthan Neeser                                         #
# Date:     15.12.2022                                              #
# Version:  2.3  													#
# License:	GNU GENERAL PUBLIC LICENSE V3							#
#####################################################################

# Global temperature unit for plots, section images, summaries and terminal output: 'K' or 'C'
# Config inputs and the data files (sim_data.csv, summary.json, summary.csv) always stay in K
TEMPERATURE_UNIT = 'K'

if TEMPERATURE_UNIT not in ('K', 'C'):
    raise ValueError('Invalid TEMPERATURE_UNIT in Units.py. Select: "K" or "C"')


def temperature(T):
    # convert a temperature [K] to the display unit, works for floats and arrays
    return T - 273.15 if TEMPERATURE_UNIT == 'C' else T


def temperature_unit():
    # display unit string for labels
    return '°C' if TEMPERATURE_UNIT == 'C' else 'K'
