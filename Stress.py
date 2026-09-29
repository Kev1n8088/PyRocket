#####################################################################
#                           PyRocket								#
# 2D Regenertive Cooling Simulation for Bipropellant Rocket Engines #
#                                                                   #
# Creator:  Joanthan Neeser                                         #
# Date:     15.12.2022                                              #
# Version:  2.3  													#
# License:	GNU GENERAL PUBLIC LICENSE V3							#
#####################################################################

# Hot wall stress estimate after Sutton & Biblarz (2017), Ch. 8, used by single runs (run.py) and the cooling optimiser (optimize.py)

import csv
import numpy as np


def has_stress_data(material):
    # temperature dependent Young's modulus, expansion and yield strength are set with Material.mechanical_properties
    return all(hasattr(material, name) for name in ('E_T', 'cte_T', 'yield_T'))


def wall_stress(material, T_wall_hot, q, t_wall, width, p_coolant, p_gas, T_wall_cold, k=None, longitudinal='shell'):
    """[Summary]
    Hot wall stresses per station (Sutton & Biblarz 2017, Ch. 8):
      tangential:   (p_coolant - p_gas)/2 * (width/t_wall)^2 (wall between ribs as a clamped beam)
                    + E a q t_wall / (2 (1 - nu) k)          (through-thickness temperature gradient)
      longitudinal: E a dT, fully restrained; dT = T_wall_hot - T_wall_cold ('shell') or q t_wall / k ('wall')
    Both compressive at the hot face, combined as von Mises (2D plane stress), against the yield strength at the hot face temperature.
    k [W/m/K] per station or constant; the material's k at the mean wall temperature if None.
    Returns arrays: sigma_t, sigma_l, sigma_vm, yield, margin (yield / von Mises - 1), T_mean.
    """
    T_wall_hot, q = np.asarray(T_wall_hot, dtype=float), np.asarray(q, dtype=float)
    k_mean = material.k(T_wall_hot) if k is None else k
    T_mean = T_wall_hot - q * t_wall / (2 * k_mean)                     # mean temperature of the hot wall
    k = material.k(T_mean) if k is None else k
    E, cte, nu = material.E_T(T_mean), material.cte_T(T_mean), material.v
    sigma_t = (np.asarray(p_coolant) - np.asarray(p_gas)) / 2 * (np.asarray(width) / t_wall)**2 + E * cte * q * t_wall / (2 * (1 - nu) * k)
    dT = np.maximum(T_wall_hot - np.asarray(T_wall_cold), 0.0) if longitudinal == 'shell' else q * t_wall / k
    sigma_l = E * cte * dT
    sigma_vm = np.sqrt(sigma_t**2 + sigma_l**2 - sigma_t * sigma_l)
    strength = material.yield_T(T_wall_hot)
    return {'sigma_t': sigma_t, 'sigma_l': sigma_l, 'sigma_vm': sigma_vm, 'yield': strength, 'margin': strength / sigma_vm - 1, 'T_mean': T_mean}


def chamber_stress(material, out, cooling_geometry, p_gas, longitudinal='shell'):
    """[Summary]
    Wall stresses at every station of a finished run, from its output (an Output1D) and cooling geometry. The channel width is taken
    normal to the flow. For two-pass cooling the higher of the two coolant pressures loads the wall.
    """
    cg = cooling_geometry
    two_pass = getattr(cg, 'passes', 1) == 2
    p_coolant = np.maximum(out.P_c, out.P_c2) if two_pass else out.P_c
    return wall_stress(material, out.T_wall_i, out.q, cg.t_w_i, cg.psi_c_n * cg.r_i, p_coolant, p_gas, out.T_wall_c, longitudinal=longitudinal)


def write_csv(stress, geometry, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x coordinate [mm]', 'sigma_tangential [Pa]', 'sigma_longitudinal [Pa]', 'sigma_von_mises [Pa]', 'yield strength [Pa]',
                         'margin [-]', 'T_wall_mean [K]'])
        for i in range(len(geometry[:, 0])):
            writer.writerow([geometry[i, 0] * 1e3] + [float(stress[key][i]) for key in ('sigma_t', 'sigma_l', 'sigma_vm', 'yield', 'margin', 'T_mean')])
