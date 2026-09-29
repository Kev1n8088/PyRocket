#####################################################################
#                           PyRocket								#
# 2D Regenertive Cooling Simulation for Bipropellant Rocket Engines #
#                                                                   #
# Creator:  Joanthan Neeser                                         #
# Date:     15.12.2022                                              #
# Version:  2.3  													#
# License:	GNU GENERAL PUBLIC LICENSE V3							#
#####################################################################

import importlib.util
import os
import numpy as np

from IsentropicRelations import Isentropic
from CEAClass import BipropCEA
from GeomClass import ChamberGeometry, CoolingGeometry
from Output import Output1D, Settings2D
from FilmCooling import FilmCooling
from PropLibrary import create_coolant


def case_name(config_path):
    # name of a case is the file name of its config without extension
    return os.path.splitext(os.path.basename(config_path))[0]


def read_inputs(config_path, overrides=None):
    """[Summary]
    Executes a config file and returns its input parameters as a module, without building any objects.
    overrides is an optional dict of input names and values that replace those in the file, e.g. for sweeps. 
    Values derived from an input inside the config file (e.g. m_dot_f from m_dot) are not recomputed.
    """
    spec = importlib.util.spec_from_file_location('case_' + case_name(config_path), config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    for name, value in (overrides or {}).items():
        if name == 'material' and isinstance(value, str):
            # material given by its name in MaterialLib, e.g. 'IN718'
            import MaterialLib
            value = getattr(MaterialLib, value)
        setattr(cfg, name, value)

    cfg.config_path = os.path.abspath(config_path)
    return cfg


def load_config(config_path, save_path=None, overrides=None):
    """[Summary]
    Loads a config file containing only input parameters (see config.py) and builds the gas, coolant and geometry objects
    from them. Returns the config module with the built objects attached, e.g. cfg.chamber, cfg.cea, cfg.coolant.
    Output is written to save_path, defaults to 'results/<config name>'. See read_inputs for overrides.
    """
    cfg = read_inputs(config_path, overrides)
    cfg.save_path = save_path if save_path is not None else os.path.join('results', case_name(config_path))
    build(cfg)
    return cfg


def build_geometry(cfg):
    # chamber contour and cooling channel geometry from the config inputs, cheap to evaluate (no CEA or coolant properties)

    # optional inputs, parabolic nozzle by default
    nozzle_type = getattr(cfg, 'nozzle_type', 'parabola')
    L_e         = getattr(cfg, 'L_e', None)

    # generate chamber geometry, r_1 and r_2 follow RPA nomenclature
    chamber = ChamberGeometry(D_c=cfg.D_c, D_t=cfg.D_t, D_e=cfg.D_e, L_cyl=cfg.L_cyl, r_2=cfg.r_2, r_1=cfg.r_1, r_n=cfg.r_n,
                              phi_conv=cfg.phi_conv, phi_div=cfg.phi_div, phi_e=cfg.phi_e, step_size=cfg.step_size, nozzle_type=nozzle_type, L_e=L_e,
                              step_size_throat=getattr(cfg, 'step_size_throat', None), throat_zone=getattr(cfg, 'throat_zone', None))
    chamber.contour()

    # generate cooling channel geometry, straight axial channels unless a helix angle or pitch is set
    cooling_geom = CoolingGeometry(chamber.geometry, cfg.h_c, cfg.psi, cfg.t_w_i, cfg.t_w_o, cfg.n,
                                   helix_angle=getattr(cfg, 'helix_angle', 0.0),
                                   pitch_deg_per_mm=getattr(cfg, 'helix_pitch_deg_per_mm', None),
                                   channel_width=getattr(cfg, 'channel_width', None),
                                   passes=2 if getattr(cfg, 'two_pass', False) else 1)
    cooling_geom.channel_geometry()
    return chamber, cooling_geom


def build(cfg):
    ######################################
    # GAS, COOLANT and GEOMETRY SETUP
    ######################################

    # create folder to store sim files
    os.makedirs(cfg.save_path, exist_ok=True)

    # generate chamber and cooling channel geometry
    cfg.chamber, cfg.cooling_geom = build_geometry(cfg)
    cfg.chamber.plot_contour(cfg.save_path)
    cfg.geometry = cfg.chamber.geometry
    cfg.cooling_geom.set_thermocouples(cfg.TC_x, cfg.TC_r)

    # generate settings class for 2D thermal sim
    cfg.settings = Settings2D(cfg.cell_size, cfg.time_step, cfg.tolerance, cfg.max_iter, cfg.save_fig, cfg.print_result, cfg.run_time, cfg.start_idx, cfg.adaptive_up, cfg.log_TC, cfg.cooling_geom.thermocouples)
    cfg.settings.steady_solver = getattr(cfg, 'steady_solver', 'direct')  # 'direct' or 'pseudo-transient' for run_time = 'steady_state'

    # generate output class
    cfg.output = Output1D(cfg.geometry, cfg.save_path)

    # calculate hot gas properties from NASA CEA
    cfg.cea = BipropCEA(cfg.fuel, cfg.ox, cfg.Pc)
    cfg.cea.metric_cea_output('throat', cfg.MR, cfg.chamber.expansion_ratio)

    # isentropic properties for the hot gas along the chamber contour
    cfg.gas = Isentropic(cfg.Pc, cfg.cea.Tc, cfg.cea.gamma, cfg.cea.Pr, cfg.geometry[:,0], cfg.geometry[:,1])
    cfg.gas.calculate()

    # coolant Properties from thermo library (or custom coolant registered in PropLibrary)
    cfg.coolant = create_coolant(cfg.cooling_fluid, ws=cfg.fluid_mass_frac, T=cfg.inlet_temp, P=cfg.inlet_pressure)

    # film cooling model
    cfg.film = FilmCooling(cfg.cea, cfg.coolant, cfg.injection_velocity, cfg.injector_diameter, cfg.D_c, cfg.m_dot - cfg.m_dot_film, cfg.m_dot_film)
    cfg.film.film_length()
    cfg.film.cooled_idx = np.where(cfg.geometry[:,0] <= cfg.film.liquid_film_length)[0]
