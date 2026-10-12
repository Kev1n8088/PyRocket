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


##################################################
# operating points
##################################################

# keys of an entry of throttle_points in a config
POINT_KEYS = ('name', 'throttle', 'MR_core', 'inlet', 'overrides')


def core_mixture_ratio(cfg):
    # core (combustion) mixture ratio, i.e. of the propellant that actually burns, the film excluded. Older configs that
    # only define a single MR (not yet split into core and film) fall back to it
    MR_core = getattr(cfg, 'MR_core', None)
    return cfg.MR if MR_core is None else MR_core


def core_mass_flow(cfg):
    # core (combustion) mass flow, the film excluded as it is assumed not to burn. Older configs without m_dot_core fall
    # back to their total mass flow less the film
    m_dot_core = getattr(cfg, 'm_dot_core', None)
    return (cfg.m_dot - cfg.m_dot_film) if m_dot_core is None else m_dot_core


def throttle_overrides(cfg, throttle, MR_core=None, inlet=None):
    """[Summary]
    Config inputs of one throttle point, derived from the design point inputs of cfg (as returned by read_inputs).
    The core mass flow and the chamber pressure scale with throttle. MR_core is the core mixture ratio at this point,
    which a fixed area injector shifts when the engine is throttled; it defaults to the design point ratio. The film keeps
    the fraction of the fuel flow it has at the design point, so it is recomputed from the throttled fuel flow rather than
    scaled with the throttle, and the coolant flow follows as core fuel + film.
    inlet is the jacket inlet pressure treatment, 'hold' (throttle valve downstream of the jacket, the default) or 'scale'
    (valve upstream, so the inlet follows Pc plus the feed pressure drop, which goes with m_dot^2). It defaults to the
    config's throttle_inlet.
    """
    throttle = float(throttle)
    if not 0.0 < throttle <= 1.0:
        raise ValueError('A throttle point needs a throttle fraction of the design mass flow in (0, 1], got %r' % throttle)

    MR_core = core_mixture_ratio(cfg) if MR_core is None else float(MR_core)
    inlet   = getattr(cfg, 'throttle_inlet', 'hold') if inlet is None else inlet
    if inlet not in ('hold', 'scale'):
        raise ValueError("The jacket inlet pressure of a throttle point is either 'hold' or 'scale', got %r" % inlet)

    m_dot_core = core_mass_flow(cfg) * throttle
    m_dot_f    = m_dot_core / (MR_core + 1)
    m_dot_ox   = m_dot_core - m_dot_f

    # the film is the same fraction of the fuel flow as at the design point, at the fuel flow of this point
    film_fraction = cfg.m_dot_film / cfg.m_dot_f if cfg.m_dot_f else 0.0
    m_dot_film    = film_fraction * m_dot_f

    Pc = cfg.Pc * throttle
    m_dot_coolant = m_dot_f + m_dot_film
    overrides = {'MR_core': MR_core, 'm_dot_core': m_dot_core, 'm_dot_f': m_dot_f, 'm_dot_ox': m_dot_ox,
                 'm_dot_film': m_dot_film, 'm_dot': m_dot_core + m_dot_film, 'm_dot_coolant': m_dot_coolant,
                 'Pc': Pc}
    if inlet == 'scale':
        # the jacket and the injector downstream of the throttle valve keep their flow areas, so the pressure drop across
        # them goes with the square of the flow through them. That is the coolant flow, which only tracks the throttle
        # while the mixture ratio is held: a point that shifts O/F moves the fuel flow off the core throttle fraction
        flow = m_dot_coolant / cfg.m_dot_coolant if cfg.m_dot_coolant else throttle
        overrides['inlet_pressure'] = Pc + (cfg.inlet_pressure - cfg.Pc) * flow**2
    return overrides


def operating_points(config_path, cfg=None):
    """[Summary]
    The cases a config expands into: its design point, followed by one case per entry of its throttle_points (see config.py).
    Returns a list of (case name, overrides, extra), where overrides are the config inputs of that point and extra is stored
    with its summary. A config without throttle_points gives the design point alone, i.e. the plain case it has always been.
    """
    cfg = read_inputs(config_path) if cfg is None else cfg
    base = case_name(config_path)
    points = [(base, {}, {'point': 'design', 'throttle': 1.0, 'MR_core': core_mixture_ratio(cfg)})]

    for i, point in enumerate(getattr(cfg, 'throttle_points', None) or []):
        unknown = [k for k in point if k not in POINT_KEYS]
        if unknown:
            raise ValueError('throttle_points[%d] of %s has unknown key(s) %s. Use: %s'
                             % (i, base, ', '.join(sorted(unknown)), ', '.join(POINT_KEYS)))
        if 'throttle' not in point:
            raise ValueError("throttle_points[%d] of %s needs a 'throttle', its fraction of the design mass flow" % (i, base))

        throttle  = float(point['throttle'])
        name      = str(point.get('name') or '%gpct' % round(100 * throttle, 6))
        overrides = throttle_overrides(cfg, throttle, point.get('MR_core'), point.get('inlet'))
        overrides.update(point.get('overrides') or {})
        points.append((base + '_' + name, overrides,
                       {'point': name, 'throttle': throttle, 'MR_core': overrides['MR_core']}))

    names = [p[0] for p in points]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError('The throttle points of %s must have unique names, they name the output folders. Duplicates: %s'
                         % (base, ', '.join(duplicates)))
    return points


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

    # core (combustion) mixture ratio and mass flow: the propellant that actually burns, excluding film coolant, which is
    # assumed not to combust at all
    cfg.MR_core    = core_mixture_ratio(cfg)
    cfg.m_dot_core = core_mass_flow(cfg)

    # calculate hot gas properties from NASA CEA, at the core combustion mixture ratio
    cfg.cea = BipropCEA(cfg.fuel, cfg.ox, cfg.Pc)
    cfg.cea.metric_cea_output('throat', cfg.MR_core, cfg.chamber.expansion_ratio)

    # isentropic properties for the hot gas along the chamber contour
    cfg.gas = Isentropic(cfg.Pc, cfg.cea.Tc, cfg.cea.gamma, cfg.cea.Pr, cfg.geometry[:,0], cfg.geometry[:,1])
    cfg.gas.calculate()

    # coolant Properties from thermo library (or custom coolant registered in PropLibrary)
    cfg.coolant = create_coolant(cfg.cooling_fluid, ws=cfg.fluid_mass_frac, T=cfg.inlet_temp, P=cfg.inlet_pressure)

    # film cooling model: main gas flow is the core flow only, the film does not add to it as it is assumed not to burn
    cfg.film = FilmCooling(cfg.cea, cfg.coolant, cfg.injection_velocity, cfg.injector_diameter, cfg.D_c, cfg.m_dot_core, cfg.m_dot_film)
    cfg.film.film_length()
    cfg.film.cooled_idx = np.where(cfg.geometry[:,0] < cfg.film.liquid_film_length)[0]
