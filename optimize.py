#####################################################################
#                           PyRocket								#
# 2D Regenertive Cooling Simulation for Bipropellant Rocket Engines #
#                                                                   #
# Creator:  Joanthan Neeser                                         #
# Date:     15.12.2022                                              #
# Version:  2.3  													#
# License:	GNU GENERAL PUBLIC LICENSE V3							#
#####################################################################

# Cooling design optimisation: sweeps the cooling channel geometry with the fast model (FastCooling: the cooling march along the chamber with
# a fast structured-grid section solve) and verifies the best designs with the 2D solver.
#
# Fixed from the config: chamber contour, propellants and mixture ratio, film cooling, coolant inlet (start_idx), one or two passes.
# Fixed by option: hot wall thickness (--wall, default the config's t_w_i).
# Varied: material, channel count, channel width (normal to the flow), channel height, a constant helix angle, and
# whether the channel width is constant along the chamber or grows with the radius (constant fraction of the circumference).
# Limits: maximum helix angle, minimum rib and channel width, minimum channel height, maximum coolant pressure drop, optionally a maximum
# coolant-side wall temperature (e.g. RP-1 coking) and minimum boiling margin, and the maximum service temperature of each material.
# Objective: the largest minimum stress margin along the chamber (RPE wall stress estimate, see Stress.wall_stress).
#
# Usage:
#   python3 optimize.py configs/ipa_lox_lp.py --max-dp 10 --max-angle 30 --min-rib 0.8 --min-channel-width 1 --min-channel-height 1 --jobs 16
#   python3 optimize.py config.py --materials IN718 CuCr1Zr --density coarse --verify 2
#
# Results in <out>/optimize_<config>/: designs_fast.csv (every evaluated design), optimize.png, verified/<design>/ (2D results) and best_design.txt

import argparse
import csv
import io
import contextlib
import itertools
import multiprocessing
import os
import sys
import tempfile
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import Summary
import Units
import MaterialLib
from CaseSetup import load_config, read_inputs, case_name
from GeomClass import CoolingGeometry
from FastCooling import FastHeatTransfer, QuietOutput, CoolantTable
from Stress import chamber_stress

MATERIALS = ('IN718', 'AlSi10Mg', 'CuCr1Zr')
MODES     = ('constant_width', 'constant_fill')
DENSITY   = {'coarse': dict(n=4, width=3, height=3, angle=3),
             'normal': dict(n=6, width=4, height=4, angle=4),
             'fine':   dict(n=8, width=5, height=5, angle=5)}


##################################################
# evaluation of one design with the 1D model (in worker processes)
##################################################

CASE = {}


def base_overrides(config_path):
    # the channel design of the config is replaced by the designs; only make sure the config's own channels build (two-pass needs an even count)
    cfg = read_inputs(config_path)
    return {'n': cfg.n + 1} if getattr(cfg, 'two_pass', False) and cfg.n % 2 else {}


def quiet_warnings():
    # the isentropic Mach number solve warns at the throat (double root at M = 1), harmless
    warnings.filterwarnings('ignore', message='The iteration is not making good progress')


def init_worker(config_path, table, tsat, limits, longitudinal, section_model):
    # build the fixed part of the case once per worker: contour, CEA, isentropic flow, film cooling
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    quiet_warnings()
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = load_config(config_path, tempfile.mkdtemp(prefix='pyrocket_optimize_'), base_overrides(config_path))
    CASE.update(cfg=cfg, table=table, tsat=tsat, limits=limits, longitudinal=longitudinal, section_model=section_model)


def width_inputs(cfg, d):
    # config inputs for the channel width of a design: a constant channel width normal to the flow, or for a width growing with the
    # radius the constant fill factor psi that gives the design width at the throat
    if d['mode'] == 'constant_width':
        return {'channel_width': float(d['width']), 'psi': 0.5}
    r_i_throat = float(np.min(cfg.geometry[:, 1])) + d['wall']
    return {'channel_width': None, 'psi': float(d['n'] * d['width'] / (2 * np.pi * r_i_throat * np.cos(np.radians(d['angle']))))}


def design_geometry(cfg, d):
    # cooling geometry of a design, raises ValueError if the channels do not fit
    w = width_inputs(cfg, d)
    if w['psi'] >= 1:
        raise ValueError('channels wider than the circumference allows')
    cg = CoolingGeometry(cfg.geometry, float(d['height']), w['psi'], float(d['wall']), float(cfg.t_w_o), int(d['n']), helix_angle=float(d['angle']),
                         pitch_deg_per_mm=None, channel_width=w['channel_width'], passes=cfg.cooling_geom.passes)
    cg.channel_geometry()
    return cg, w


def evaluate(d, cfg=None, table=None, tsat=None, limits=None, longitudinal=None, out=None, material=None):
    """[Summary]
    Metrics of a design from the 1D model, or from a finished 2D run when out (an Output1D with results) is given.
    """
    cfg = cfg or CASE['cfg']
    limits = limits or CASE['limits']
    longitudinal = longitudinal or CASE['longitudinal']
    table = table if table is not None else CASE.get('table')
    tsat = tsat if tsat is not None else CASE.get('tsat')
    result = dict(d)
    try:
        cg, _ = design_geometry(cfg, d)
    except ValueError as e:
        result.update(status='geometry', note=str(e), score=-1e9)
        return result

    w_ch, w_rib = cg.psi_c_n * cg.r_i, cg.psi_w_n * cg.r_i
    result.update(channel_min=float(w_ch.min()), rib_min=float(w_rib.min()), channel_max=float(w_ch.max()))
    tol = 1e-9
    if w_ch.min() < limits['min_channel'] - tol or w_rib.min() < limits['min_rib'] - tol or d['height'] < limits['min_height'] - tol or d['angle'] > limits['max_angle'] + tol:
        result.update(status='geometry', note='outside the geometry limits', score=-1e9)
        return result

    material = material or getattr(MaterialLib, d['material'])
    if out is None:
        out = QuietOutput(cfg.geometry, cfg.save_path)
        sim = FastHeatTransfer(cea=cfg.cea, gas=cfg.gas, geometry=cfg.geometry, material=material, coolant=cfg.coolant, cooling_geometry=cg,
                               m_dot=cfg.m_dot, m_dot_coolant=cfg.m_dot_coolant, T_amb=cfg.ambient_temp, output=out, settings2D=cfg.settings,
                               model=cfg.hot_gas_method, cool_model=cfg.cooling_method, eta_c_star=cfg.eta_c_star, film=cfg.film,
                               bartz_knockdown=getattr(cfg, 'bartz_knockdown', 1.0), turnaround_loss=getattr(cfg, 'turnaround_loss', 1.5),
                               table=table, section_model=CASE.get('section_model', 'grid'))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                sim.run()
        except Exception as e:
            result.update(status='failed', note=repr(e)[:200], score=-1e9)
            return result

    two = cg.passes == 2
    outlet = (0 if cfg.start_idx == 0 else -1) if two else (0 if cfg.start_idx == -1 else -1)
    T_out, P_out = (out.T_c2[outlet], out.P_c2[outlet]) if two else (out.T_c[outlet], out.P_c[outlet])
    st = chamber_stress(material, out, cg, cfg.gas.p_s, longitudinal=longitudinal)
    i = int(np.argmin(st['margin']))
    x = cfg.geometry[:, 0]

    dp = cfg.inlet_pressure - P_out
    T_max = float(np.max(out.T_wall_i))
    result.update(T_wall_max=T_max, x_T_wall_max=float(x[np.argmax(out.T_wall_i)]), T_wall_median=float(np.median(out.T_wall_i)),
                  dp=float(dp), T_coolant_out=float(T_out), q_max=float(np.max(out.q)), v_max=float(np.max(out.v_coolant)),
                  margin=float(st['margin'][i]), x_margin=float(x[i]), sigma_t=float(st['sigma_t'][i]), sigma_l=float(st['sigma_l'][i]),
                  sigma_vm=float(st['sigma_vm'][i]), yield_strength=float(st['yield'][i]), T_limit=float(material.T_max),
                  temperature_margin=float(material.T_max - T_max))
    walls = [(out.T_wall_c, out.P_c)] + ([(out.T_wall_c2, out.P_c2)] if two else [])
    result['T_wall_coolant_max'] = float(max(np.max(T) for T, _ in walls))
    if tsat is not None:
        result['boil_margin'] = float(min(np.min(np.interp(P, tsat[0], tsat[1]) - T) for T, P in walls))

    violations = []
    if dp > limits['max_dp']:
        violations.append((dp - limits['max_dp']) / limits['max_dp'])
    if T_max > material.T_max:
        violations.append((T_max - material.T_max) / 100.0)
    coolant_wall = limits.get('max_coolant_wall') is not None and result['T_wall_coolant_max'] > limits['max_coolant_wall']
    if coolant_wall:
        violations.append((result['T_wall_coolant_max'] - limits['max_coolant_wall']) / 100.0)
    boiling = limits.get('min_boil_margin') is not None and result.get('boil_margin') is not None and result['boil_margin'] < limits['min_boil_margin']
    if boiling:
        violations.append((limits['min_boil_margin'] - result['boil_margin']) / 100.0)
    result['status'] = 'ok' if not violations else 'limits'
    result['note'] = '' if not violations else ', '.join(n for n, v in (('pressure drop', dp > limits['max_dp']), ('wall temperature', T_max > material.T_max),
                                                                          ('coolant wall temperature', coolant_wall), ('coolant boiling', boiling)) if v)
    result['score'] = result['margin'] if not violations else -10.0 - sum(violations)
    return result


##################################################
# design space
##################################################

def levels(lo, hi, count):
    return [lo] if count <= 1 or hi <= lo else list(np.linspace(lo, hi, count))


def grid(cfg, args, limits):
    # all designs of the initial grid within the geometry limits at the throat
    density = DENSITY[args.density]
    r_t = float(np.min(cfg.geometry[:, 1]))
    even = cfg.cooling_geom.passes == 2
    wall = limits['wall']
    designs = []
    for material, mode, angle, height in itertools.product(args.materials, args.modes, levels(0.0, limits['max_angle'], density['angle']),
                                                           levels(limits['min_height'], limits['max_height'], density['height'])):
        circumference = 2 * np.pi * (r_t + wall) * np.cos(np.radians(angle))           # at the throat, normal to the flow
        n_max = int(circumference / (limits['min_channel'] + limits['min_rib']))
        n_min = max(4, int(round(n_max * 0.35)))
        n_values = sorted({int(round(v)) for v in levels(n_min, n_max, density['n'])})
        if even:
            n_values = sorted({n - n % 2 for n in n_values if n >= 4})
        for n in n_values:
            w_max = circumference / n - limits['min_rib']
            if w_max < limits['min_channel']:
                continue
            for width in levels(limits['min_channel'], w_max, density['width']):
                designs.append(dict(material=material, mode=mode, n=n, width=width, height=height, wall=wall, angle=angle))
    return designs


def neighbours(d, steps, limits, even):
    # compass (pattern) search neighbours of a design: one step up and down per variable (Kolda et al. 2003), within the limits
    result = []
    for key, lo, hi in (('width', limits['min_channel'], None), ('height', limits['min_height'], limits['max_height']),
                        ('angle', 0.0, limits['max_angle'])):
        for sign in (-1, 1):
            v = d[key] + sign * steps[key]
            if v < lo - 1e-12 or (hi is not None and v > hi + 1e-12):
                continue
            result.append(dict(d, **{key: v}))
    for sign in (-1, 1):
        n = d['n'] + sign * (2 if even else 1)
        if n >= 4:
            result.append(dict(d, n=n))
    return result


def key_of(d):
    return (d['material'], d['mode'], d['n'], round(d['width'], 7), round(d['height'], 7), round(d['wall'], 7), round(d['angle'], 5))


def label(d):
    return '%s_%s_n%d_w%.2f_h%.2f_t%.2f_a%.1f' % (d['material'], 'cw' if d['mode'] == 'constant_width' else 'cf', d['n'], d['width'] * 1e3,
                                                   d['height'] * 1e3, d['wall'] * 1e3, d['angle'])


##################################################
# 2D verification
##################################################

def verify_overrides(cfg, d):
    # config overrides that give the design in a 2D run; the mesh cell size follows the wall thickness as in the config
    material = getattr(MaterialLib, d['material'])
    cell = cfg.cell_size / cfg.t_w_i * d['wall']
    return dict({'material': d['material'], 'n': int(d['n']), 'h_c': float(d['height']), 't_w_i': float(d['wall']),
                 'helix_angle': float(d['angle']), 'helix_pitch_deg_per_mm': None,
                 'cell_size': cell, 'time_step': cell**2 / material.alpha, 'save_fig': False, 'print_result': False}, **width_inputs(cfg, d))


def run_2d(config_path, folder, d, overrides):
    from run import run_case
    return run_case(config_path, folder, True, overrides, {'design': label(d)})


##################################################
# output
##################################################

def write_csv(rows, path):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot(results, verified, limits, path, title):
    unit = Units.temperature_unit()
    colors = {m: Summary.SERIES[i] for i, m in enumerate(MATERIALS)}
    fig = plt.figure(figsize=(14, 9), facecolor=Summary.SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=0.38, wspace=0.25)

    ax = fig.add_subplot(gs[0, :])
    Summary.style_axes(ax)
    for m in MATERIALS:
        ok = [r for r in results if r['material'] == m and r['status'] == 'ok']
        lim = [r for r in results if r['material'] == m and r['status'] == 'limits']
        if ok:
            ax.plot([r['dp'] / 1e5 for r in ok], [r['margin'] for r in ok], 'o', ms=4, color=colors[m], alpha=0.55, markeredgewidth=0)
        if lim:
            ax.plot([r['dp'] / 1e5 for r in lim], [r['margin'] for r in lim], 'o', ms=4, color=colors[m], alpha=0.35, markerfacecolor='none', markeredgewidth=0.8)
        if ok or lim:
            ax.plot([], [], 'o', ms=6, color=colors[m], label=m + ('' if ok else ' (none within limits)'))
    for v in verified:
        r = v['1d']
        ax.plot(r['dp'] / 1e5, r['margin'], 'o', ms=12, markerfacecolor='none', markeredgecolor=Summary.TEXT, markeredgewidth=1.5)
        ax.annotate(v['name'], (r['dp'] / 1e5, r['margin']), textcoords='offset points', xytext=(8, 6), fontsize=7.5, color=Summary.TEXT)
    ax.axvline(limits['max_dp'] / 1e5, color=Summary.TEXT_2, linestyle='--', linewidth=1)
    ax.text(limits['max_dp'] / 1e5, 1.0, ' max pressure drop', transform=ax.get_xaxis_transform(), va='top', fontsize=8, color=Summary.TEXT_2)
    ax.axhline(0, color=Summary.BASELINE, linewidth=1)
    finite = [r['dp'] / 1e5 for r in results if r['status'] in ('ok', 'limits')]
    if finite:
        ax.set_xlim(0, min(max(finite), 3 * limits['max_dp'] / 1e5) * 1.05)
    ax.set_xlabel('Coolant pressure drop [bar]', fontsize=9, color=Summary.TEXT_2)
    ax.set_title('Fast model designs: minimum stress margin (yield / von Mises - 1)  —  filled: within limits, open: outside a pressure drop or temperature limit, ringed: verified in 2D',
                 fontsize=9.5, loc='left')
    ax.legend(frameon=False, fontsize=8, loc='lower right', labelcolor=Summary.TEXT)

    names = [v['name'] for v in verified]
    y = np.arange(len(verified))[::-1]
    for col, (key, conv, heading) in enumerate((('T_wall_max', Units.temperature, 'Max hot wall temperature [%s]' % unit), ('margin', lambda v: v, 'Minimum stress margin [-]'))):
        ax = fig.add_subplot(gs[1, col])
        Summary.style_axes(ax)
        ax.grid(axis='y', visible=False)
        ax.tick_params(axis='y', length=0)
        one = [conv(v['1d'][key]) for v in verified]
        two = [conv(v['2d'][key]) if v.get('2d') and key in v['2d'] else np.nan for v in verified]
        ax.barh(y + 0.18, one, height=0.34, color=Summary.SEQUENTIAL[3], edgecolor=Summary.SURFACE, label='fast model')
        ax.barh(y - 0.18, two, height=0.34, color=Summary.SEQUENTIAL[8], edgecolor=Summary.SURFACE, label='2D')
        for yi, a, b in zip(y, one, two):
            for value, offset in ((a, 0.18), (b, -0.18)):
                if np.isfinite(value):
                    ax.text(value, yi + offset, (' %.3g' if value >= 0 else '%.3g ') % value, va='center', ha='left' if value >= 0 else 'right', fontsize=7.5, color=Summary.TEXT_2)
        values = [v for v in one + two if np.isfinite(v)]
        if values:
            lo, hi = min(min(values), 0), max(max(values), 0)
            ax.set_xlim(lo - 0.18 * (hi - lo or 1), hi + 0.18 * (hi - lo or 1))
        ax.axvline(0, color=Summary.BASELINE, linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(names if col == 0 else [], fontsize=7.5, color=Summary.TEXT)
        ax.set_title(heading, fontsize=9.5, loc='left')
        if col == 1:
            ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(0, -0.08), ncol=2, labelcolor=Summary.TEXT)
    fig.suptitle(title, x=0.125, ha='left', fontsize=12, color=Summary.TEXT)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=Summary.SURFACE)
    plt.close(fig)


def config_lines(cfg, d, cfg_name):
    w = width_inputs(cfg, d)
    lines = ['# best design from optimize.py for %s, replaces the cooling channel inputs of the config' % cfg_name,
             'material  = matlib.%s' % d['material'],
             'n         = %d' % d['n'],
             'h_c       = %.12g                             # channel height [m]' % d['height'],
             't_w_i     = %.12g                             # hot wall thickness [m]' % d['wall'],
             'helix_pitch_deg_per_mm = None',
             'helix_angle            = %.12g' % d['angle']]
    if w['channel_width'] is not None:
        lines.append('channel_width = %.12g                         # channel width normal to the flow [m], the same along the chamber' % w['channel_width'])
    else:
        lines += ['channel_width = None                          # width grows with the radius: %.3f mm normal to the flow at the throat' % (d['width'] * 1e3),
                  'psi       = %.12g' % w['psi']]
    lines.append('cell_size = %.6g * t_w_i                          # keep the mesh resolution relative to the wall thickness' % (cfg.cell_size / cfg.t_w_i))
    return '\n'.join(lines)


##################################################
# main
##################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimise the cooling channel geometry of a config with the 1D model, verify with the 2D solver.')
    parser.add_argument('config', help='config file; chamber, propellants, film cooling, start_idx and one/two-pass stay as in the config')
    parser.add_argument('--materials', nargs='+', default=list(MATERIALS), choices=MATERIALS, help='wall materials to consider (default: all)')
    parser.add_argument('--modes', nargs='+', default=list(MODES), choices=MODES, help='channel width along the chamber: constant, or growing with the radius (default: both)')
    parser.add_argument('--max-dp', type=float, required=True, help='maximum coolant pressure drop [bar]')
    parser.add_argument('--max-angle', type=float, default=0.0, help='maximum helix angle from the meridian [deg] (default 0, axial channels)')
    parser.add_argument('--min-rib', type=float, required=True, help='minimum rib width normal to the flow [mm]')
    parser.add_argument('--min-channel-width', type=float, required=True, help='minimum channel width normal to the flow [mm]')
    parser.add_argument('--min-channel-height', type=float, required=True, help='minimum channel height [mm]')
    parser.add_argument('--max-channel-height', type=float, default=None, help='maximum channel height [mm] (default 3 x minimum)')
    parser.add_argument('--wall', type=float, default=None, help="hot wall thickness [mm], fixed for all designs (default: the config's t_w_i)")
    parser.add_argument('--max-coolant-wall-temp', type=float, default=None,
                        help='maximum temperature of the coolant-side (wetted) wall [K], e.g. to limit RP-1 coking (default: not limited)')
    parser.add_argument('--min-boil-margin', type=float, default=None,
                        help='minimum margin of the coolant wetted wall below the coolant saturation temperature [K] (default: not limited; negative allows some boiling)')
    parser.add_argument('--density', choices=sorted(DENSITY), default='normal', help='points per variable of the initial grid (default: normal)')
    parser.add_argument('--refine', type=int, default=6, help='pattern search rounds around the best designs (default 6)')
    parser.add_argument('--starts', type=int, default=6, help='number of best designs to refine (default 6)')
    parser.add_argument('--verify', type=int, default=3, help='number of designs to verify with the 2D solver (default 3)')
    parser.add_argument('--longitudinal', choices=('shell', 'wall'), default='shell',
                        help="temperature difference of the longitudinal thermal stress: hot wall to coolant-side wall (shell, default) or across the hot wall (wall)")
    parser.add_argument('--section-model', choices=('grid', 'network'), default='grid',
                        help="fast section model: structured finite volume grid (default, within a few K of the 2D solver) or resistance network (faster, less accurate for wide ribs of low conductivity materials)")
    parser.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1), help='parallel processes (default: all CPUs but one)')
    parser.add_argument('--out', default='results', help='output folder (default: results)')
    args = parser.parse_args()
    quiet_warnings()

    mm = 1e-3
    if args.wall is None:
        t_w_i = read_inputs(args.config).t_w_i
        if not isinstance(t_w_i, (int, float)):
            sys.exit("The config's t_w_i is a function of x, give the hot wall thickness with --wall")
        args.wall = t_w_i / mm
    limits = {'max_dp': args.max_dp * 1e5, 'max_angle': args.max_angle, 'min_rib': args.min_rib * mm, 'min_channel': args.min_channel_width * mm,
              'min_height': args.min_channel_height * mm, 'max_height': (args.max_channel_height or 3 * args.min_channel_height) * mm,
              'wall': args.wall * mm, 'max_coolant_wall': args.max_coolant_wall_temp, 'min_boil_margin': args.min_boil_margin}
    if limits['max_height'] < limits['min_height']:
        sys.exit('Maximum channel height must not be below the minimum')
    if limits['wall'] <= 0:
        sys.exit('The hot wall thickness must be positive')

    name = case_name(args.config)
    folder = os.path.join(args.out, 'optimize_' + name)
    os.makedirs(folder, exist_ok=True)
    base_folder = os.path.join(folder, '_base')
    start = time.time()

    print('Setting up %s' % name, flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = load_config(args.config, base_folder, base_overrides(args.config))

    # coolant property table over the expected temperatures and pressures, saturation temperature for the boiling margin
    T_lo, P_hi = cfg.inlet_temp - 5.0, cfg.inlet_pressure + 2e5
    P_lo = max(cfg.inlet_pressure - 3 * limits['max_dp'] - 5e5, 1e5)
    table = CoolantTable(cfg.cooling_fluid, cfg.fluid_mass_frac, T_lo, cfg.inlet_temp + 450.0, P_lo, P_hi)
    tsat = None
    try:
        import thermo
        chem = thermo.Chemical(cfg.coolant.IDs[int(np.argmax(cfg.coolant.ws))])
        P_s = np.linspace(P_lo, min(P_hi, chem.Pc * 0.999), 40)
        tsat = (P_s, np.array([chem.Tsat(p) for p in P_s]))
    except Exception:
        pass

    designs = grid(cfg, args, limits)
    two = cfg.cooling_geom.passes == 2
    print('Design space: %d designs (%s, %s, %s pass, start_idx %d, hot wall %.3g mm)' % (len(designs), ', '.join(args.materials), ', '.join(args.modes),
          'two' if two else 'one', cfg.start_idx, args.wall), flush=True)

    ctx = multiprocessing.get_context('spawn')
    results = {}
    with ctx.Pool(args.jobs, initializer=init_worker, initargs=(args.config, table, tsat, limits, args.longitudinal, args.section_model)) as pool:

        def run_batch(batch, stage):
            todo = [d for d in batch if key_of(d) not in results]
            for i, r in enumerate(pool.imap_unordered(evaluate, todo, chunksize=max(1, len(todo) // (args.jobs * 8) or 1)), 1):
                results[key_of(r)] = r
                if i % max(1, len(todo) // 50) == 0 or i == len(todo):
                    print('%s: designs evaluated %d / %d' % (stage, i, len(todo)), flush=True)

        # stage 1: grid
        run_batch(designs, 'Grid')

        # stage 2: pattern search around the best designs of the grid
        def best(n, pool_of=None):
            rows = sorted((pool_of or results.values()), key=lambda r: -r['score'])
            return [r for r in rows if r['status'] in ('ok', 'limits')][:n]

        density = DENSITY[args.density]
        base_steps = {'width': (limits['min_channel']) / max(density['width'], 2), 'height': (limits['max_height'] - limits['min_height']) / max(2 * density['height'], 2) or 0.1 * mm,
                      'angle': limits['max_angle'] / max(2 * density['angle'], 2)}
        starts = []
        for r in best(len(results)):
            if len(starts) >= args.starts:
                break
            # different materials and width modes first, then the next best
            if any(s['material'] == r['material'] and s['mode'] == r['mode'] for s in starts) and len({(s['material'], s['mode']) for s in starts}) < len(args.materials) * len(args.modes):
                continue
            starts.append(r)
        searches = [{'point': s, 'steps': dict(base_steps)} for s in starts]
        design_keys = ('material', 'mode', 'n', 'width', 'height', 'wall', 'angle')
        for round_ in range(1, args.refine + 1):
            batch = []
            for s in searches:
                s['candidates'] = neighbours({k: s['point'][k] for k in design_keys}, s['steps'], limits, two)
                batch += s['candidates']
            if not batch:
                break
            run_batch(batch, 'Refinement %d/%d' % (round_, args.refine))
            for s in searches:
                cands = [results[key_of(c)] for c in s['candidates'] if key_of(c) in results]
                better = [c for c in cands if c['score'] > s['point']['score'] + 1e-6]
                if better:
                    s['point'] = max(better, key=lambda c: c['score'])
                else:
                    s['steps'] = {k: v / 2 for k, v in s['steps'].items()}

    rows = sorted(results.values(), key=lambda r: -r['score'])
    write_csv(rows, os.path.join(folder, 'designs_fast.csv'))
    ok = [r for r in rows if r['status'] == 'ok']
    print('Fast model done: %d designs in %.0f s, %d within all limits' % (len(rows), time.time() - start, len(ok)), flush=True)

    # stage 3: 2D verification of the best designs, the best of each material first
    picks = []
    candidates = [r for r in (ok or rows) if r['status'] in ('ok', 'limits')]
    for distinct in (lambda r: (r['material'],), lambda r: (r['material'], r['mode']), lambda r: (r['material'], r['mode'], r['n'])):
        for r in candidates:
            if len(picks) >= args.verify:
                break
            if all(distinct(p) != distinct(r) for p in picks):
                picks.append(r)
    verified = []
    if picks and args.verify > 0:
        print('Verifying %d design(s) with the 2D solver' % len(picks), flush=True)
        jobs = [(args.config, os.path.join(folder, 'verified', label(p)), {k: p[k] for k in design_keys}, verify_overrides(cfg, p)) for p in picks]
        with ctx.Pool(min(args.jobs, len(jobs)), maxtasksperchild=1) as pool:
            cases = pool.starmap(run_2d, jobs)
        for p, case, job in zip(picks, cases, jobs):
            entry = {'name': label(p), '1d': p, 'case': case}
            if case.get('status') == 'ok':
                d = np.genfromtxt(os.path.join(job[1], 'sim_data.csv'), delimiter=',', skip_header=1)
                out = QuietOutput(cfg.geometry, job[1])
                out.T_wall_i, out.q, out.T_c, out.P_c, out.v_coolant, out.T_wall_c = d[:, 6], d[:, 5], d[:, 7], d[:, 8], d[:, 10], d[:, 12]
                out.T_c2, out.P_c2, out.T_wall_c2 = d[:, 13], d[:, 14], d[:, 18]
                entry['2d'] = evaluate(job[2], cfg=cfg, table=table, tsat=tsat, limits=limits, longitudinal=args.longitudinal, out=out)
            verified.append(entry)
        Summary.plot_summary([dict(e['case'], case=e['name']) for e in verified if e['case'].get('status') == 'ok'],
                             os.path.join(folder, 'verified'), os.path.join(folder, 'verified_summary.png'), heading='2D verification, %s' % name)
        write_csv([dict(design=e['name'], **{'fast_' + k: v for k, v in e['1d'].items() if k not in design_keys}, **{'2d_' + k: v for k, v in (e.get('2d') or {}).items() if k not in design_keys})
                   for e in verified], os.path.join(folder, 'verified.csv'))

    plot(rows, verified, limits, os.path.join(folder, 'optimize.png'), 'Cooling optimisation, %s' % name)

    # recommendation: the best design verified in 2D, by its 2D margin
    good = [e for e in verified if e.get('2d') and e['2d'].get('status') in ('ok', 'limits')]
    best_entry = max(good, key=lambda e: e['2d']['score']) if good else None
    unit = Units.temperature_unit()
    print('')
    print('%-44s %8s %8s %9s %9s %9s %9s  %s' % ('design (fast model, then 2D)', 'margin', '2D', 'Tw max', '2D', 'dp [bar]', '2D', 'status (2D)'))
    for e in verified:
        a, b = e['1d'], e.get('2d') or {}
        f = lambda v, fmt: fmt % v if isinstance(v, (int, float)) else '-'
        print('%-44s %8s %8s %9s %9s %9s %9s  %s' % (e['name'], f(a.get('margin'), '%+.3f'), f(b.get('margin'), '%+.3f'), f(Units.temperature(a['T_wall_max']), '%.1f'),
              f(Units.temperature(b['T_wall_max']) if 'T_wall_max' in b else None, '%.1f'), f(a['dp'] / 1e5, '%.2f'), f(b['dp'] / 1e5 if 'dp' in b else None, '%.2f'),
              (b.get('status', e['case'].get('status')) + (' (' + b['note'] + ')' if b.get('note') else '')) if b else e['case'].get('error', '')))
    print('')
    if best_entry:
        b = best_entry['2d']
        print('Best design (2D): %s' % best_entry['name'])
        print('  %s, %d channels, width %.3f mm (%s), height %.3f mm, hot wall %.3f mm, helix %.1f deg' % (b['material'], b['n'], b['width'] * 1e3, b['mode'].replace('_', ' '), b['height'] * 1e3, b['wall'] * 1e3, b['angle']))
        print('  minimum stress margin %+.3f at x = %.0f mm (von Mises %.0f MPa: tangential %.0f, longitudinal %.0f; yield %.0f MPa)' % (
            b['margin'], b['x_margin'] * 1e3, b['sigma_vm'] / 1e6, b['sigma_t'] / 1e6, b['sigma_l'] / 1e6, b['yield_strength'] / 1e6))
        print('  max hot wall %.1f %s (limit %.1f %s), max coolant-side wall %.1f %s%s, pressure drop %.2f bar (limit %.2f), coolant outlet %.1f %s%s' % (
            Units.temperature(b['T_wall_max']), unit, Units.temperature(b['T_limit']), unit, Units.temperature(b['T_wall_coolant_max']), unit,
            (' (limit %.1f %s)' % (Units.temperature(limits['max_coolant_wall']), unit)) if limits['max_coolant_wall'] is not None else '', b['dp'] / 1e5, limits['max_dp'] / 1e5,
            Units.temperature(b['T_coolant_out']), unit, (', coolant wall %.1f K %s boiling' % (abs(b['boil_margin']), 'below' if b['boil_margin'] >= 0 else 'above')) if b.get('boil_margin') is not None else ''))
        with open(os.path.join(folder, 'best_design.txt'), 'w') as f:
            f.write(config_lines(cfg, best_entry['1d'], name) + '\n')
        print('Config lines for this design: %s' % os.path.join(folder, 'best_design.txt'))
    elif verified:
        print('No verified design ran successfully in 2D, see verified/*/log.txt')
    print('Results in %s (%.0f s)' % (folder, time.time() - start))
