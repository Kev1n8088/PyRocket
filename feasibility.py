#####################################################################
#                           PyRocket								#
# 2D Regenertive Cooling Simulation for Bipropellant Rocket Engines #
#                                                                   #
# Creator:  Joanthan Neeser                                         #
# Date:     15.12.2022                                              #
# Version:  2.3  													#
# License:	GNU GENERAL PUBLIC LICENSE V3							#
#####################################################################

# Quick feasibility screen of a cooling request, to run before optimize.py. Takes the same limits as optimize.py.
#
# Stage 1, energy budget (no solver, < 1 s): the hot gas heat load with the hot wall at the maximum service temperature of the material is
# the smallest heat load any design within the material limit can have (the load only grows as the wall gets colder). Put through the
# coolant enthalpy this gives the lowest possible bulk outlet temperature. Each channel wall is hotter than the coolant in it, so a design
# can only exist if this outlet temperature is below the bulk boiling point at chamber pressure (the coolant is injected into the chamber
# and must not flash there; the jacket is always above chamber pressure, so this also covers boiling in the channels), the maximum
# coolant-side wall temperature (coking) and the maximum coolant outlet temperature. No channel geometry changes this, only the heat load (chamber pressure
# and shape, mixture ratio, film cooling, c* efficiency) or the coolant mass flow do.
#
# Stage 2, best case geometry (fast model, seconds): the designs the optimiser ends up near when the limits are tight, minimum channel
# width and height at the maximum helix angle, for a range of channel counts, for each flow arrangement (one or two passes, coolant inlet
# at the injector or the nozzle). The closest of these to the limits shows how far off the request is. Optional throttle points are
# run with the best design of each arrangement.
#
# Usage:
#   python3 feasibility.py configs/a.py --max-dp 10 --max-angle 30 --min-rib 0.8 --min-channel-width 1 --min-channel-height 1
#   python3 feasibility.py configs/rp1.py --max-dp 20 --max-angle 30 --min-rib 0.8 --min-channel-width 1 --min-channel-height 1 \
#       --max-coolant-wall-temp 590 --throttle 0.7
#   python3 feasibility.py configs/a.py ... --quick          # energy budget only, for every wall material unless --material is given
#
# Results are printed and written to <out>/feasibility_<config>/feasibility.json

import argparse
import contextlib
import io
import json
import multiprocessing
import os
import sys
import tempfile
import time

import numpy as np

import MaterialLib
import Summary
import Units
from CaseSetup import load_config, read_inputs, case_name
from FastCooling import FastHeatTransfer, QuietOutput, CoolantTable
from optimize import design_geometry, label, levels, quiet_warnings

# flow arrangements: (two passes, start_idx)
ARRANGEMENTS = {'1p-inj': (False, 0), '1p-noz': (False, -1), '2p-inj': (True, 0), '2p-noz': (True, -1)}
MASS_FLOWS   = ('m_dot', 'm_dot_f', 'm_dot_ox', 'm_dot_coolant', 'm_dot_film')
MATERIALS    = ('AlSi10Mg', 'CuCr1Zr', 'IN718')     # energy budget only (--quick) runs all of these unless --material is given


def fmt_T(T):
    return '%.0f %s' % (Units.temperature(T), Units.temperature_unit()) if T is not None and np.isfinite(T) else '-'


##################################################
# cases
##################################################

def case_overrides(config_path, arrangement=None, throttle=1.0, throttle_inlet='hold'):
    # config overrides for a flow arrangement and a throttle point. Throttling scales all mass flows and the chamber pressure; the jacket
    # inlet pressure is held (throttle valve downstream of the jacket) or scaled as Pc + dp_feed with dp_feed ~ m_dot^2 (valve upstream)
    cfg = read_inputs(config_path)
    overrides = {}
    if arrangement is not None:
        two, start = ARRANGEMENTS[arrangement]
        overrides.update(two_pass=two, start_idx=start)
        if two and cfg.n % 2:
            overrides['n'] = cfg.n + 1
    if throttle != 1.0:
        for name in MASS_FLOWS:
            if hasattr(cfg, name):
                overrides[name] = getattr(cfg, name) * throttle
        overrides['Pc'] = cfg.Pc * throttle
        if throttle_inlet == 'scale':
            overrides['inlet_pressure'] = cfg.Pc * throttle + (cfg.inlet_pressure - cfg.Pc) * throttle**2
    return overrides


def build_case(config_path, overrides, material):
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = load_config(config_path, tempfile.mkdtemp(prefix='pyrocket_feasibility_'), overrides)
    cfg.material = material
    return cfg


def make_sim(cfg, cooling_geometry, material, table, out=None):
    return FastHeatTransfer(cea=cfg.cea, gas=cfg.gas, geometry=cfg.geometry, material=material, coolant=cfg.coolant, cooling_geometry=cooling_geometry,
                            m_dot=cfg.m_dot, m_dot_coolant=cfg.m_dot_coolant, T_amb=cfg.ambient_temp, output=out or QuietOutput(cfg.geometry, cfg.save_path),
                            settings2D=cfg.settings, model=cfg.hot_gas_method, cool_model=cfg.cooling_method, eta_c_star=cfg.eta_c_star, film=cfg.film,
                            bartz_knockdown=getattr(cfg, 'bartz_knockdown', 1.0), turnaround_loss=getattr(cfg, 'turnaround_loss', 1.5), table=table)


def saturation(cfg):
    # thermo Chemical of the dominant coolant species, for Tsat and the critical pressure; None for custom coolants
    try:
        import thermo
        return thermo.Chemical(cfg.coolant.IDs[int(np.argmax(cfg.coolant.ws))])
    except Exception:
        return None


##################################################
# stage 1: energy budget
##################################################

def energy_budget(cfg, material, table, limits):
    sim = make_sim(cfg, cfg.cooling_geom, material, table)
    r, x = cfg.geometry[:, 1], cfg.geometry[:, 0]
    ds = np.concatenate(([0.0], np.hypot(np.diff(x), np.diff(r))))
    Q_rad = float(sum(sim.radiation(i) * 2 * np.pi * r[i] * ds[i] for i in range(len(r))))
    Q_floor = sim.estimated_heat_load(T_wall=material.T_max) + Q_rad          # hot wall at the service limit: smallest possible heat load
    Q_600 = sim.estimated_heat_load(T_wall=600.0) + Q_rad                     # for reference, a typical copper hot wall temperature

    # coolant enthalpy rise from the inlet at the lowest jacket pressure the limits allow (lowest saturation temperature), but not below
    # chamber pressure, which the jacket must stay above to inject. From its own table at this pressure: interpolating between the widely
    # spaced pressure rows of the stage 2 table ends the liquid range early near saturation. The table needs two rows, the second 1 bar
    # higher is liquid wherever the first is
    P_low = max(cfg.inlet_pressure - limits['max_dp'], cfg.Pc, 1e5)
    curve = CoolantTable(cfg.cooling_fluid, cfg.fluid_mass_frac, cfg.inlet_temp, cfg.inlet_temp + 450.0, P_low, P_low + 1e5, n_T=451, n_P=2)
    T_grid, H = [cfg.inlet_temp], [0.0]
    for T in curve.T[1:]:
        state = curve.state(T, P_low)
        if state is None:
            break
        H.append(H[-1] + state.Cp * (T - T_grid[-1]))
        T_grid.append(T)
    T_grid, H = np.array(T_grid), np.array(H)
    T_liquid_end = float(T_grid[-1])                                          # end of the liquid (or compressed liquid) property range

    def outlet(Q):
        dH = Q / cfg.m_dot_coolant
        return float(np.interp(dH, H, T_grid)) if dH <= H[-1] else None       # None: leaves the liquid range first

    def allowed(T_limit):
        return float(np.interp(min(T_limit, T_liquid_end), T_grid, H)) * cfg.m_dot_coolant

    # bulk boiling at chamber pressure: the coolant must not flash when injected (the jacket is above Pc, so this covers the channels too)
    chem = saturation(cfg)
    subcritical = chem is not None and cfg.Pc < chem.Pc
    checks = []
    if subcritical and limits['min_bulk_boil_margin'] is not None:
        checks.append(('bulk boiling at Pc %.1f bar' % (cfg.Pc / 1e5), float(chem.Tsat(cfg.Pc)) - limits['min_bulk_boil_margin']))
    if limits['max_coolant_wall'] is not None:
        checks.append(('coolant-side wall (e.g. coking)', limits['max_coolant_wall']))
    if limits['max_coolant_out'] is not None:
        checks.append(('coolant outlet', limits['max_coolant_out']))
    # the models only handle liquid coolant; below saturation this is the boiling check again, so only shown when it ends earlier
    if not (subcritical and limits['min_bulk_boil_margin'] is not None and T_liquid_end >= float(chem.Tsat(cfg.Pc)) - 2.0):
        checks.append(('liquid range of the coolant model', T_liquid_end))

    T_out = outlet(Q_floor)
    rows = []
    for name, T_limit in checks:
        Q_allowed = allowed(T_limit)
        rows.append(dict(limit=name, T_limit=T_limit, Q_allowed=Q_allowed, ok=Q_floor <= Q_allowed,
                         margin=(T_limit - T_out) if T_out is not None else None,
                         m_dot_needed=Q_floor / (Q_allowed / cfg.m_dot_coolant) if Q_allowed > 0 else None))
    return dict(Q_floor=Q_floor, Q_600=Q_600, Q_rad=Q_rad, Q_per_kg=Q_floor / cfg.m_dot_coolant, T_out_floor=T_out, T_out_600=outlet(Q_600),
                T_liquid_end=T_liquid_end, P_low=P_low, Pc=cfg.Pc, supercritical=chem is not None and cfg.Pc >= chem.Pc, m_dot_coolant=cfg.m_dot_coolant,
                checks=rows, ok=all(r['ok'] for r in rows))


def print_budget(title, b, material):
    print('\n%s' % title)
    print('  heat load, hot wall at the %s limit (%s): %.0f kW, %.0f kJ per kg of coolant (%.0f kW with a %s hot wall)' % (
          material.name, fmt_T(material.T_max), b['Q_floor'] / 1e3, b['Q_per_kg'] / 1e3, b['Q_600'] / 1e3, fmt_T(600.0)))
    if b['T_out_floor'] is None:
        outlet = 'above %s, the coolant leaves the liquid range' % fmt_T(b['T_liquid_end'])
    else:
        outlet = '%s (%s with a %s hot wall)' % (fmt_T(b['T_out_floor']), fmt_T(b['T_out_600']) if b['T_out_600'] else 'beyond the liquid range', fmt_T(600.0))
    print('  lowest possible bulk outlet: %s, coolant %.3f kg/s%s' % (outlet, b['m_dot_coolant'],
          ', chamber above the critical pressure (no bulk boiling)' if b['supercritical'] else ''))
    for r in b['checks']:
        extra = '' if r['ok'] else ', needs %.0f%% less heat load or %.3f kg/s coolant' % (100 * (1 - r['Q_allowed'] / b['Q_floor']), r['m_dot_needed'])
        print('  %-4s %-34s limit %8s  margin %8s  heat load allowed %4.0f kW%s' % (
              'ok' if r['ok'] else 'FAIL', r['limit'], fmt_T(r['T_limit']), ('%+.0f K' % r['margin']) if r['margin'] is not None else '-',
              r['Q_allowed'] / 1e3, extra))


##################################################
# stage 2: best case geometries with the fast model
##################################################

WORKER = {}


def init_worker(config_path, material_name, table, limits, throttle_inlet):
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    quiet_warnings()
    WORKER.update(config_path=config_path, material=getattr(MaterialLib, material_name), table=table, limits=limits, throttle_inlet=throttle_inlet, cases={})


def worker_case(arrangement, throttle):
    key = (arrangement, throttle)
    if key not in WORKER['cases']:
        overrides = case_overrides(WORKER['config_path'], arrangement, throttle, WORKER['throttle_inlet'])
        WORKER['cases'][key] = build_case(WORKER['config_path'], overrides, WORKER['material'])
    return WORKER['cases'][key]


def excess(r, limits, material):
    # how far a design is beyond its most violated temperature limit [K], negative when all are met
    values = [r['T_wall_max'] - material.T_max]
    if limits['max_coolant_wall'] is not None:
        values.append(r['T_wall_coolant_max'] - limits['max_coolant_wall'])
    if limits['max_coolant_out'] is not None:
        values.append(r['T_out'] - limits['max_coolant_out'])
    if limits['min_bulk_boil_margin'] is not None and r['bulk_margin'] is not None:
        values.append(limits['min_bulk_boil_margin'] - r['bulk_margin'])
    return float(max(values))


def evaluate(task):
    arrangement, throttle, d = task
    limits, material = WORKER['limits'], WORKER['material']
    result = dict(arrangement=arrangement, throttle=throttle, design=d, label=label(d))
    try:
        cfg = worker_case(arrangement, throttle)
        cg, _ = design_geometry(cfg, d)
        out = QuietOutput(cfg.geometry, cfg.save_path)
        sim = make_sim(cfg, cg, material, WORKER['table'], out)
        with contextlib.redirect_stdout(io.StringIO()):
            sim.run()
    except Exception as e:
        result.update(status='failed', note=repr(e)[:160])
        return result

    two = cg.passes == 2
    outlet = (0 if cfg.start_idx == 0 else -1) if two else (0 if cfg.start_idx == -1 else -1)
    T_out, P_out = (out.T_c2[outlet], out.P_c2[outlet]) if two else (out.T_c[outlet], out.P_c[outlet])
    x = cfg.geometry[:, 0]
    walls = [(out.T_wall_c, out.T_c, out.P_c)] + ([(out.T_wall_c2, out.T_c2, out.P_c2)] if two else [])
    i_pass, i_x = max(((p, int(np.argmax(w[0]))) for p, w in enumerate(walls)), key=lambda pi: walls[pi[0]][0][pi[1]])
    # bulk boiling at chamber pressure, where the coolant is injected, and in the channels in case the jacket drops below Pc
    T_bulk = np.concatenate([T for _, T, _ in walls])
    bulk = np.concatenate([Summary.saturation_margin(cfg, [np.max(T_bulk)], [cfg.Pc])] +
                          [Summary.saturation_margin(cfg, T, P) for _, T, P in walls])
    r, ds = cfg.geometry[:, 1], np.concatenate(([0.0], np.hypot(np.diff(x), np.diff(cfg.geometry[:, 1]))))
    result.update(status='ok', dp=float(cfg.inlet_pressure - P_out), T_out=float(T_out), T_wall_max=float(np.max(out.T_wall_i)),
                  T_wall_coolant_max=float(walls[i_pass][0][i_x]), x_T_wall_coolant_max=float(x[i_x]), v_max=float(np.max(out.v_coolant)),
                  bulk_margin=float(np.nanmin(bulk)) if np.any(np.isfinite(bulk)) else None,
                  Q=float(np.sum(np.asarray(out.q) * 2 * np.pi * r * ds)))
    result['excess'] = excess(result, limits, material)
    result['dp_ok'] = result['dp'] <= limits['max_dp']
    return result


def aggressive_designs(cfg, arrangement, limits, material_name):
    # minimum channel width (and 1.5 x) and height at the maximum helix angle, channel counts from the most that fit the throat down
    two = ARRANGEMENTS[arrangement][0]
    r_t = float(np.min(cfg.geometry[:, 1]))
    circumference = 2 * np.pi * (r_t + limits['wall']) * np.cos(np.radians(limits['max_angle']))
    n_max = int(circumference / (limits['min_channel'] + limits['min_rib']))
    n_values = sorted({int(round(v)) for v in levels(max(4, int(round(n_max * 0.35))), n_max, 6)})
    if two:
        n_values = sorted({n - n % 2 for n in n_values if n >= 4})
    designs = []
    for mode in ('constant_width', 'constant_fill'):
        for n in n_values:
            for width in (limits['min_channel'], 1.5 * limits['min_channel']):
                if circumference / n - limits['min_rib'] >= width - 1e-12:
                    designs.append(dict(material=material_name, mode=mode, n=n, width=width, height=limits['min_height'], wall=limits['wall'],
                                        angle=limits['max_angle']))
    return designs


def pick_best(rows):
    # closest to the limits among the designs within the pressure drop limit, else the lowest pressure drop
    ok = [r for r in rows if r['status'] == 'ok']
    within = [r for r in ok if r['dp_ok']]
    if within:
        return min(within, key=lambda r: r['excess'])
    return min(ok, key=lambda r: r['dp']) if ok else None


def print_designs(title, best, limits):
    print('\n%s' % title)
    print('  %-7s %-38s %8s %9s %11s %8s %8s %8s  %s' % ('arrange', 'best design', 'dp [bar]', 'hot wall', 'cool. wall', 'at x mm', 'outlet', 'excess', 'note'))
    for arrangement, r in best.items():
        if r is None:
            print('  %-7s no design ran' % arrangement)
            continue
        note = '' if r['dp_ok'] else 'pressure drop above %.1f bar for every design' % (limits['max_dp'] / 1e5)
        if r.get('bulk_margin') is not None and r['bulk_margin'] < 0:
            note = (note + ', ' if note else '') + 'bulk boiling (%.0f K)' % r['bulk_margin']
        print('  %-7s %-38s %8.1f %9s %11s %8.0f %8s %+7.0f K  %s' % (arrangement, r['label'], r['dp'] / 1e5, fmt_T(r['T_wall_max']),
              fmt_T(r['T_wall_coolant_max']), r['x_T_wall_coolant_max'] * 1e3, fmt_T(r['T_out']), r['excess'], note))


##################################################
# main
##################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick feasibility screen of a cooling request: energy budget, then the best case channel geometries with the fast model.')
    parser.add_argument('config', help='config file')
    parser.add_argument('--max-dp', type=float, required=True, help='maximum coolant pressure drop [bar]')
    parser.add_argument('--max-angle', type=float, default=0.0, help='maximum helix angle from the meridian [deg] (default 0, axial channels)')
    parser.add_argument('--min-rib', type=float, required=True, help='minimum rib width normal to the flow [mm]')
    parser.add_argument('--min-channel-width', type=float, required=True, help='minimum channel width normal to the flow [mm]')
    parser.add_argument('--min-channel-height', type=float, required=True, help='minimum channel height [mm]')
    parser.add_argument('--wall', type=float, default=None, help="hot wall thickness [mm] (default: the config's t_w_i)")
    parser.add_argument('--material', default=None, choices=MATERIALS, help="wall material (default: the config's; with --quick all of them)")
    parser.add_argument('--max-coolant-wall-temp', type=float, default=None, help='maximum coolant-side (wetted) wall temperature [K], e.g. RP-1 coking')
    parser.add_argument('--max-coolant-outlet-temp', type=float, default=None, help='maximum coolant temperature at the jacket outlet [K]')
    parser.add_argument('--min-bulk-boil-margin', type=float, default=0.0,
                        help='minimum margin of the bulk coolant below its saturation temperature at chamber pressure [K] (default 0: no flashing at injection). Ignored above the critical pressure')
    parser.add_argument('--allow-bulk-boiling', action='store_true', help='do not limit bulk boiling')
    parser.add_argument('--arrangements', nargs='+', default=['all'], choices=['all', 'config'] + list(ARRANGEMENTS),
                        help="flow arrangements for stage 2: 1p/2p = one/two passes, inj/noz = coolant inlet at the injector/nozzle (default: all)")
    parser.add_argument('--throttle', nargs='*', type=float, default=[], help='throttle points as fractions of the design mass flow, e.g. 0.7 0.5')
    parser.add_argument('--throttle-inlet', choices=('hold', 'scale'), default='hold',
                        help='jacket inlet pressure when throttled: held (valve after the jacket, default) or scaled with the feed pressure drop (valve before the jacket)')
    parser.add_argument('--marginal', type=float, default=15.0, help='excess over the limits [K] up to which a request counts as marginal (default 15)')
    parser.add_argument('--quick', action='store_true', help='energy budget only, for every wall material unless --material is given')
    parser.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1), help='parallel processes (default: all CPUs but one)')
    parser.add_argument('--out', default='results', help='output folder (default: results)')
    args = parser.parse_args()
    quiet_warnings()
    start = time.time()

    mm = 1e-3
    base = read_inputs(args.config)
    if args.wall is None:
        if not isinstance(base.t_w_i, (int, float)):
            sys.exit("The config's t_w_i is a function of x, give the hot wall thickness with --wall")
        args.wall = base.t_w_i / mm
    material_name = args.material or base.material.name
    if not (args.quick and args.material is None) and (not hasattr(MaterialLib, material_name) or getattr(getattr(MaterialLib, material_name), 'T_max', None) is None):
        sys.exit('Material %s has no maximum service temperature in MaterialLib, choose one with --material' % material_name)
    material = getattr(MaterialLib, material_name, base.material)
    limits = {'max_dp': args.max_dp * 1e5, 'max_angle': args.max_angle, 'min_rib': args.min_rib * mm, 'min_channel': args.min_channel_width * mm,
              'min_height': args.min_channel_height * mm, 'wall': args.wall * mm, 'max_coolant_wall': args.max_coolant_wall_temp,
              'max_coolant_out': args.max_coolant_outlet_temp, 'min_bulk_boil_margin': None if args.allow_bulk_boiling else args.min_bulk_boil_margin}

    name = case_name(args.config)
    folder = os.path.join(args.out, 'feasibility_' + name)
    os.makedirs(folder, exist_ok=True)
    print('Feasibility of %s: %s, hot wall %.2f mm, dp <= %.1f bar, helix <= %.0f deg, channels >= %.2f x %.2f mm, ribs >= %.2f mm' % (
          name, material_name, args.wall, args.max_dp, args.max_angle, args.min_channel_width, args.min_channel_height, args.min_rib), flush=True)

    # coolant property table, as in optimize.py
    cfg = build_case(args.config, {}, material)
    inlets = [cfg.inlet_pressure] + [case_overrides(args.config, None, t, args.throttle_inlet).get('inlet_pressure', cfg.inlet_pressure) for t in args.throttle]
    P_lo = max(min(inlets) - 3 * limits['max_dp'] - 5e5, 1e5)
    table = CoolantTable(cfg.cooling_fluid, cfg.fluid_mass_frac, cfg.inlet_temp - 5.0, cfg.inlet_temp + 450.0, P_lo, cfg.inlet_pressure + 2e5)

    # stage 1, for the wall material (energy budget only: for every material, the floor heat load depends on its service temperature).
    # budget: the stage 2 material's, budgets: per material
    budget_materials = list(MATERIALS) if args.quick and args.material is None else [material_name]
    report = {'config': os.path.abspath(args.config), 'material': material_name, 'materials': budget_materials, 'limits': limits, 'args': vars(args),
              'budget': {}, 'budgets': {m: {} for m in budget_materials}, 'designs': {}}
    throttles = [1.0] + [t for t in args.throttle if t != 1.0]
    for t in throttles:
        cfg_t = cfg if t == 1.0 else build_case(args.config, case_overrides(args.config, None, t, args.throttle_inlet), material)
        for m in budget_materials:
            b = energy_budget(cfg_t, getattr(MaterialLib, m), table, limits)
            report['budgets'][m][str(t)] = b
            print_budget('Stage 1, energy budget%s%s' % ('' if t == 1.0 else ' at %.0f%% throttle' % (100 * t), '' if len(budget_materials) == 1 else ', ' + m),
                         b, getattr(MaterialLib, m))
    report['budget'] = report['budgets'].get(material_name) or report['budgets'][budget_materials[0]]
    budget_ok = report['budget']['1.0']['ok']

    # stage 2
    best_all = {}
    if not args.quick:
        arrangements = list(ARRANGEMENTS) if 'all' in args.arrangements else sorted({
            a for a in args.arrangements if a != 'config'} | ({[k for k, v in ARRANGEMENTS.items() if v == (bool(getattr(base, 'two_pass', False)), base.start_idx)][0]}
                                                             if 'config' in args.arrangements else set()))
        tasks = []
        for a in arrangements:
            cfg_a = build_case(args.config, case_overrides(args.config, a), material)
            tasks += [(a, 1.0, d) for d in aggressive_designs(cfg_a, a, limits, material_name)]
        print('\nStage 2: %d best case designs (%s), fast model' % (len(tasks), ', '.join(arrangements)), flush=True)

        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(args.jobs, initializer=init_worker, initargs=(args.config, material_name, table, limits, args.throttle_inlet)) as pool:
            rows = []
            for i, r in enumerate(pool.imap_unordered(evaluate, tasks), 1):
                rows.append(r)
                if i % max(1, len(tasks) // 10) == 0 or i == len(tasks):
                    print('  designs evaluated %d / %d' % (i, len(tasks)), flush=True)
            best = {a: pick_best([r for r in rows if r['arrangement'] == a]) for a in arrangements}
            best_all['1.0'] = best
            print_designs('Stage 2, closest design to the limits per arrangement (excess: K beyond the most violated temperature limit)', best, limits)

            for t in throttles[1:]:
                picks = [(a, t, r['design']) for a, r in best.items() if r is not None]
                throttled = {r['arrangement']: r for r in pool.map(evaluate, picks)}
                best_all[str(t)] = {a: throttled.get(a) for a in arrangements}
                print_designs('Same designs at %.0f%% throttle (jacket inlet %s)' % (100 * t, 'held' if args.throttle_inlet == 'hold' else 'scaled'), best_all[str(t)], limits)
        report['designs'] = best_all

    # verdict
    print('')
    failed = {m: [r['limit'] for r in report['budgets'][m]['1.0']['checks'] if not r['ok']] for m in budget_materials}
    if len(budget_materials) > 1:
        passed = [m for m in budget_materials if not failed[m]]
        why = '; '.join('%s breaks the %s limit' % (m, ', '.join(failed[m])) for m in budget_materials if failed[m])
        if not passed:
            verdict = 'NOT FEASIBLE with any wall material: %s. Reduce the heat load or raise the coolant flow' % why
        else:
            verdict = 'ENERGY BUDGET OK with %s%s: run without --quick and --material to check the best case geometries' % (
                      ', '.join(passed), ' (%s)' % why if why else '')
    elif not budget_ok:
        verdict = 'NOT FEASIBLE: the heat load alone breaks the %s limit, no channel design can meet it. Reduce the heat load or raise the coolant flow' % ', '.join(failed[budget_materials[0]])
    elif args.quick:
        verdict = 'ENERGY BUDGET OK: run without --quick to check the best case geometries'
    else:
        results = [r for r in best_all['1.0'].values() if r is not None and r['dp_ok']]
        if not results:
            verdict = 'NOT FEASIBLE: every best case design exceeds the pressure drop limit'
        else:
            top = min(results, key=lambda r: r['excess'])
            if top['excess'] <= 0:
                verdict = 'FEASIBLE: %s %s meets all limits with %.0f K to spare, run optimize.py with this arrangement' % (top['arrangement'], top['label'], -top['excess'])
            elif top['excess'] <= args.marginal:
                verdict = 'MARGINAL: the closest design (%s %s) misses by %.0f K, the optimiser may close it, film or chamber changes likely needed' % (
                          top['arrangement'], top['label'], top['excess'])
            else:
                verdict = 'UNLIKELY: even the most aggressive geometry (%s %s) misses by %.0f K, change the heat load, coolant flow or limits first' % (
                          top['arrangement'], top['label'], top['excess'])
        for t in throttles[1:]:
            rows_t = [r for r in best_all.get(str(t), {}).values() if r is not None and r['status'] == 'ok']
            if rows_t:
                worst = min(rows_t, key=lambda r: r['excess'])
                verdict += '\n  at %.0f%% throttle the best of these designs is %+.0f K from the limits' % (100 * t, worst['excess'])
    print(verdict)
    report['verdict'] = verdict

    with open(os.path.join(folder, 'feasibility.json'), 'w') as f:
        json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print('Results in %s (%.0f s)' % (folder, time.time() - start))
