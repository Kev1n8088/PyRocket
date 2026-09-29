#####################################################################
#                           PyRocket								#
# 2D Regenertive Cooling Simulation for Bipropellant Rocket Engines #
#                                                                   #
# Creator:  Joanthan Neeser                                         #
# Date:     15.12.2022                                              #
# Version:  2.3  													#
# License:	GNU GENERAL PUBLIC LICENSE V3							#
#####################################################################

# Helix sweep: finds the helix pitch (or angle) with the lowest maximum inner wall temperature, subject to a maximum
# pitch/angle, a maximum coolant pressure drop, and optionally a maximum coolant-side wall temperature and minimum channel and rib widths.
#
# Usage:
#   python3 sweep.py configs/rp1_lox.py --param pitch --max 3 --max-dp 10
#   python3 sweep.py configs/rp1_lox.py --param angle --max 50 --max-dp 8 --min-channel-width 0.8 --min-rib-width 0.8 --jobs 4
#
# The channel count n and fill factor psi of the config are kept. psi is the fraction of the circumference covered by channels
# in a cut normal to the engine axis, so with increasing helix angle the channels and ribs get narrower normal to the flow path
# (by cos(alpha)), which raises the coolant velocity and pressure drop and is what the width limits act on.
#
# The search starts with an even grid from 0 (axial channels) to the maximum, then refines around the best feasible point.
# Each point is a full simulation, written to <out>/sweep_<config>_<param>/<param>_<value>/. Finished points are reused
# when the sweep is repeated with an unchanged config, use --fresh to run them again.

import argparse
import hashlib
import multiprocessing
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import Summary
import Units
from CaseSetup import read_inputs, build_geometry, case_name
from run import run_case


# swept parameters: axis label, folder name format and the config inputs they set
PARAMS = {
    'pitch': {'label': 'Helix pitch [deg/mm]',               'fmt': '%.4f', 'overrides': lambda v: {'helix_pitch_deg_per_mm': float(v)}},
    'angle': {'label': 'Helix angle from meridian [deg]',    'fmt': '%.3f', 'overrides': lambda v: {'helix_angle': float(v), 'helix_pitch_deg_per_mm': None}},
}


def channel_widths(config_path, param, value):
    # narrowest channel and rib width normal to the flow along the contour [m] and largest helix angle [deg], taken at the
    # channel bottom (inner radius) where both are smallest
    cfg = read_inputs(config_path, PARAMS[param]['overrides'](value))
    _, cg = build_geometry(cfg)
    return np.min(cg.psi_c_n * cg.r_i), np.min(cg.psi_w_n * cg.r_i), np.degrees(np.max(cg.alpha))


def geometric_limit(config_path, param, v_max, min_channel, min_rib):
    # largest parameter value in [0, v_max] that keeps the channel and rib widths above their minimum. Widths shrink
    # monotonically with the helix angle (cos(alpha)), so bisection applies
    def ok(v):
        w_c, w_r, _ = channel_widths(config_path, param, v)
        return w_c >= min_channel and w_r >= min_rib

    if not ok(0.0):
        return None
    if ok(v_max):
        return v_max
    lo, hi = 0.0, v_max
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


class HelixSweep():
    def __init__(self, config_path, param, v_max, max_dp, min_channel, min_rib, out, jobs, fresh, section_images=False, logs=False, max_coolant_wall=None):
        self.config_path = config_path
        self.param       = param
        self.v_max       = v_max                          # maximum pitch [deg/mm] or angle [deg]
        self.max_dp      = max_dp                         # maximum coolant pressure drop [Pa]
        self.max_coolant_wall = max_coolant_wall          # maximum coolant-side wall temperature [K], None: not limited
        self.min_channel = min_channel                    # minimum channel width normal to the flow [m]
        self.min_rib     = min_rib                        # minimum rib width normal to the flow [m]
        self.jobs        = jobs
        self.fresh       = fresh
        self.section_images = section_images             # keep the per-section temperature images of the config
        self.logs        = logs                           # write the output of every point to its log.txt
        self.folder      = os.path.join(out, 'sweep_%s_%s' % (case_name(config_path), param))
        self.points      = {}                             # parameter value -> case summary

        with open(config_path, 'rb') as f:
            self.config_hash = hashlib.sha256(f.read()).hexdigest()

    def name(self, value):
        return self.param + '_' + PARAMS[self.param]['fmt'] % value

    def feasible(self, case):
        return case.get('status') == 'ok' and case['dp_coolant'] <= self.max_dp and (
            self.max_coolant_wall is None or case['T_wall_coolant_max'] <= self.max_coolant_wall)

    def violation(self, case):
        # the limits a finished point exceeds, for the table
        return ', '.join(n for n, v in (('dp > max', case['dp_coolant'] > self.max_dp),
                                        ('coolant wall > max', self.max_coolant_wall is not None and case['T_wall_coolant_max'] > self.max_coolant_wall)) if v)

    def evaluate(self, values):
        # run the simulations for the given parameter values in parallel, reusing finished points of an earlier sweep
        todo = []
        for v in [float(v) for v in values]:
            save_path = os.path.join(self.folder, self.name(v))
            key = '%s|%s|%r' % (self.config_hash, self.param, v)
            if not self.fresh and os.path.isfile(os.path.join(save_path, 'summary.json')):
                case = Summary.read_case(save_path)
                if case.get('sweep_key') == key and case.get('status') == 'ok' and 'T_wall_coolant_max' in case:
                    self.points[v] = case
                    continue

            w_c, w_r, alpha = channel_widths(self.config_path, self.param, v)
            extra = {'sweep_key': key, 'sweep_param': self.param, 'sweep_value': v, 'alpha_max_deg': alpha,
                     'w_channel_min': w_c, 'w_rib_min': w_r}
            overrides = PARAMS[self.param]['overrides'](v)
            if not self.section_images:
                overrides['save_fig'] = False
            todo.append((self.config_path, save_path, self.jobs > 1 or self.logs, overrides, extra))

        if todo:
            print('Running %d point(s): %s' % (len(todo), ', '.join(self.name(t[4]['sweep_value']) for t in todo)))
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(processes=max(1, min(self.jobs, len(todo))), maxtasksperchild=1) as pool:
                for case in pool.starmap(run_case, todo):
                    self.points[case['sweep_value']] = case

    def best(self):
        feasible = [v for v, c in self.points.items() if self.feasible(c)]
        return min(feasible, key=lambda v: self.points[v]['T_wall_max']) if feasible else None

    def refine_candidates(self, v_best, min_spacing):
        # new points on both sides of the best feasible point. Towards an infeasible neighbour (pressure drop above the limit)
        # the point is placed where the pressure drop is interpolated to reach the limit, as the optimum is often on that boundary
        values = sorted(self.points)
        i = values.index(v_best)
        candidates = []
        if i > 0:
            candidates.append(0.5 * (values[i-1] + v_best))
        if i < len(values) - 1:
            v_r, c_r, c_b = values[i+1], self.points[values[i+1]], self.points[v_best]
            if c_r.get('status') == 'ok' and not self.feasible(c_r) and c_r['dp_coolant'] > c_b['dp_coolant']:
                t = (self.max_dp - c_b['dp_coolant']) / (c_r['dp_coolant'] - c_b['dp_coolant'])
                candidates.append(v_best + np.clip(t, 0.1, 0.9) * (v_r - v_best))
            else:
                candidates.append(0.5 * (v_best + v_r))
        return [v for v in candidates if min(abs(v - p) for p in values) > min_spacing]

    def run(self, n_points, n_refine, v_hi):
        if v_hi <= 0:
            # the width limits only allow axial channels
            self.evaluate([0.0])
            return self.best()
        self.evaluate(list(np.linspace(0.0, v_hi, n_points)))
        for _ in range(n_refine):
            v_best = self.best()
            if v_best is None:
                break
            candidates = self.refine_candidates(v_best, min_spacing=1e-3 * v_hi)
            if not candidates:
                break
            self.evaluate(candidates)
        return self.best()


def plot_sweep(sweep, v_hi, v_best, path):
    # maximum wall temperature, pressure drop and channel widths over the swept parameter, one panel per quantity
    values = sorted(sweep.points)
    ok = [v for v in values if sweep.points[v].get('status') == 'ok']
    feas = [v for v in ok if sweep.feasible(sweep.points[v])]
    infeas = [v for v in ok if not sweep.feasible(sweep.points[v])]
    unit = Units.temperature_unit()
    T = lambda vs: [Units.temperature(sweep.points[v]['T_wall_max']) for v in vs]
    dp = lambda vs: [sweep.points[v]['dp_coolant'] / 1e5 for v in vs]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, facecolor=Summary.SURFACE, gridspec_kw={'hspace': 0.3})
    for ax in axes:
        Summary.style_axes(ax)

    for ax, f, title in ((axes[0], T, 'Max inner wall temperature [%s]' % unit), (axes[1], dp, 'Coolant pressure drop [bar]')):
        ax.plot(ok, f(ok), color=Summary.SERIES[0], linewidth=2, zorder=2)
        ax.plot(feas, f(feas), 'o', color=Summary.SERIES[0], markersize=8, markeredgecolor=Summary.SURFACE, markeredgewidth=1.5, zorder=3)
        ax.plot(infeas, f(infeas), 'o', color=Summary.SURFACE, markersize=8, markeredgecolor=Summary.SERIES[0], markeredgewidth=2, zorder=3)
        ax.set_title(title, fontsize=10, loc='left')

    axes[1].axhline(sweep.max_dp / 1e5, color=Summary.TEXT_2, linewidth=1, linestyle='--')
    axes[1].text(0.01, sweep.max_dp / 1e5, ' max pressure drop %.3g bar' % (sweep.max_dp / 1e5), transform=axes[1].get_yaxis_transform(),
                 va='bottom', fontsize=8, color=Summary.TEXT_2)

    if v_best is not None:
        c = sweep.points[v_best]
        for ax, y in ((axes[0], Units.temperature(c['T_wall_max'])), (axes[1], c['dp_coolant'] / 1e5)):
            ax.plot([v_best], [y], 'o', color=Summary.TEXT, markersize=11, markerfacecolor='none', markeredgewidth=2, zorder=4)
        axes[0].annotate('best: %s = %.4g\n%.1f %s, %.2f bar' % (sweep.param, v_best, Units.temperature(c['T_wall_max']), unit, c['dp_coolant'] / 1e5),
                         (v_best, Units.temperature(c['T_wall_max'])), textcoords='offset points', xytext=(10, 12), fontsize=8, color=Summary.TEXT)

    # channel and rib widths normal to the flow, evaluated densely from the geometry alone
    v_dense = np.linspace(0.0, sweep.v_max, 60)
    widths = np.array([channel_widths(sweep.config_path, sweep.param, v)[:2] for v in v_dense]) * 1e3
    axes[2].plot(v_dense, widths[:,0], color=Summary.SERIES[0], linewidth=2, label='channel')
    axes[2].plot(v_dense, widths[:,1], color=Summary.SERIES[1], linewidth=2, label='rib')
    for w, label in ((widths[-1,0], 'channel'), (widths[-1,1], 'rib')):
        axes[2].annotate(label, (v_dense[-1], w), textcoords='offset points', xytext=(4, 0), va='center', fontsize=8, color=Summary.TEXT_2)
    for limit, label in ((sweep.min_channel, 'min channel'), (sweep.min_rib, 'min rib')):
        if limit > 0:
            axes[2].axhline(limit * 1e3, color=Summary.TEXT_2, linewidth=1, linestyle='--')
            axes[2].text(0.01, limit * 1e3, ' %s %.3g mm' % (label, limit * 1e3), transform=axes[2].get_yaxis_transform(), va='bottom', fontsize=8, color=Summary.TEXT_2)
    axes[2].legend(frameon=False, fontsize=8, loc='upper right', labelcolor=Summary.TEXT)
    axes[2].set_title('Narrowest width normal to the flow [mm]', fontsize=10, loc='left')
    axes[2].set_xlabel(PARAMS[sweep.param]['label'], fontsize=9, color=Summary.TEXT_2)

    if v_hi < sweep.v_max:
        for ax in axes:
            ax.axvline(v_hi, color=Summary.BASELINE, linewidth=1, linestyle=':')
        axes[2].text(v_hi, 0.98, ' width limit', transform=axes[2].get_xaxis_transform(), va='top', fontsize=8, color=Summary.TEXT_2)

    fig.suptitle('Helix sweep, %s  (filled: feasible, open: pressure drop or coolant-side wall above limit)' % case_name(sweep.config_path),
                 x=0.125, ha='left', fontsize=11, color=Summary.TEXT)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=Summary.SURFACE)
    plt.close(fig)


def print_table(sweep, v_best):
    unit = Units.temperature_unit().replace('°', '')
    print('')
    print('%-16s %9s %10s %9s %11s %9s %9s' % (sweep.param, 'alpha max', 'Tw max[%s]' % unit, 'dp [bar]', 'channel[mm]', 'rib [mm]', ''))
    for v in sorted(sweep.points):
        c = sweep.points[v]
        if c.get('status') != 'ok':
            print('%-16.4g %s %s' % (v, 'failed', c.get('error', '')))
            continue
        flag = 'best' if v == v_best else sweep.violation(c)
        print('%-16.4g %9.1f %10.1f %9.2f %11.3f %9.3f %9s' % (v, c['alpha_max_deg'], Units.temperature(c['T_wall_max']), c['dp_coolant'] / 1e5,
                                                              c['w_channel_min'] * 1e3, c['w_rib_min'] * 1e3, flag))
    print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find the helix pitch or angle with the lowest maximum inner wall temperature.')
    parser.add_argument('config', help='config file of the case to sweep')
    parser.add_argument('--param', choices=sorted(PARAMS), default='pitch', help='sweep the helix pitch [deg/mm] or the helix angle from the meridian [deg] (default: pitch)')
    parser.add_argument('--max', type=float, required=True, help='maximum pitch [deg/mm] or angle [deg]')
    parser.add_argument('--max-dp', type=float, required=True, help='maximum coolant pressure drop [bar]')
    parser.add_argument('--max-coolant-wall-temp', type=float, default=None,
                        help='maximum temperature of the coolant-side (wetted) wall [K], e.g. to limit RP-1 coking (default: not limited)')
    parser.add_argument('--min-channel-width', type=float, default=0.0, help='minimum channel width normal to the flow [mm] (default: none)')
    parser.add_argument('--min-rib-width', type=float, default=0.0, help='minimum rib width normal to the flow [mm] (default: none)')
    parser.add_argument('--points', type=int, default=7, help='points of the initial grid from 0 to max (default: 7)')
    parser.add_argument('--refine', type=int, default=3, help='refinement rounds around the best point, up to 2 points each (default: 3)')
    parser.add_argument('--jobs', type=int, default=1, help='number of simulations to run in parallel (default: 1)')
    parser.add_argument('--out', default='results', help='output folder (default: results)')
    parser.add_argument('--fresh', action='store_true', help='run all points again instead of reusing finished ones')
    parser.add_argument('--section-images', action='store_true', help='keep the per-section temperature images of the config (off by default, they take about 70 ms per section)')
    parser.add_argument('--logs', action='store_true', help='write the output of every point to its log.txt, also when running one point at a time')
    args = parser.parse_args()

    if args.param == 'angle' and not 0 < args.max < 90:
        sys.exit('The maximum helix angle must be between 0 and 90 deg')
    if args.max <= 0 or args.points < 2:
        sys.exit('--max must be positive and --points at least 2')

    if args.jobs > 1:
        # one thread per simulation, avoids oversubscribing the cpu with parallel cases
        for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            os.environ.setdefault(var, '1')

    # the width limits cap the search range before any simulation is run
    min_channel, min_rib = args.min_channel_width * 1e-3, args.min_rib_width * 1e-3
    v_hi = geometric_limit(args.config, args.param, args.max, min_channel, min_rib)
    if v_hi is None:
        w_c, w_r, _ = channel_widths(args.config, args.param, 0.0)
        sys.exit('Even axial channels violate the width limits: channel %.3f mm, rib %.3f mm' % (w_c * 1e3, w_r * 1e3))
    if v_hi < args.max:
        print('Width limits cap the %s at %.4g (requested maximum %.4g)' % (args.param, v_hi, args.max))

    sweep = HelixSweep(args.config, args.param, args.max, args.max_dp * 1e5, min_channel, min_rib, args.out, args.jobs, args.fresh, args.section_images, args.logs,
                       args.max_coolant_wall_temp)
    v_best = sweep.run(args.points, args.refine, v_hi)

    os.makedirs(sweep.folder, exist_ok=True)
    cases = [sweep.points[v] for v in sorted(sweep.points)]
    Summary.write_csv(cases, os.path.join(sweep.folder, 'sweep.csv'))
    plot_sweep(sweep, v_hi, v_best, os.path.join(sweep.folder, 'sweep.png'))
    Summary.plot_summary(cases, sweep.folder, os.path.join(sweep.folder, 'sweep_summary.png'), ordered=True,
                         heading='Helix sweep, %s' % case_name(args.config))
    print_table(sweep, v_best)

    if v_best is None:
        print('No feasible point: every evaluated %s exceeds the pressure drop (%.3g bar)%s limit' % (
            args.param, args.max_dp, ' or coolant-side wall temperature (%.1f K)' % args.max_coolant_wall_temp if args.max_coolant_wall_temp is not None else ''))
        sys.exit(1)

    c = sweep.points[v_best]
    active = []
    if c['dp_coolant'] > 0.95 * sweep.max_dp:
        active.append('pressure drop')
    if sweep.max_coolant_wall is not None and c['T_wall_coolant_max'] > sweep.max_coolant_wall - 5.0:
        active.append('coolant-side wall temperature')
    if v_best >= 0.999 * v_hi:
        active.append('width limit' if v_hi < args.max else 'maximum %s' % args.param)
    print('Best %s: %.4g  (max helix angle %.1f deg), max inner wall temp %.1f %s, pressure drop %.2f bar%s' % (
        args.param, v_best, c['alpha_max_deg'], Units.temperature(c['T_wall_max']), Units.temperature_unit(), c['dp_coolant'] / 1e5,
        ('; limited by ' + ', '.join(active)) if active else ''))
    print('Set helix_pitch_deg_per_mm = %.4g in the config' % v_best if args.param == 'pitch' else 'Set helix_angle = %.4g and helix_pitch_deg_per_mm = None in the config' % v_best)
    print('Results in ' + sweep.folder)
