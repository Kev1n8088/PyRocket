#####################################################################
#                           PyRocket								#
# 2D Regenertive Cooling Simulation for Bipropellant Rocket Engines #
#                                                                   #
# Creator:  Joanthan Neeser                                         #
# Date:     15.12.2022                                              #
# Version:  2.3  													#
# License:	GNU GENERAL PUBLIC LICENSE V3							#
#####################################################################

import csv
import json
import os
import numpy as np
import matplotlib.pyplot as plt

import Units

trapezoid = getattr(np, 'trapezoid', None) or np.trapz

# critical statistics shown in the summary: key in the case summary, bar chart title, display unit, conversion from SI
METRICS = [
    ('T_wall_max',       'Max inner wall temp',        Units.temperature_unit(), Units.temperature),
    ('T_wall_median',    'Median inner wall temp',     Units.temperature_unit(), Units.temperature),
    ('T_wall_throat',    'Inner wall temp at throat',  Units.temperature_unit(), Units.temperature),
    ('T_coolant_out',    'Coolant outlet temp',        Units.temperature_unit(), Units.temperature),
    ('dp_coolant',       'Coolant pressure drop',      'bar',    lambda v: v / 1e5),
    ('dp_injector',      'Coolant outlet P minus Pc',  'bar',    lambda v: v / 1e5),
    ('q_max',            'Peak heat flux',             'MW/m²',  lambda v: v / 1e6),
    ('Q_total',          'Total heat load',            'kW',     lambda v: v / 1e3),
    ('boil_margin_wall', 'Coolant wall below boiling', 'K',      lambda v: v),
    ('stress_margin_min', 'Minimum wall stress margin', '-',     lambda v: v),
]

# light mode palette, categorical slots in fixed order
SURFACE   = '#fcfcfb'
TEXT      = '#0b0b0b'
TEXT_2    = '#52514e'
GRID      = '#e4e3df'
BASELINE  = '#a3a29d'
SERIES    = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
SEQUENTIAL = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']


def saturation_margin(cfg, T, P):
    # distance of temperatures T [K] below the saturation temperature of the dominant coolant species at the local pressures P [Pa].
    # nan where it can not be evaluated: custom coolants without saturation data, or supercritical pressure (no boiling).
    # Saturation temperature from thermo (Bell 2016-)
    margin = np.full(len(T), np.nan)
    try:
        import thermo
        chemical = thermo.Chemical(cfg.coolant.IDs[int(np.argmax(cfg.coolant.ws))])
        for i in range(len(T)):
            if 0 < P[i] < chemical.Pc:
                margin[i] = chemical.Tsat(P[i]) - T[i]
    except Exception:
        pass
    return margin


def nan_to_none(v):
    # json has no nan
    return None if v is None or not np.isfinite(v) else float(v)


def case_stats(cfg, runtime, stress=None):
    # critical statistics of a finished case, all in SI units. stress: the wall stresses from Stress.chamber_stress, if available
    out = cfg.output
    x = cfg.geometry[:,0]
    r = cfg.geometry[:,1]
    it = int(np.argmin(r))                                          # throat index
    outlet = 0 if cfg.start_idx == -1 else -1                       # last section the coolant passes
    s = np.concatenate(([0], np.cumsum(np.hypot(np.diff(x), np.diff(r)))))  # arc length along the contour
    cg = cfg.cooling_geom
    wall_margin = saturation_margin(cfg, out.T_wall_c, out.P_c)                # coolant wetted wall below boiling [K]
    bulk_margin = saturation_margin(cfg, out.T_c, out.P_c)                     # coolant bulk below boiling [K]
    T_c_out, P_c_out = out.T_c[outlet], out.P_c[outlet]
    v_max, Re_min, T_wall_c_max = np.max(out.v_coolant), np.min(out.Re), np.max(out.T_wall_c)

    two_pass = getattr(cg, 'passes', 1) == 2
    if two_pass:
        # the coolant leaves through the return pass, at the end where it entered
        outlet = 0 if cfg.start_idx == 0 else -1
        T_c_out, P_c_out = out.T_c2[outlet], out.P_c2[outlet]
        wall_margin = np.fmin(wall_margin, saturation_margin(cfg, out.T_wall_c2, out.P_c2))
        bulk_margin = np.fmin(bulk_margin, saturation_margin(cfg, out.T_c2, out.P_c2))
        v_max, Re_min = max(v_max, np.max(out.v_coolant2)), min(Re_min, np.min(out.Re2))
        T_wall_c_max = max(T_wall_c_max, np.max(out.T_wall_c2))

    stats = {
        'case':          os.path.splitext(os.path.basename(cfg.config_path))[0],
        'status':        'ok',
        'config_path':   cfg.config_path,
        'runtime_s':     runtime,
        'x_throat':      float(x[it]),
        'T_wall_max':    float(np.max(out.T_wall_i)),
        'x_T_wall_max':  float(x[np.argmax(out.T_wall_i)]),
        'T_wall_median': float(np.median(out.T_wall_i)),
        'T_wall_throat': float(out.T_wall_i[it]),
        'T_coolant_out': float(T_c_out),
        'dT_coolant':    float(T_c_out - cfg.inlet_temp),
        'P_coolant_out': float(P_c_out),
        'dp_coolant':    float(cfg.inlet_pressure - P_c_out),
        'dp_injector':   float(P_c_out - cfg.Pc),
        'q_max':         float(np.max(out.q)),
        'x_q_max':       float(x[np.argmax(out.q)]),
        'Q_total':       float(trapezoid(out.q * 2 * np.pi * r, s)),
        'v_coolant_max': float(v_max),
        'Re_min':        float(Re_min),
        'T_wall_coolant_max': float(T_wall_c_max),
        'boil_margin_wall':   nan_to_none(np.nanmin(wall_margin)) if np.any(np.isfinite(wall_margin)) else None,
        'x_boil_margin_wall': float(x[np.nanargmin(wall_margin)]) if np.any(np.isfinite(wall_margin)) else None,
        'boil_margin_bulk':   nan_to_none(np.nanmin(bulk_margin)) if np.any(np.isfinite(bulk_margin)) else None,
        'w_channel_min': float(np.min(cg.psi_c_n * cg.r_i)),
        'w_rib_min':     float(np.min(cg.psi_w_n * cg.r_i)),
        'h_c_min':       float(np.min(cg.h_c)),
        'alpha_max_deg': float(np.degrees(np.max(cg.alpha))),
    }
    if stress is not None:
        i = int(np.argmin(stress['margin']))
        stats.update({'stress_margin_min': float(stress['margin'][i]), 'x_stress_margin_min': float(x[i]),
                      'sigma_vm_at_min_margin': float(stress['sigma_vm'][i]), 'sigma_t_at_min_margin': float(stress['sigma_t'][i]),
                      'sigma_l_at_min_margin': float(stress['sigma_l'][i]), 'yield_at_min_margin': float(stress['yield'][i]),
                      'sigma_vm_max': float(np.max(stress['sigma_vm'])),
                      'stress_longitudinal': getattr(cfg, 'stress_longitudinal', 'shell')})
    if two_pass:
        stats.update({'passes': 2, 'dp_turnaround': float(out.dp_turnaround), 'two_pass_iterations': int(out.two_pass_iterations),
                      'T_coolant_turnaround': float(out.T_c[-1 if cfg.start_idx == 0 else 0])})
    return stats


def write_case(stats, save_path):
    with open(os.path.join(save_path, 'summary.json'), 'w') as f:
        json.dump(stats, f, indent=2)


def read_case(save_path):
    with open(os.path.join(save_path, 'summary.json')) as f:
        return json.load(f)


def write_csv(cases, path):
    keys = []
    for c in cases:
        keys += [k for k in c if k not in keys]

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(cases)


def case_colors(n, ordered=False):
    # categorical hues for up to 8 cases, ordered sequential ramp beyond that or for ordered cases (e.g. sweeps)
    if n <= len(SERIES) and not ordered:
        return SERIES[:n]
    if n == 1:
        return [SEQUENTIAL[len(SEQUENTIAL) // 2]]
    return [SEQUENTIAL[round(i * (len(SEQUENTIAL) - 1) / (n - 1))] for i in range(n)]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.title.set_color(TEXT)


def plot_summary(cases, results_dir, path, ordered=False, heading='PyRocket case summary'):
    """[Summary]
    Summary figure comparing all successful cases. Top row: profiles along the chamber contour read from each
    case's sim_data.csv. Bottom rows: one bar chart per critical statistic, one bar per case.
    """
    cases = [c for c in cases if c.get('status') == 'ok']
    if not cases:
        return

    n = len(cases)
    colors = case_colors(n, ordered)
    names = [c['case'] for c in cases]

    bar_rows = int(np.ceil(len(METRICS) / 4))
    bar_height = 0.9 + 0.32 * n
    fig = plt.figure(figsize=(17, 3.6 + bar_rows * bar_height + 0.9), facecolor=SURFACE)
    grid = fig.add_gridspec(1 + bar_rows, 4, height_ratios=[3.6] + [bar_height] * bar_rows, hspace=0.55, wspace=0.28)

    # profiles along the contour, columns of sim_data.csv
    profiles = [
        (6, 'Inner wall temperature [' + Units.temperature_unit() + ']',   Units.temperature),
        (7, 'Coolant bulk temperature [' + Units.temperature_unit() + ']', Units.temperature),
        (8, 'Coolant pressure [bar]',       lambda v: v / 1e5),
        (5, 'Heat flux [MW/m²]',            lambda v: v / 1e6),
    ]
    for col, (data_col, title, conv) in enumerate(profiles):
        ax = fig.add_subplot(grid[0, col])
        style_axes(ax)
        for case, color in zip(cases, colors):
            data = np.genfromtxt(os.path.join(results_dir, case['case'], 'sim_data.csv'), delimiter=',', skip_header=1)
            ax.plot(data[:,1], conv(data[:,data_col]), color=color, linewidth=2, label=case['case'])
            # return pass of two-pass cooling, dashed in the colour of its case
            if data_col in (7, 8) and data.shape[1] > 14 and np.all(np.isfinite(data[:,data_col + 6])):
                ax.plot(data[:,1], conv(data[:,data_col + 6]), color=color, linewidth=1.5, linestyle='--')
            ax.axvline(case['x_throat'] * 1e3, color=BASELINE, linewidth=1, linestyle=':', zorder=0)
        ax.set_title(title, fontsize=10, loc='left')
        ax.set_xlabel('x coordinate [mm]  (dotted: throat' + (', dashed: return pass' if data_col in (7, 8) else '') + ')', fontsize=8, color=TEXT_2)

    # one bar chart per statistic, cases share the same order and color in every panel
    y = np.arange(n)[::-1]
    for i, (key, title, unit, conv) in enumerate(METRICS):
        ax = fig.add_subplot(grid[1 + i // 4, i % 4])
        style_axes(ax)
        ax.grid(axis='y', visible=False)
        ax.tick_params(axis='y', length=0)
        values = np.array([conv(c[key]) if c.get(key) is not None else np.nan for c in cases], dtype=float)
        if not np.any(np.isfinite(values)):
            ax.set_title(f'{title} [{unit}]', fontsize=10, loc='left')
            ax.text(0.5, 0.5, 'not available for these cases', transform=ax.transAxes, ha='center', va='center', fontsize=8, color=TEXT_2)
            ax.set_yticks([])
            continue
        ax.barh(y, values, height=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
        ax.axvline(0, color=BASELINE, linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(names if i % 4 == 0 else [], fontsize=8, color=TEXT)
        ax.set_title(f'{title} [{unit}]', fontsize=10, loc='left')

        # direct value labels at the bar ends, with room for them past the longest bar
        vmax, vmin = np.nanmax(values), np.nanmin(values)
        span = max(vmax, 0) - min(vmin, 0) or 1.0
        ax.set_xlim(min(vmin, 0) - 0.02 * span - (0.25 * span if vmin < 0 else 0),
                    max(vmax, 0) + 0.25 * span)
        for yi, v in zip(y, values):
            if not np.isfinite(v):
                continue
            ax.text(v + (0.02 if v >= 0 else -0.02) * span, yi, f'{v:.3g}', va='center',
                    ha='left' if v >= 0 else 'right', fontsize=8, color=TEXT_2)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.125, 0.995), ncol=min(n, 8), frameon=False, fontsize=9, labelcolor=TEXT)
    fig.suptitle(heading, x=0.125, ha='left', y=1.03, fontsize=13, color=TEXT)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
