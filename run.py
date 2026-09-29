#####################################################################
#                           PyRocket								#
# 2D Regenertive Cooling Simulation for Bipropellant Rocket Engines #
#                                                                   #
# Creator:  Joanthan Neeser                                         #
# Date:     15.12.2022                                              #
# Version:  2.3  													#
# License:	GNU GENERAL PUBLIC LICENSE V3							#
#####################################################################

# Usage:
#   python3 run.py                          run every config file in configs/
#   python3 run.py config.py                run a single config
#   python3 run.py a.py b.py --jobs 2       run several configs, two at a time
#   python3 run.py --summary-only           rebuild the summary from existing results
# Each case is written to <out>/<config name>/, the comparison to <out>/summary.png and <out>/summary.csv

import argparse
import glob
import multiprocessing
import os
import sys
import time
import traceback

import matplotlib
matplotlib.use('Agg')

import Summary
import Units


def expand_configs(paths):
    # folders are expanded to all python files inside them
    configs = []
    for path in paths:
        if os.path.isdir(path):
            configs += sorted(p for p in glob.glob(os.path.join(path, '*.py')) if not os.path.basename(p).startswith('_'))
        elif os.path.isfile(path):
            configs.append(path)
        else:
            sys.exit('Config not found: ' + path)

    if not configs:
        sys.exit('No config files found in: ' + ', '.join(paths))

    names = [os.path.splitext(os.path.basename(c))[0] for c in configs]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        sys.exit('Config file names must be unique, they name the output folders. Duplicates: ' + ', '.join(duplicates))
    return configs


def run_case(config_path, save_path, quiet, overrides=None, extra=None):
    # runs a single config in its own process, so cases can not affect each other
    # overrides replace config inputs (see CaseSetup.read_inputs), extra is stored with the case summary
    os.makedirs(save_path, exist_ok=True)
    if quiet:
        # send all output of parallel cases to their log files, including output of gmsh
        log = open(os.path.join(save_path, 'log.txt'), 'w')
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        sys.stdout.reconfigure(line_buffering=True)          # progress shows up in the log while the case runs

    start = time.time()
    try:
        from CaseSetup import load_config
        import RegenCooling as rc
        import PlottingFunctions as pl
        import Stress

        config = load_config(config_path, save_path, overrides)

        # give terminal message about simulation settings
        config.settings.output_msg()
        config.output.output_msg()

        # heat transfer sim
        sim = rc.HeatTransfer(
            cea=config.cea,
            gas=config.gas,
            geometry=config.chamber.geometry,
            material=config.material,
            coolant=config.coolant,
            cooling_geometry=config.cooling_geom,
            m_dot=config.m_dot,
            m_dot_coolant=config.m_dot_coolant,
            T_amb = config.ambient_temp,
            output=config.output,
            settings2D=config.settings,
            model=config.hot_gas_method,
            cool_model=config.cooling_method,
            eta_c_star=config.eta_c_star,
            film = config.film,
            bartz_knockdown = getattr(config, 'bartz_knockdown', 1.0),
            turnaround_loss = getattr(config, 'turnaround_loss', 1.5),
        )

        sim.run()

        # plotting
        plot = pl.Plotting1D(save_path=config.save_path, geometry=config.geometry, save=True, show=False)
        plot.temperature_plot()
        plot.pressure_plot()
        plot.heat_transfer_coeff_plot()
        plot.heat_flux_plot()
        plot.reynolds_plot()

        # hot wall stress margin, for materials with temperature dependent mechanical properties
        stress = None
        longitudinal = getattr(config, 'stress_longitudinal', 'shell')
        if Stress.has_stress_data(config.material):
            stress = Stress.chamber_stress(config.material, config.output, config.cooling_geom, config.gas.p_s, longitudinal)
            Stress.write_csv(stress, config.geometry, os.path.join(config.save_path, 'stress.csv'))
            plot.stress_plot(stress, longitudinal)
        else:
            print('No stress estimate: %s has no temperature dependent mechanical properties in MaterialLib' % config.material.name)

        stats = Summary.case_stats(config, time.time() - start, stress)

    except Exception as e:
        traceback.print_exc()
        stats = {
            'case':        os.path.splitext(os.path.basename(config_path))[0],
            'status':      'failed',
            'config_path': os.path.abspath(config_path),
            'runtime_s':   time.time() - start,
            'error':       repr(e),
        }

    # cases are named after their output folder, which is the config name for normal runs
    stats['case'] = os.path.basename(os.path.normpath(save_path))
    stats.update(extra or {})

    Summary.write_case(stats, save_path)
    sys.stdout.flush()
    return stats


def print_table(cases):
    print('')
    unit = Units.temperature_unit().replace('°', '')
    print('%-24s %-7s %10s %10s %10s %10s %10s' % ('case', 'status', 'Tw max[%s]' % unit, 'Tw med[%s]' % unit, 'dp [bar]', 'q [MW/m2]', 'time [s]'))
    for c in cases:
        if c['status'] == 'ok':
            print('%-24s %-7s %10.1f %10.1f %10.2f %10.2f %10.0f' % (c['case'], c['status'], Units.temperature(c['T_wall_max']), Units.temperature(c['T_wall_median']),
                                                                     c['dp_coolant'] / 1e5, c['q_max'] / 1e6, c['runtime_s']))
        else:
            print('%-24s %-7s %s' % (c['case'], c['status'], c.get('error', '')))
    print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run one or more PyRocket configs and summarise the results.')
    parser.add_argument('configs', nargs='*', default=['configs'], help='config files or folders of config files (default: every config in configs/)')
    parser.add_argument('--out', default='results', help='output folder, each case is written to <out>/<config name>/ (default: results)')
    parser.add_argument('--jobs', type=int, default=1, help='number of cases to run in parallel, output then goes to <out>/<config name>/log.txt (default: 1)')
    parser.add_argument('--summary-only', action='store_true', help='skip the simulations and rebuild the summary from existing results')
    parser.add_argument('--section-images', action='store_true', help='keep the per-section temperature images of the configs in batch runs (off for more than one case, they take about 70 ms per section)')
    parser.add_argument('--logs', action='store_true', help='write the output of every case to <out>/<config name>/log.txt, also when running one case at a time')
    args = parser.parse_args()

    configs = expand_configs(args.configs)
    save_paths = [os.path.join(args.out, os.path.splitext(os.path.basename(c))[0]) for c in configs]

    if args.summary_only:
        cases = []
        for c, save_path in zip(configs, save_paths):
            if os.path.isfile(os.path.join(save_path, 'summary.json')):
                cases.append(Summary.read_case(save_path))
            else:
                print('No results found for ' + c + ', skipping')
    else:
        if args.jobs > 1:
            # one thread per case, avoids oversubscribing the cpu with parallel cases
            for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
                os.environ.setdefault(var, '1')
            print('Running %d cases, %d in parallel. Progress is written to <case>/log.txt' % (len(configs), args.jobs))

        # fresh process for every case
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=max(1, min(args.jobs, len(configs))), maxtasksperchild=1) as pool:
            overrides = {'save_fig': False} if len(configs) > 1 and not args.section_images else None
            cases = pool.starmap(run_case, [(c, s, args.jobs > 1 or args.logs, overrides) for c, s in zip(configs, save_paths)])

    if not cases:
        sys.exit('No cases to summarise')

    os.makedirs(args.out, exist_ok=True)
    Summary.write_csv(cases, os.path.join(args.out, 'summary.csv'))
    Summary.plot_summary(cases, args.out, os.path.join(args.out, 'summary.png'))
    print_table(cases)
    print('Summary written to ' + os.path.join(args.out, 'summary.png') + ' and ' + os.path.join(args.out, 'summary.csv'))

    if any(c['status'] != 'ok' for c in cases):
        sys.exit(1)
