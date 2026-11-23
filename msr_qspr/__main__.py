"""Command-line interface: msr-qspr <command> (equivalently python -m msr_qspr <command>).

Every command prints one JSON summary on standard output; progress messages go to
standard error. Outputs are written under --artifacts (default ./artifacts), one
directory per stage; restartable worker state is kept under --workspace
(default <artifacts>/workspace).
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from msr_qspr import artifacts, config

STAGES = tuple(artifacts.STAGE_DIRECTORIES)


def _parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--artifacts', type=Path, metavar='DIR',
                        help='Stage output directory (default: ./artifacts)')
    common.add_argument('--workspace', type=Path, metavar='DIR',
                        help='Worker outputs, checkpoints and caches (default: <artifacts>/workspace)')
    parser = argparse.ArgumentParser(prog='msr-qspr', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True, metavar='command')

    def add(name, help_text, description=None):
        return commands.add_parser(name, parents=[common], help=help_text, description=description or help_text)

    add('verify', 'Check the workbook, Abraham descriptor files and grouped partitions')
    add('characterize', 'Describe all accepted observations: tables, figures and report')
    add('plan', 'List the candidate x outer-fold work units and the fitting budget, without reading responses')

    compare = add('compare', 'Run grouped outer validation for candidate x outer-fold units',
                  'Run every unit, or select one scenario, repetition, outer fold and candidate. '
                  'Completed units are verified and skipped, so interrupted runs resume.')
    compare.add_argument('--scenario', choices=list(config.VALIDATION_SETTINGS))
    compare.add_argument('--repetition', type=int, choices=range(1, max(
        len(v['seeds']) for v in config.VALIDATION_SETTINGS.values()) + 1))
    compare.add_argument('--outer-fold', type=int, choices=range(1, config.GROUPED_OUTER_FOLDS + 1))
    compare.add_argument('--candidate', choices=[e.name for e in config.study_experiments()])

    progress = add('progress', 'Monitor the comparison: interactive dashboard in a terminal, JSON snapshot otherwise')
    progress.add_argument('--refresh-seconds', type=float, default=2.0, help='Dashboard refresh interval (default: 2)')

    add('aggregate', 'Verify every completed unit and publish the comparison tables')
    add('select', 'Apply the selection rule to the published comparison and record the selected model')
    add('fit', 'Refit the selected procedure on all development observations')

    interpret = add('interpret', 'Attribute the fitted model predictions on the development observations')
    interpret.add_argument('--bundle', type=Path, metavar='DIR', help='Fitted bundle (default: the one in final_model/)')

    add('reliability', 'Assess applicability and local error of the selected model on its outer folds')

    randomize = add('randomize', 'Rerun the selected procedure on shuffled responses (target randomization)',
                    'Run every unit, a single unit (--permutation, --scenario and --repetition), '
                    'or publish the verified units with --aggregate.')
    randomize.add_argument('--permutations', type=int, required=True, help='Number of permutations')
    randomize.add_argument('--permutation', type=int, help='Zero-based permutation number of one unit')
    randomize.add_argument('--scenario', choices=list(config.VALIDATION_SETTINGS))
    randomize.add_argument('--repetition', type=int)
    randomize.add_argument('--aggregate', action='store_true', help='Publish metrics from all verified units')

    holdout = add('holdout', 'Score the fitted bundle once on the reserved holdout observations')
    holdout.add_argument('--bundle', type=Path, metavar='DIR', help='Fitted bundle (default: the one in final_model/)')
    holdout.add_argument('--confirm-holdout', action='store_true', required=True,
                         help='Confirm access to the reserved responses')

    report = add('report', 'Build tables and figures from verified stage outputs, without fitting')
    report.add_argument('--stage', action='append', choices=STAGES, help='Build only this stage (repeatable)')

    screen = add('screen', 'Rank candidate surfactants for one contaminant with the fitted bundle')
    screen.add_argument('--contaminant-smiles', required=True)
    screen.add_argument('--water-solubility', type=float, required=True, help='Contaminant water solubility (g/L)')
    screen.add_argument('--surfactants', type=Path, required=True,
                        help="CSV with 'surfactant', 'smiles' and 'cmc' (mol/L); optional 'surfactant_type' "
                             "and 'molecular_weight' (g/mol)")
    screen.add_argument('--contaminant-type', default='Unknown')
    screen.add_argument('--contaminant-name', help='Curated name; required for stereoisomers a structure cannot identify')
    screen.add_argument('--contaminant-cas', help='CAS number; alternative stereoisomer confirmation')
    screen.add_argument('--bundle', type=Path, metavar='DIR', help='Fitted bundle (default: the one in final_model/)')
    output = screen.add_mutually_exclusive_group()
    output.add_argument('--output', type=Path, help='Write the ranking to this CSV file')
    output.add_argument('--save', metavar='NAME', help='Save the ranking as a case in screening/cases/ for report')

    add('app', 'Open the interactive prediction application')
    return parser


def _emit(value):
    print(json.dumps(value, indent=2, default=str))


def _log(message):
    print(message, file=sys.stderr, flush=True)


def _configure(args):
    if args.artifacts is not None:
        config.ARTIFACTS_DIR = args.artifacts.resolve()
        os.environ['MSR_ARTIFACTS_DIR'] = str(config.ARTIFACTS_DIR)
    if args.workspace is not None:
        config.WORKSPACE_DIR = args.workspace.resolve()
        os.environ['MSR_WORKSPACE_DIR'] = str(config.WORKSPACE_DIR)


def _validate(parser, args):
    if args.command == 'compare':
        if args.repetition and any(args.repetition > len(config.VALIDATION_SETTINGS[s]['seeds'])
                                   for s in ([args.scenario] if args.scenario else config.VALIDATION_SETTINGS)):
            parser.error('unseen_study has one repetition; select a chemical-axis scenario for repetitions 2 and 3.')
        if args.candidate and not (args.scenario and args.repetition and args.outer_fold):
            parser.error('--candidate requires --scenario, --repetition and --outer-fold.')
    if args.command == 'randomize':
        if args.permutations < 1:
            parser.error('--permutations must be positive.')
        unit = [args.permutation, args.scenario, args.repetition]
        if any(v is not None for v in unit) and (None in unit or args.aggregate):
            parser.error('A unit requires --permutation, --scenario and --repetition, without --aggregate.')
    if args.command == 'progress' and not (0 < args.refresh_seconds < float('inf')):
        parser.error('--refresh-seconds must be positive.')
    if args.command == 'screen' and args.save is not None and not re.fullmatch(r'[A-Za-z0-9_-]+', args.save):
        parser.error('--save accepts letters, numbers, underscores and hyphens.')


def _study():
    """Run metadata of the current inputs and code, and the comparison identity."""
    from msr_qspr import study
    value = study.run_metadata()
    return value, study.run_fingerprint(value, study.COMPARISON)


def _selected_study():
    """The recorded selection, checked against its checksum."""
    from msr_qspr import study
    path = artifacts.artifact_directory('comparison') / 'selected_model.json'
    if not path.is_file():
        raise artifacts.MissingArtifact(f'No selection record in {path.parent}; run aggregate and select first.')
    value = json.loads(path.read_text())
    if not artifacts.cache_matches(path, study.run_fingerprint(value, 'final_model')):
        raise artifacts.InvalidArtifact(f'{path} differs from its recorded checksum.')
    return value


def _bundle(args):
    if getattr(args, 'bundle', None) is not None:
        return args.bundle.resolve()
    directory = artifacts.bundle_directory()
    if not (directory / 'prediction_ad_bundle.json').is_file():
        raise artifacts.MissingArtifact(f'No fitted bundle in {artifacts.artifact_directory("final_model")}; run fit.')
    return directory


def _run(args):
    command = args.command
    if command == 'plan':
        from msr_qspr.study import work_units
        from msr_qspr.framework import computational_budget
        return {'command': command, 'budget': computational_budget(), **work_units()}
    if command == 'verify':
        from msr_qspr.data import verify
        return {'command': command, **verify()}
    if command == 'characterize':
        from msr_qspr.characterization import run
        return {'command': command, **run(artifacts.artifact_directory('characterization'))}
    if command == 'report':
        from msr_qspr.reporting import build_report
        result = build_report(config.ARTIFACTS_DIR, stages=args.stage, log=_log)
        result['status'] = 'failed' if result['invalid'] or result['error'] else 'complete'
        return {'command': command, **result}
    if command == 'screen':
        return _screen(args)
    if command == 'app':
        from streamlit.web import cli
        sys.argv = ['streamlit', 'run', str(Path(__file__).with_name('application.py'))]
        cli.main()
        return None
    if command == 'progress':
        value, identity = _study()
        run_root = artifacts.artifact_directory('runs', study_id=identity)
        if sys.stdin.isatty() and sys.stdout.isatty():
            from msr_qspr.dashboard import run
            run(run_root, args.refresh_seconds)
            return None
        from msr_qspr.status import inspect_status
        return {'command': command, 'run_root': str(run_root), **inspect_status(run_root)}

    from msr_qspr import reporting, workflow
    if command in {'compare', 'aggregate', 'select'}:
        value, identity = _study()
        comparison = artifacts.artifact_directory('comparison')
        if command == 'compare':
            for scenario in ([args.scenario] if args.scenario else config.VALIDATION_SETTINGS):
                workflow.run_primary_experiment(scenario, 'temp_off', experiments=config.study_experiments(),
                    study=value, fold_number=args.outer_fold, repetition=args.repetition, candidate=args.candidate)
            return {'command': command, 'status': 'complete', 'comparison_identity': identity,
                    'run_root': str(artifacts.artifact_directory('runs', study_id=identity)),
                    'selection': {k: getattr(args, k) for k in ('scenario', 'repetition', 'outer_fold', 'candidate')}}
        if command == 'aggregate':
            return {'command': command, 'status': 'complete',
                    **reporting.aggregate_primary_results(output_dir=comparison, study=value)}
        selected = reporting.select_final_specification(comparison, value)
        record = selected['stages']['final_model']['selection_record']
        return {'command': command, 'status': 'complete', 'selected_configuration': record['configuration'],
                'selection_record': record}

    value = _selected_study()
    comparison = artifacts.artifact_directory('comparison')
    if command == 'holdout':
        from msr_qspr.final import evaluate_holdout
        metrics = evaluate_holdout(study=value, bundle_root=_bundle(args), confirmed=args.confirm_holdout)
        return {'command': command, 'status': 'complete', 'output': str(artifacts.artifact_directory('holdout')),
                'metrics': metrics}
    if command == 'randomize':
        value['stages']['y_randomization']['n_permutations'] = args.permutations
    prepared = workflow._load_dev_full('temp_off')
    common = dict(X=prepared['X'], y_log=prepared['y_log'], study=value,
                  selection=value['stages']['final_model']['selection_record'])
    if command == 'fit':
        from msr_qspr.final import fit_selected_final_model
        bundle = fit_selected_final_model(**common, development=prepared['df_full'], comparison_table=comparison)
        return {'command': command, 'status': 'complete', 'configuration': bundle['experiment']['name'],
                'bundle': str(artifacts.bundle_directory()), 'model_sha256': bundle['model_sha256']}
    from msr_qspr import diagnostics
    if command == 'interpret':
        from msr_qspr.preprocessing import project_experiment_predictors
        from msr_qspr.study import experiment_specification
        experiment = experiment_specification(value['stages']['final_model']['tuning_policy']['experiment'])
        common['X'] = project_experiment_predictors(common['X'], experiment)
        result = diagnostics.run_shap(**common, comparison_directory=comparison, bundle_directory=_bundle(args),
                                      output_directory=artifacts.artifact_directory('interpretation'))
        return {'command': command, 'status': 'complete', 'shap': result['shap']['status'],
                'output': str(artifacts.artifact_directory('interpretation'))}
    if command == 'reliability':
        result = diagnostics.run_reliability(**common, comparison_directory=comparison,
                                             development=prepared['df_full'])
        status = result['observation_status']
        return {'command': command, 'status': 'complete', 'output': str(artifacts.artifact_directory('reliability')),
                'rows': len(status), 'share_inside': float(status.ad_status.eq('inside').mean())}
    plan = diagnostics.prepare_y_randomization(**common, comparison_directory=comparison,
                                               development=prepared['df_full'])
    if args.permutation is not None:
        record = diagnostics.run_y_randomization_unit(plan, args.permutation, args.scenario, args.repetition)
        return {'command': command, 'unit': record, 'status': record['status']}
    if not args.aggregate:
        for unit in diagnostics.randomization_units(plan):
            diagnostics.run_y_randomization_unit(plan, *unit)
    permutations = diagnostics.aggregate_y_randomization(plan)['permutations']
    failed = int(permutations.status.ne('complete').sum())
    return {'command': command, 'status': 'failed' if failed else 'complete',
            'output': str(artifacts.artifact_directory('y_randomization')), 'permutations': args.permutations,
            'failed_permutation_scenarios': failed,
            'mean_shuffled_RMSE': permutations.groupby('Scenario').mean_repetition_RMSE.mean().to_dict()}


def _screen(args):
    import pandas as pd
    from msr_qspr.inference import MSRPredictor
    from msr_qspr.screening import COLUMN_DEFINITIONS, SCREENING_NOTICE, rank_surfactants
    if args.bundle is not None:
        os.environ['MSR_BUNDLE_DIR'] = str(args.bundle.resolve())
    surfactants = pd.read_csv(args.surfactants)
    predictor = MSRPredictor()
    table = rank_surfactants(predictor, contaminant_smiles=args.contaminant_smiles,
        water_solubility=args.water_solubility, surfactants=surfactants, contaminant_type=args.contaminant_type,
        contaminant_name=args.contaminant_name, contaminant_cas=args.contaminant_cas)
    result = {'command': 'screen', 'status': 'complete', 'notice': SCREENING_NOTICE}
    if args.save is not None:
        folder = artifacts.artifact_directory('screening') / 'cases'
        folder.mkdir(parents=True, exist_ok=True)
        csv_path, metadata_path = folder / f'{args.save}.csv', folder / f'{args.save}.json'
        if csv_path.exists() or metadata_path.exists():
            raise ValueError(f'Screening case {args.save} already exists; choose another name.')
        with csv_path.open('x') as handle:
            table.to_csv(handle, index=False)
        try:
            with metadata_path.open('x') as handle:
                json.dump({'model_sha256': predictor.bundle['model_sha256'],
                           'ranking_sha256': artifacts.file_sha256(csv_path),
                           'contaminant_name': args.contaminant_name, 'contaminant_smiles': args.contaminant_smiles,
                           'contaminant_cas': args.contaminant_cas, 'contaminant_type': args.contaminant_type,
                           'water_solubility_g_per_l': args.water_solubility,
                           'surfactants': json.loads(surfactants.to_json(orient='records')),
                           'columns': COLUMN_DEFINITIONS}, handle, indent=2)
        except BaseException:
            csv_path.unlink(missing_ok=True)
            raise
        result['output'] = str(csv_path)
    elif args.output is not None:
        table.to_csv(args.output, index=False)
        result['output'] = str(args.output)
    else:
        result['ranking'] = table.to_dict('records')
    return result


def main(argv=None):
    """Run one command; return the process exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    _validate(parser, args)
    _configure(args)
    try:
        result = _run(args)
    except (artifacts.MissingArtifact, artifacts.InvalidArtifact, FileNotFoundError, ValueError) as error:
        _log(f'error: {error}')
        _emit({'command': args.command, 'status': 'failed', 'error': str(error)})
        return 1
    if result is None:
        return 0
    _emit(result)
    return 1 if result.get('status') == 'failed' else 0


def cli():
    sys.exit(main())


if __name__ == '__main__':
    cli()
