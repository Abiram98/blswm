"""Command line entry for CQPP serving, training and evaluation."""
import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    dataset = commands.add_parser('dataset', help='Build query-set JSONL from real FIFO workload runs')
    dataset.add_argument('--workload', required=True, help='Workload directory or query CSV')
    dataset.add_argument('--runs', nargs='+', required=True, help='Completed FIFO run directories or run.json files')
    dataset.add_argument('--format', choices=['queryset', 'candidate'], default='queryset', help='Joint query-set JSONL or candidate-only CSV/TSV')
    dataset.add_argument('--split', choices=['train_validation', 'test'], required=True)
    dataset.add_argument('--sample-mode', choices=['all_query_start_events', 'all_decision_events', 'aligned_hard_prefix'], default='all_query_start_events')
    dataset.add_argument('--validation-fraction', type=float, default=.2)
    dataset.add_argument('--threshold', type=float, default=120.)
    dataset.add_argument('--seed', type=int, default=0)
    dataset.add_argument('--out', required=True)
    feature = commands.add_parser('featurizer', help='Fit graph features from knowledge-graph statistics')
    feature.add_argument('--predicate-stats', required=True)
    feature.add_argument('--entity-stats', required=True)
    feature.add_argument('--literal-stats')
    feature.add_argument('--bins', type=int, default=10)
    feature.add_argument('--scaling', choices=['binner','none'], default='binner')
    feature.add_argument('--out', required=True)
    for command in ('fit', 'tune'):
        p = commands.add_parser(command, help='Train or tune a query performance model')
        p.add_argument('--model-type', choices=['planrgcn','dwns','qsps','qsps-regression','p-iconq'], required=True)
        p.add_argument('--data', required=True)
        p.add_argument('--validation', required=True)
        p.add_argument('--featurizer', required=True, help='Trusted fitted featurizer.pickle')
        p.add_argument('--feature-bins', type=int, help='Refit the supplied feature statistics with this many bins')
        p.add_argument('--feature-scaling', choices=['binner','none'], help='Refit feature scaling')
        p.add_argument('--init-checkpoint', help='Trusted compatible best_model.pt to fine-tune')
        p.add_argument('--encoder-checkpoint', help='Trusted PlanRGCN checkpoint for query encoder initialization')
        p.add_argument('--freeze-query-encoder', action='store_true')
        p.add_argument('--out', required=True)
        p.add_argument('--thresholds', type=float, nargs='+', default=[120.])
        p.add_argument('--hidden-dim1', type=int, default=512)
        p.add_argument('--hidden-dim2', type=int, default=128)
        p.add_argument('--lstm-hidden-dim', type=int, default=128)
        p.add_argument('--lstm-num-layers', type=int, default=1)
        p.add_argument('--lstm-dropout', type=float)
        p.add_argument('--dropout', type=float, default=.1)
        p.add_argument('--epochs', type=int, default=10)
        p.add_argument('--batch-size', type=int, default=1)
        p.add_argument('--optimizer', choices=['adam','adamw'], default='adam')
        p.add_argument('--label-smoothing', type=float, default=0.)
        p.add_argument('--selection-metric', choices=['loss','macro_f1'], default='loss')
        p.add_argument('--lr', type=float, default=.001)
        p.add_argument('--weight-decay', type=float, default=0.)
        p.add_argument('--grad-clip', type=float, default=1.)
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--device', default='cpu')
        p.add_argument('--threads', type=int, default=1)
        p.add_argument('--jar')
        if command == 'tune':
            p.add_argument('--search', required=True, help='JSON object mapping hyperparameters to candidate lists')
            p.add_argument('--samples', type=int, default=4)
            p.add_argument('--cpus', type=int, default=2)
            p.add_argument('--gpus', type=float, default=0.)
            p.add_argument('--cpus-per-trial', type=float, default=1.)
            p.add_argument('--gpus-per-trial', type=float, default=0.)
            p.add_argument('--max-concurrent-trials', type=int, default=1)
    serve = commands.add_parser('serve')
    serve.add_argument('--model-dir', required=True)
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, default=9010)
    for name in ('train', 'evaluate'):
        p = commands.add_parser(name)
        p.add_argument('--model-dir', required=True, help='Trusted checkpoint to initialize/load')
        p.add_argument('--data', required=True, help='JSONL states with inline query_text and labels or observed runtimes')
        p.add_argument('--out', required=True)
        if name == 'train':
            p.add_argument('--validation', required=True)
            p.add_argument('--epochs', type=int, default=10)
            p.add_argument('--lr', type=float, default=.001)
            p.add_argument('--seed', type=int, default=42)
    for p in (serve, *[commands.choices[n] for n in ('train', 'evaluate')]):
        p.add_argument('--device', default='cpu')
        p.add_argument('--jar', help='Query graph extractor JAR, defaults to src/qpp/java/qpe/target/qpe-1.0-SNAPSHOT.jar')
        p.add_argument('--threads', type=int, default=4, help='Torch CPU threads')
    args = parser.parse_args(argv)
    try:
        if args.command == 'dataset':
            from .dataset import build_dataset
            result = build_dataset(args.workload, args.runs, args.out, split=args.split,
                                   validation_fraction=args.validation_fraction, seed=args.seed,
                                   threshold=args.threshold, sample_mode=args.sample_mode, dataset_format=args.format)
            print(json.dumps(result, indent=2))
            return 0
        if args.command == 'featurizer':
            from .model_training import make_featurizer
            result=make_featurizer(args.predicate_stats,args.entity_stats,args.literal_stats,args.out,
                                   bins=args.bins,scaling=args.scaling)
            print(json.dumps(result,indent=2))
            return 0
        if args.command in ('fit', 'tune'):
            import math
            from .model_training import fit, tune_models
            config = vars(args).copy()
            if (any(not math.isfinite(t) or t <= 0 for t in args.thresholds)
                    or args.thresholds != sorted(set(args.thresholds))
                    or min(args.hidden_dim1,args.hidden_dim2,args.lstm_hidden_dim,args.lstm_num_layers,args.epochs,args.threads,args.batch_size) < 1
                    or any(not math.isfinite(x) for x in (args.dropout,args.label_smoothing,args.lr,args.weight_decay,args.grad_clip))
                    or args.seed < 0 or not 0 <= args.dropout < 1 or not 0 <= args.label_smoothing < 1
                    or args.lr <= 0 or args.weight_decay < 0 or args.grad_clip <= 0):
                raise ValueError('Invalid model dimensions, thresholds or training settings')
            result = fit(config, args.out) if args.command == 'fit' else tune_models(config, args.out)
            print(json.dumps(result, indent=2))
            return 0
        import torch
        if args.threads < 1:
            raise ValueError('threads must be positive')
        torch.set_num_threads(args.threads)
        from .model import load_predictor
        model = load_predictor(args.model_dir, args.device, args.jar)
        if args.command == 'serve':
            from .server import server_for
            with server_for(model, args.host, args.port) as server:
                print(json.dumps(dict(model.info(), host=args.host, port=server.server_address[1])), flush=True)
                server.serve_forever()
        else:
            from .model import TrainedPredictor
            if isinstance(model, TrainedPredictor):
                if args.command == 'train':
                    raise ValueError('Use qpp fit to train these model types')
                from .model_training import evaluate_model
                from .training import write_report
                report = evaluate_model(model, args.data)
                write_report(args.out, report)
                print(json.dumps(report))
                return 0
            from .training import train, evaluate, read_states, write_report
            if args.command == 'train':
                report = train(model, args)
            else:
                report = evaluate(model, read_states(args.data, len(model.thresholds)+1, model.task))
                write_report(args.out, report)
            print(json.dumps(report), flush=True)
        return 0
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'qpp: {exc}\n')
