"""Command line entry for CQPP serving, training and evaluation."""
import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
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
        p.add_argument('--jar', help='Query graph extractor JAR, defaults to java/qpe/target/qpe-1.0-SNAPSHOT.jar')
        p.add_argument('--threads', type=int, default=4, help='Torch CPU threads')
    args = parser.parse_args(argv)
    try:
        import torch
        if args.threads < 1:
            raise ValueError('threads must be positive')
        torch.set_num_threads(args.threads)
        from .model import Predictor
        model = Predictor(args.model_dir, args.device, args.jar)
        if args.command == 'serve':
            from .server import server_for
            with server_for(model, args.host, args.port) as server:
                print(json.dumps(dict(model.info(), host=args.host, port=server.server_address[1])), flush=True)
                server.serve_forever()
        else:
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
