"""All-position CQPP supervision from saved concurrency states.

Rows contain S, class labels or observed runtimes, and optional set_size/sample_id. Targets remain paired with
input positions when pos reorders a state. Query text must be inline; paths in
historical dataset metadata are never followed.
"""
import json
import math
import random
import shutil
from pathlib import Path
from .protocol import normalize
from .artifacts import output_path, new_directory


def read_states(path, num_classes, task='classification'):
    states = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            queries = normalize(raw.get('S'))
            if task == 'regression':
                # Existing datasets retain interval labels in y. Never treat
                # those class indices as runtime seconds.
                labels = raw.get('y_runtime')
                if labels is None:
                    labels = [query.get('observed_runtime') for query in raw['S']]
            else:
                labels = raw.get('y')
            if not isinstance(labels, list) or len(labels) != len(queries) or raw.get('set_size', len(queries)) != len(queries):
                raise ValueError(f'{path}:{line_number}: S, y and set_size must match')
            if task == 'regression':
                if any(isinstance(y, bool) or not isinstance(y, (int, float)) or not math.isfinite(y) or y < 0 for y in labels):
                    raise ValueError(f'{path}:{line_number}: regression requires nonnegative observed_runtime or y_runtime in seconds')
            else:
                if any(isinstance(y, bool) or not isinstance(y, int) or not 0 <= y < num_classes for y in labels):
                    raise ValueError(f'{path}:{line_number}: invalid class label')
                for q in queries:
                    if 'label' in q and q['label'] != labels[q['input_index']]:
                        raise ValueError(f'{path}:{line_number}: conflicting per-query label')
            states.append(dict(queries=queries, labels=[labels[q['input_index']] for q in queries],
                               sample_id=str(raw.get('sample_id', line_number)),
                               split_group_id=raw.get('split_group_id')))
    if not states:
        raise ValueError('Dataset is empty')
    return states


def evaluate_regression(predictor, states):
    import torch
    predictions, squared_error, absolute_error, total = [], 0., 0., 0
    predictor.model.eval()
    with torch.inference_mode():
        for row in states:
            values = predictor.model(*predictor.tensors(row['queries']))[0, :, 0]
            if not torch.isfinite(values).all():
                raise ValueError('Nonfinite regression predictions')
            labels = torch.tensor(row['labels'], dtype=torch.float32, device=predictor.device)
            error = values-labels
            squared_error += float(error.square().sum())
            absolute_error += float(error.abs().sum())
            total += len(labels)
            predictions.append(dict(sample_id=row['sample_id'], query_ids=[q['query_id'] for q in row['queries']],
                                    input_indices=[q['input_index'] for q in row['queries']],
                                    observed_runtime_seconds=row['labels'], predicted_runtime_seconds=values.cpu().tolist()))
    return dict(prediction_task='regression', prediction_unit='seconds', runtime_semantics='total_runtime',
                states=len(states), queries=total, rmse=math.sqrt(squared_error/total),
                mae=absolute_error/total, predictions=predictions)


def evaluate(predictor, states):
    import torch
    if predictor.task == 'regression':
        return evaluate_regression(predictor, states)
    count = len(predictor.thresholds) + 1
    confusion = [[0] * count for _ in range(count)]
    predictions, total_loss, total = [], 0., 0
    predictor.model.eval()
    with torch.inference_mode():
        for row in states:
            logits = predictor.model(*predictor.tensors(row['queries']))[0]
            labels = torch.tensor(row['labels'], dtype=torch.long, device=predictor.device)
            total_loss += float(torch.nn.functional.cross_entropy(logits, labels, reduction='sum'))
            classes = logits.argmax(-1).cpu().tolist()
            for truth, predicted in zip(row['labels'], classes):
                confusion[truth][predicted] += 1
                total += 1
            predictions.append(dict(sample_id=row['sample_id'], query_ids=[q['query_id'] for q in row['queries']],
                                    input_indices=[q['input_index'] for q in row['queries']],
                                    labels=row['labels'], predicted_classes=classes))
    f1 = []
    for c in range(count):
        denominator = sum(confusion[c]) + sum(row[c] for row in confusion)
        f1.append(2 * confusion[c][c] / denominator if denominator else 0.)
    return dict(states=len(states), queries=total, cross_entropy=total_loss / total,
                accuracy=sum(confusion[c][c] for c in range(count)) / total,
                macro_f1=sum(f1) / count, confusion_matrix=confusion, predictions=predictions)


def write_report(path, report):
    target = output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')


def train(predictor, args):
    import torch
    if args.epochs < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('epochs and lr must be positive')
    torch.manual_seed(args.seed)
    randomizer = random.Random(args.seed)
    states = read_states(args.data, len(predictor.thresholds)+1, predictor.task)
    validation = read_states(args.validation, len(predictor.thresholds)+1, predictor.task)
    regression = predictor.task == 'regression'
    selection_metric = 'rmse' if regression else 'cross_entropy'
    # Never choose a checkpoint using a group that also appears in training.
    groups = {r['split_group_id'] for r in states if r['split_group_id'] is not None}
    validation_groups = {r['split_group_id'] for r in validation if r['split_group_id'] is not None}
    if groups & validation_groups:
        raise ValueError('Training and validation split groups overlap')
    if Path(args.data).resolve() == Path(args.validation).resolve():
        raise ValueError('Training and validation must use separate files')
    destination = output_path(args.out)
    if destination == predictor.directory or destination.is_relative_to(predictor.directory):
        raise ValueError('Training output must be outside the source model directory')
    destination = new_directory(destination)
    optimizer = torch.optim.Adam(predictor.model.parameters(), lr=args.lr)
    history, best = [], math.inf
    for epoch in range(1, args.epochs+1):
        predictor.model.train()
        indices = list(range(len(states)))
        randomizer.shuffle(indices)
        loss_sum, total = 0., 0
        for index in indices:
            row = states[index]
            logits = predictor.model(*predictor.tensors(row['queries']))[0]
            labels = torch.tensor(row['labels'], dtype=torch.float32 if regression else torch.long, device=predictor.device)
            if regression:
                mse = torch.nn.functional.mse_loss(logits[:, 0], labels)
                loss = torch.sqrt(mse+1e-8)
            else:
                loss = torch.nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), 1.)
            optimizer.step()
            loss_sum += float(mse.detach() if regression else loss.detach()) * len(labels)
            total += len(labels)
        metrics = evaluate(predictor, validation)
        if regression:
            history.append(dict(epoch=epoch, training_rmse=math.sqrt(loss_sum/total), validation_rmse=metrics['rmse'], validation_mae=metrics['mae']))
        else:
            history.append(dict(epoch=epoch, training_cross_entropy=loss_sum/total,
                                validation_cross_entropy=metrics['cross_entropy'], validation_macro_f1=metrics['macro_f1']))
        if metrics[selection_metric] < best:
            best = metrics[selection_metric]
            metadata = dict(runtime_thresholds=predictor.thresholds, normalizer=predictor.normalizer,
                            variant=predictor.dimensions['variant'], input_d=predictor.dimensions['input_d'])
            if regression:
                metadata.update(loss_type='rmse', target='observed_runtime', target_units='seconds', output_dim=1)
            torch.save(dict(model_state=predictor.model.state_dict(), metadata=metadata, epoch=epoch), destination / 'best_model.pt')
    # Save the fitted featurizer unchanged, never refit scalers on evaluation data.
    shutil.copyfile(predictor.directory / 'featurizer.pickle', destination / 'featurizer.pickle')
    config = dict(predictor.dimensions, runtime_thresholds=predictor.thresholds)
    if regression:
        config.update(loss_type='rmse', target='observed_runtime', target_units='seconds', output_dim=1)
    (destination / 'model_config.json').write_text(json.dumps(config, indent=2) + '\n')
    report = dict(epochs=args.epochs, history=history, prediction_task=predictor.task,
                  training_states=len(states), validation_states=len(validation), seed=args.seed)
    report[f'best_validation_{selection_metric}'] = best
    (destination / 'training.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
