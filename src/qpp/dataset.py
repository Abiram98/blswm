"""Build query-set training data from completed policy execution records."""
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from collections import defaultdict
from wrk_gen.queries import new_directory, fingerprint


def build_dataset(workload, runs, output, *, split='train_validation', validation_fraction=.2,
                  seed=0, threshold=120., sample_mode='all_query_start_events', dataset_format='queryset'):
    if sample_mode not in ('all_query_start_events','all_decision_events','aligned_hard_prefix'):
        raise ValueError('Invalid sample mode')
    if dataset_format not in ('queryset', 'candidate'):
        raise ValueError('Invalid dataset format')
    if dataset_format == 'candidate' and sample_mode != 'all_query_start_events':
        raise ValueError('Candidate datasets require query start events')
    if not 0 < validation_fraction < 1 or not math.isfinite(threshold) or threshold <= 0 or seed < 0:
        raise ValueError('Invalid split fraction, seed or runtime threshold')
    source=Path(workload).resolve()
    ql=source/'ql.csv' if source.is_dir() else source
    with ql.open(newline='',encoding='utf-8-sig') as stream:
        rows=list(csv.DictReader(stream))
    texts={(row.get('id') or row.get('query_id') or ''):row.get('text') or row.get('query_text') or row.get('queryString') for row in rows}
    if len(texts)!=len(rows) or any(not k or not isinstance(v,str) or not v.strip() for k,v in texts.items()):
        raise ValueError('Workload needs unique IDs and nonempty query texts')
    samples=[]; hashes={str(ql):fingerprint(ql)}; sample_traces={}
    workload_id=ql.parent.name
    for run_number, path in enumerate(runs,1):
        path=Path(path).resolve();path=path/'run.json' if path.is_dir() else path
        if str(path) in hashes:raise ValueError('Duplicate execution input')
        hashes[str(path)]=fingerprint(path)
        result=json.loads(path.read_text())
        if not result.get('complete') or result.get('mode')!='real':raise ValueError(f'Execution must be complete and real: {path}')
        expected_hash=hashlib.sha256(json.dumps(texts,sort_keys=True,allow_nan=False).encode()).hexdigest()
        if result.get('catalog_hash') and result['catalog_hash']!=expected_hash:
            raise ValueError(f'Execution catalog does not match workload: {path}')
        records=result['records']
        if not records or any(r['status']!='success' or r['query_id'] not in texts for r in records):
            raise ValueError(f'Execution contains failed or unknown queries: {path}')
        if any(r['policy']!='fifo' for r in records):raise ValueError('Dataset construction requires FIFO executions')
        records=sorted(records,key=lambda r:(r['admission_s'],r['occurrence_id']))
        if len({r['occurrence_id'] for r in records})!=len(records):raise ValueError('Duplicate occurrence IDs')
        for record in records:
            a,b=record['admission_s'],record['completion_s']
            if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (a,b)) or a<0 or b<a:
                raise ValueError('Invalid execution interval')
        events=sorted({(r['admission_s'],'query_start') for r in records} |
                      ({(r['completion_s'],'query_finish') for r in records} if sample_mode=='all_decision_events' else set()))
        for index,(now,event) in enumerate(events,1):
            selected=[(i,r,'running' if r['admission_s']<now else 'candidate') for i,r in enumerate(records,1)
                      if r['admission_s']<now<r['completion_s'] or event=='query_start' and r['admission_s']==now]
            if not selected:continue
            selected.sort(key=lambda x:(0 if x[2]=='running' else 1,x[1]['admission_s'],x[0]))
            cut=max(i for i,r in enumerate(records,1) if r['admission_s']<=now)
            prefix=sum(r['completion_s']-r['admission_s']>=threshold for r in records[:cut])
            items=[]
            for pos,(order,r,role) in enumerate(selected):
                duration=r['completion_s']-r['admission_s'];label=int(duration>=threshold)
                items.append(dict(pos=pos,query_id=r['query_id'],query_order_index=order,role=role,
                    arrival_time=r['arrival_s'],start_time=r['admission_s'],end_time=r['completion_s'],
                    elapsed_at_decision=max(0.,now-r['admission_s']),remaining_time_from_decision=max(0.,r['completion_s']-now),
                    observed_runtime=duration,runtime_interval=f'[{threshold}, inf)' if label else f'[0.0, {threshold})',
                    label=label,query_path=None,query_text=texts[r['query_id']],query_text_available=True))
            group=(f'{workload_id}::qidx{cut}' if event=='query_start' else f'{workload_id}::{event}::qidx{cut}')
            sample_id=hashlib.sha1(f'{path}::{index}::{event}'.encode()).hexdigest()[:16]
            sample_traces['qs_'+sample_id]=records
            samples.append(dict(sample_id='qs_'+sample_id,split_group_id=group,workload_id=workload_id,
                run_id=f'{result.get("run_id","run")}:{run_number}',policy='fifo',mpl=int(result['mpl']),
                decision_time=now,decision_event_type=event,query_order_cutpoint=cut,hard_prefix_count=prefix,
                hard_cutpoint_target=prefix,hard_cutpoint_deviation=0,set_size=len(items),S=items,
                y=[x['label'] for x in items],runtime_intervals=[x['runtime_interval'] for x in items]))
    targets=[]
    if sample_mode=='aligned_hard_prefix' and samples:
        traces=defaultdict(list)
        for sample in samples:traces[sample['run_id']].append(sample)
        for trace in traces.values():trace.sort(key=lambda row:(row['decision_time'],row['query_order_cutpoint']))
        sets=[{row['hard_prefix_count'] for row in trace} for trace in traces.values()]
        shared=set.intersection(*sets); union=set.union(*sets)
        targets=sorted(union if shared=={0} and len(union)>1 else shared or union)
        aligned=[]
        for trace in traces.values():
            for target in targets:
                choices=[(i,row) for i,row in enumerate(trace) if row['hard_prefix_count']==target]
                if choices:
                    hard=[pair for pair in choices if any(q['role']=='candidate' and q['label']==1 for q in pair[1]['S'])]
                    diverse=[]
                    for i,row in choices:
                        labels={q['label'] for item in trace[max(0,i-2):min(len(trace),i+4)]
                                for q in item['S'] if q['role']=='candidate'}
                        if len(labels)>1:diverse.append((i,row))
                    pool=hard or diverse or choices
                    _,chosen=min(pool,key=lambda pair:(abs(pair[1]['query_order_cutpoint']-target),pair[1]['query_order_cutpoint'],pair[0]))
                else:
                    chosen=min(trace,key=lambda row:(abs(row['hard_prefix_count']-target),abs(row['query_order_cutpoint']-target),row['query_order_cutpoint']))
                chosen=dict(chosen,hard_cutpoint_target=target,
                            hard_cutpoint_deviation=abs(chosen['hard_prefix_count']-target))
                chosen['sample_id']='qs_'+hashlib.sha1((chosen['sample_id']+'::'+str(target)).encode()).hexdigest()[:16]
                aligned.append(chosen)
        samples=aligned
    if not samples:raise ValueError('No query-set samples')
    partitions={'test':samples}
    overlap=0
    if split=='train_validation':
        grouped=defaultdict(list)
        if dataset_format == 'candidate':
            # Keep every occurrence of the same query text in one partition.
            parent={}
            def find(key):
                parent.setdefault(key,key)
                if parent[key]!=key:parent[key]=find(parent[key])
                return parent[key]
            for sample in samples:
                identities=[q['query_text'].strip() for q in sample['S'] if q['role']=='candidate']
                for key in identities[1:]:parent[find(key)]=find(identities[0])
            for sample in samples:
                key=next(q['query_text'].strip() for q in sample['S'] if q['role']=='candidate')
                grouped[find(key)].append(sample)
        else:
            for sample in samples:grouped[sample['split_group_id']].append(sample)
        keys=sorted(grouped)
        if len(keys)<2:raise ValueError('At least two decision groups are required')
        count=max(1,min(len(keys)-1,round(len(keys)*validation_fraction)))
        rng=random.Random(seed); best=None
        for _ in range(512):
            chosen=set(rng.sample(keys,count))
            train=[s for key in keys if key not in chosen for s in grouped[key]]
            val=[s for key in keys if key in chosen for s in grouped[key]]
            a={i['query_text'].strip() for s in train for i in s['S']};b={i['query_text'].strip() for s in val for i in s['S']}
            score=(len(a&b),abs(len(val)/len(samples)-validation_fraction))
            if best is None or score<best[0]:best=(score,train,val)
        overlap=best[0][0];partitions={'train':best[1],'validation':best[2]}
    if any(fingerprint(path)!=digest for path,digest in hashes.items()):raise ValueError('Input changed during dataset creation')
    destination=new_directory(output,[ql.parent,*[Path(p).resolve() for p in runs if Path(p).is_dir()]])
    output_counts={}
    for name,entries in partitions.items():
        if dataset_format == 'queryset':
            output_counts[name]=len(entries)
            with (destination/f'{name}.jsonl').open('w') as stream:
                for entry in entries:stream.write(json.dumps(entry,sort_keys=True,allow_nan=False)+'\n')
        else:
            fields=['Query_ID','Execution_Time','Running_Queries_At_Start','Queries_Started_During','query_label','Query_File','Start_Time','End_Time','query_text','resource_before','resource_during','resource_after','too_fast_to_measure']
            candidates=[]
            for entry in entries:
                running=[dict(query_id=q['query_id'],query_text=q['query_text'],start_time=q['start_time'],end_time=q['end_time'])
                         for q in entry['S'] if q['role']=='running']
                for query in entry['S']:
                    if query['role']=='candidate':
                        started=[dict(query_id=r['query_id'],query_text=texts[r['query_id']],
                                      start_time=r['admission_s'],end_time=r['completion_s'])
                                 for r in sample_traces[entry['sample_id']]
                                 if query['start_time']<r['admission_s']<query['end_time']]
                        candidates.append(dict(Query_ID=query['query_id'],Execution_Time=query['observed_runtime'],
                            Running_Queries_At_Start=repr(running),Queries_Started_During=repr(started),
                            query_label=query['runtime_interval'],Query_File='',Start_Time=query['start_time'],
                            End_Time=query['end_time'],query_text=query['query_text'],resource_before='',
                            resource_during='',resource_after='',too_fast_to_measure=''))
            output_counts[name]=len(candidates)
            csv_name='val' if name=='validation' else name
            with (destination/f'{csv_name}.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(candidates)
            with (destination/f'planrgcn_{name}.tsv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['id','queryString','mean_latency'],delimiter='\t')
                writer.writeheader()
                writer.writerows(dict(id=row['Query_ID'],queryString=row['query_text'],mean_latency=row['Execution_Time']) for row in candidates)
    manifest=dict(schema_version=1,dataset_format=dataset_format,split=split,sample_mode=sample_mode,hard_prefix_targets=targets,threshold_seconds=threshold,seed=seed,
                  validation_fraction=validation_fraction,input_sha256=hashes,sample_counts=output_counts,
                  train_validation_overlapping_query_texts=overlap,split_method=('candidate text groups' if dataset_format=='candidate' else 'decision groups') + ', minimum context overlap among 512 seeded candidates' if split=='train_validation' else 'collective test')
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return dict(output=str(destination),**manifest)
