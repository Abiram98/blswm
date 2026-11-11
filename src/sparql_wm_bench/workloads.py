"""Seeded selection and open-loop arrival schedules. CPU units: 100% per core."""
import numpy as np
import math
from .catalog import identity
from .io import digest, write

COMPOSITIONS = dict(zip(['Homogeneous','Runtime-skewed','Long-running','CPU-demanding','Heterogeneous'],
                       [(0,350,0,0,0,0),(0,280,60,10,0,0),(0,140,180,30,0,0),(0,250,35,10,30,25),(20,210,60,10,30,20)]))
PROFILES = ('S','M','L','L100','HM','HL')

def select(catalog, config, excluded=()):
    rng = np.random.default_rng(config.get('selection_seed', 1))
    if config.get('dataset'):
        catalog=[r for r in catalog if r.get('dataset')==config['dataset']]
    if not catalog: raise ValueError('Empty dataset catalog')
    blocked = {identity(r['query_text']) for r in excluded}
    blocked_ids = {r['query_id'] for r in excluded}
    unique = {}
    for r in catalog:
        key = identity(r['query_text'])
        if key not in blocked and r['query_id'] not in blocked_ids and r.get('status','success') == 'success':
            unique.setdefault(key, r)
    rows = list(unique.values())
    def sample(pool, n):
        if len(pool) < n:
            raise ValueError(f'Insufficient distinct candidates: need {n}, available {len(pool)}')
        return [pool[int(i)] for i in rng.choice(len(pool), n, replace=False)]
    if config.get('composition'):
        counts = COMPOSITIONS[config['composition']]
        if 'counts' in config and config['counts']!=dict(zip(PROFILES,counts)):
            raise ValueError('Preset counts differ from paper composition')
        return sum((sample([r for r in rows if r['profile']==p], n) for p,n in zip(PROFILES,counts)), [])
    # Threshold is fitted on the input profiling population, before exclusion.
    threshold = float(np.percentile([r['isolated_runtime_s'] for r in catalog if r.get('status','success')=='success'],85))
    return sample([r for r in rows if r['isolated_runtime_s']<threshold],320)+sample([r for r in rows if r['isolated_runtime_s']>=threshold],30)

def generate(bag, pattern='P', seed=1, mu=.6):
    if len(bag) != 350 or not math.isfinite(mu) or mu <= 0:
        raise ValueError('Paper workloads require 350 queries and a positive rate')
    rng = np.random.default_rng(seed)
    items = [dict(occurrence_id=f'o{i:04}', **r) for i,r in enumerate(bag)]
    rng.shuffle(items)
    groups = []
    if pattern == 'RCB':
        hm = [r for r in items if r['profile']=='HM']
        bg = [r for r in items if r['profile']!='HM']
        if len(hm)!=30: raise ValueError('RCB requires exactly 30 HM')
        cursor=0
        for j,n in enumerate((106,107,107)):
            groups.extend([(bg[cursor:cursor+n],'P',mu),(hm[j*10:j*10+10],'B',.5)])
            cursor+=n
    elif pattern == 'LF':
        lead=[r for r in items if r['profile']=='L100'][:10]
        if len(lead)!=10: raise ValueError('LF requires 10 L100')
        ids={r['occurrence_id'] for r in lead}
        groups=[(lead,'B',.5),([r for r in items if r['occurrence_id'] not in ids],'P',mu)]
    elif pattern == 'PAR':
        groups=[(items[:100],'P',.2),(items[100:250],'P',1.),(items[250:],'P',.2)]
    elif pattern == 'P': groups=[(items,'P',mu)]
    else: raise ValueError('Unknown pattern')
    def burst(n,w): return np.r_[0.,np.sort(rng.uniform(0,w,n-2)),w]
    trace=[]; now=0.; prev=None
    for group,kind,param in groups:
        source=prev or (len(group),kind,param)
        n,k,p=source
        now+=float(rng.exponential(1/p) if k=='P' else burst(n,p)[1])
        offsets=np.r_[0.,np.cumsum(rng.exponential(1/param,len(group)-1))] if kind=='P' else burst(len(group),param)
        for r,offset in zip(group,offsets):
            trace.append(dict(occurrence_id=r['occurrence_id'],query_id=r['query_id'],arrival_s=float(now+offset),trace_index=len(trace)))
        now=trace[-1]['arrival_s']; prev=(len(group),kind,param)
    return trace

def save(folder, bag, trace, config):
    write(f'{folder}/bag.json',bag)
    write(f'{folder}/trace.csv',trace)
    write(f'{folder}/manifest.json',dict(generator_version=1,config=config,input_hash=digest(bag),schedule_hash=digest(trace),occurrence_ids=[r['occurrence_id'] for r in trace]))
