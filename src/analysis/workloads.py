"""Compare named sets of workload runs using per-occurrence means."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .io import output_directory
from .diagnostics import run_diagnostics
from .tables import write_latex


@dataclass
class WorkloadRunConfig:
    run_path: list[Path]
    display_name: str | None = None
    def __post_init__(self):
        if isinstance(self.run_path,(str,Path)):self.run_path=[self.run_path]
        self.run_path=[Path(p).expanduser().resolve() for p in self.run_path]
        if not self.run_path:raise ValueError('Each configuration needs at least one run')
        if not self.display_name:self.display_name=self.run_path[0].name


class WorkloadAnalyzer:
    def __init__(self,configs,prepared):
        self.configs=list(configs);self.root=Path(prepared).resolve()
        self.index={x['source']:x for x in json.loads((self.root/'index.json').read_text())['runs']}
        names=[c.display_name for c in self.configs]
        if not names or len(set(names))!=len(names):raise ValueError('Display names must be nonempty and unique')
        used=[str(p) for c in self.configs for p in c.run_path]
        if len(set(used))!=len(used):raise ValueError('Run paths must not appear more than once')
        missing=[p for p in used if p not in self.index]
        if missing:raise ValueError(f'Runs absent from preparation: {missing}')

    def analyze(self,output,metric='latency_s',reference=None,include_incomplete=False):
        if metric not in ('latency_s','wait_s','execution_s'):raise ValueError('Unsupported metric')
        frames=[];run_rows=[]
        for config in self.configs:
            populations=[]
            for path in config.run_path:
                folder=self.root/self.index[str(path)]['folder']
                meta=json.loads((folder/'summary.json').read_text())
                frame=pd.read_csv(folder/'executions.csv',dtype={'query_id':str,'occurrence_id':str})
                recorded=len(frame)
                unknown_status=int(frame.status.eq('unknown').sum())
                if not include_incomplete and (meta['complete'] is False or (~frame.status.isin(['success','unknown'])).any()):
                    raise ValueError(f'Incomplete run: {path}; use --include-incomplete to analyze completed observations')
                frame=frame[frame.status.isin(['success','unknown'])].copy()
                frame[metric]=pd.to_numeric(frame[metric],errors='coerce')
                if frame.empty or frame[metric].isna().any() or not np.isfinite(frame[metric]).all() or frame[metric].lt(0).any():
                    raise ValueError(f'{metric} is missing or invalid in {path}')
                if frame.occurrence_id.duplicated().any():raise ValueError(f'Duplicate occurrences in {path}')
                populations.append(set(zip(frame.occurrence_id,frame.query_id)))
                frame['policy']=config.display_name;frame['run']=str(path)
                frames.append(frame)
                diagnostics=run_diagnostics(frame,path,meta,complete=meta['complete'] is True and recorded==len(frame))
                run_rows.append(dict(policy=config.display_name,run=str(path),queries=len(frame),recorded=recorded,
                                     excluded=recorded-len(frame),unknown_status=unknown_status,
                                     mean_s=float(frame[metric].mean()),p50_s=float(frame[metric].median()),
                                     p95_s=float(frame[metric].quantile(.95)),workload_duration_s=meta['workload_duration_s'],**diagnostics))
            if any(p!=populations[0] for p in populations[1:]):
                raise ValueError(f'Repeated runs in {config.display_name} have different occurrence populations')
        data=pd.concat(frames,ignore_index=True)
        grouped=data.groupby(['policy','occurrence_id','query_id'],sort=False)[metric].mean().reset_index()
        stats=[]
        for label,frame in grouped.groupby('policy',sort=False):
            values=frame[metric]
            source=data[data.policy==label]
            wait=pd.to_numeric(source.wait_s,errors='coerce');execution=pd.to_numeric(source.execution_s,errors='coerce')
            stats.append(dict(policy=label,queries=len(values),mean_s=float(values.mean()),p10_s=float(values.quantile(.1)),p25_s=float(values.quantile(.25)),p50_s=float(values.median()),p75_s=float(values.quantile(.75)),
                              mean_wait_s=float(wait.mean()) if wait.notna().all() else None,mean_execution_s=float(execution.mean()) if execution.notna().all() else None,
                              p90_s=float(values.quantile(.9)),p95_s=float(values.quantile(.95)),p99_s=float(values.quantile(.99))))
        if reference is None:reference=self.configs[0].display_name
        if reference not in {c.display_name for c in self.configs}:raise ValueError('Reference is not a configured policy')
        base=grouped[grouped.policy==reference].set_index(['occurrence_id','query_id'])[metric]
        paired=[]
        for config in self.configs:
            current=grouped[grouped.policy==config.display_name].set_index(['occurrence_id','query_id'])[metric]
            matched=pd.concat([base.rename('reference'),current.rename('current')],axis=1,join='inner')
            valid=matched[matched.reference>0]
            speedup=matched[matched.current>0]
            gain=budget=None
            if set(base.index)==set(current.index):
                x=np.sort(base.to_numpy());y=np.sort(current.to_numpy());points=np.unique(np.concatenate([x,y]))
                differences=np.searchsorted(y,points,side='right')/len(y)-np.searchsorted(x,points,side='right')/len(x)
                index=int(np.argmax(differences));gain=float(differences[index]);budget=float(points[index])
            paired.append(dict(policy=config.display_name,reference=reference,matched=len(matched),
                reference_only=len(base)-len(matched),policy_only=len(current)-len(matched),
                zero_reference_excluded=len(matched)-len(valid),
                s50=float((valid.current/valid.reference).median()) if len(valid) else None,
                median_current_over_reference=float((valid.current/valid.reference).median()) if len(valid) else None,
                max_completion_share_gain=gain,max_completion_share_gain_at_s=budget,
                zero_current_excluded=len(matched)-len(speedup),
                median_speedup=float((speedup.reference/speedup.current).median()) if len(speedup) else None))
        destination=output_directory(output,[self.root,*[p for c in self.configs for p in c.run_path]])
        data.to_csv(destination/'executions.csv',index=False)
        grouped.to_csv(destination/'per_occurrence.csv',index=False)
        pd.DataFrame(run_rows).to_csv(destination/'runs.csv',index=False)
        pd.DataFrame(stats).to_csv(destination/'summary.csv',index=False)
        pd.DataFrame(paired).to_csv(destination/'paired.csv',index=False)
        pairing={p['policy']:p for p in paired}
        write_latex(destination/'workload_summary.tex',['Policy','N','P10 (s)','P25 (s)','P50 (s)','P75 (s)','P95 (s)','S50'],
                    [[s['policy'],s['queries'],s['p10_s'],s['p25_s'],s['p50_s'],s['p75_s'],s['p95_s'],pairing[s['policy']]['s50']] for s in stats])
        write_latex(destination/'run_diagnostics.tex',['Policy','Run','Backlog (s)','Running when backlogged','Prediction calls','Median TCP call (s)','Admission Q-error'],
                    [[r['policy'],Path(r['run']).name,r['backlogged_duration_s'],r['mean_running_while_backlogged'],r['prediction_calls'],r['median_prediction_call_s'],r['admission_median_q_error']] for r in run_rows])
        write_latex(destination/'paired_comparison.tex',['Policy','Reference','Matched','S50','Max completion gain','Latency at max (s)'],
                    [[p['policy'],p['reference'],p['matched'],p['s50'],p['max_completion_share_gain'],p['max_completion_share_gain_at_s']] for p in paired])
        plot_coverage=self._plots(grouped,metric,destination)
        summary=dict(metric=metric,reference=reference,configs=[dict(display_name=c.display_name,run_path=[str(p) for p in c.run_path]) for c in self.configs],runs=run_rows,summary=stats,paired=paired,plot_coverage=plot_coverage)
        (destination/'report.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        return dict(output=str(destination),policies=len(self.configs),runs=len(run_rows),metric=metric)

    @staticmethod
    def _plots(grouped,metric,destination):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        labels=list(dict.fromkeys(grouped.policy)); series=[]; coverage=[]; curves=[]; boxstats=[]
        for label in labels:
            rows=grouped[grouped.policy==label].sort_values(metric,kind='stable')
            rows=rows[rows[metric]>0]; values=rows[metric].to_numpy()
            if not len(values):raise ValueError(f'No positive {metric} values for logarithmic plots: {label}')
            series.append(values);coverage.append(dict(policy=label,included=len(values),nonpositive_excluded=int((grouped.policy.eq(label)&grouped[metric].le(0)).sum())))
            for (_,row),fraction in zip(rows.iterrows(),np.arange(1,len(values)+1)/len(values)):
                curves.append(dict(policy=label,occurrence_id=row.occurrence_id,query_id=row.query_id,value_s=float(row[metric]),cumulative_fraction=float(fraction)))
            low,q1,med,q3,high=np.quantile(values,[.1,.25,.5,.75,.95])
            boxstats.append(dict(label=label,whislo=low,q1=q1,med=med,q3=q3,whishi=high,fliers=[]))
        pd.DataFrame(curves).to_csv(destination/'cdf.csv',index=False)
        pd.DataFrame([{k:v for k,v in stat.items() if k!='fliers'} for stat in boxstats]).to_csv(destination/'boxplot_statistics.csv',index=False)
        colors=['#e6e6e6','#b8d4e8','#f1c39e','#c5bad5','#f2d4df','#dddba9','#a7cdd1']
        lines=['#686868','#377ca8','#c87837','#9467bd','#e377c2','#bcbd22','#17becf']
        markers=['^','X','p','s','*','<','>']
        def style_index(label,index):
            value=label.lower()
            if value=='fifo' or value=='fifo mpl 10':return 0
            if value=='rtq':return 1
            if 'iconq' in value:return 2
            return 3+index%4
        axis_label={'latency_s':'Latency (sec)','execution_s':'Execution time (sec)','wait_s':'Waiting time (sec)'}[metric]
        fig,ax=plt.subplots(figsize=(max(5, .95*len(labels)),3.4))
        artists=ax.bxp(boxstats,showfliers=False,patch_artist=True,widths=.6)
        for i,box in enumerate(artists['boxes']):
            style=style_index(labels[i],i);box.set(facecolor=colors[style],edgecolor='#333333',linewidth=.8,hatch=['','//','..'][style%3])
        for key in ('whiskers','caps','medians'):
            for line in artists[key]:line.set(color='#333333',linewidth=.8)
        ax.set_yscale('log');ax.set_ylabel(axis_label);ax.grid(axis='y',which='major',color='#dddddd',linewidth=.5)
        ax.set_xticks([]);ax.legend(artists['boxes'],labels,loc='upper center',bbox_to_anchor=(.5,-.08),ncol=min(3,len(labels)),fontsize=8,frameon=False);fig.tight_layout()
        fig.savefig(destination/'boxplot.pdf',bbox_inches='tight')
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(5,3.4))
        for i,(label,values) in enumerate(zip(labels,series)):
            style=style_index(label,i);ax.plot(values,np.arange(1,len(values)+1)/len(values),label=label,color=lines[style],marker=markers[style],linewidth=1,markersize=3,markerfacecolor='white',markevery=max(1,len(values)//8))
        ax.set_xscale('log');ax.set_ylim(-.05,1.05);ax.set_xlabel(axis_label);ax.set_ylabel('Fraction of queries')
        ax.grid(True,which='major',linestyle='--',linewidth=.45,alpha=.5);ax.grid(True,which='minor',linestyle='--',linewidth=.3,alpha=.3)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=min(3,len(labels)),fontsize=8,frameon=False);fig.tight_layout()
        fig.savefig(destination/'cdf.pdf',bbox_inches='tight')
        plt.close(fig)
        return coverage
