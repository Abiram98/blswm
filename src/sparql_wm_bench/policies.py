"""Pure admission rules; executor owns lifecycle and persistent policy state."""
from collections import Counter
from .schema import Decision, validate_predictions

NAMES=('fifo','rr','rtq','rtq-reeval','sqa','sqa-reeval','iconq')

class Policy:
    def __init__(self, name='fifo', mpl=10, budget=10, **options):
        if name not in NAMES or mpl < 1 or budget < 3:
            raise ValueError('Invalid policy configuration')
        self.name,self.mpl,self.budget=name,mpl,budget
        self.options=options

    def decide(self, waiting, running, now, predictor, state):
        if not waiting or len(running)>=self.mpl: return None
        if self.name=='fifo': return Decision(waiting[0],reason='fifo')
        if self.name=='iconq':
            from .iconq import IconQSchedPolicy, PolicyConfig, WaitingQuery, RunningQuery
            class Adapter:
                def predict_set(_, candidate, active, at):
                    cand=next(o for o in waiting if o.occurrence_id==candidate.query_id)
                    rs=tuple(r for r in running if r.occurrence.occurrence_id in {a.query_id for a in active})
                    return predictor.predict_joint(cand,rs,at)
            w=[WaitingQuery(o.occurrence_id,o.query_id,o.arrival_s) for o in waiting]
            r=[RunningQuery(x.occurrence.occurrence_id,x.occurrence.query_id,x.admission_s,x.prediction,
                            x.admission_s+x.prediction) for x in running]
            cfg=PolicyConfig(max_concurrency_level=self.mpl,**self.options)
            d=IconQSchedPolicy(Adapter(),cfg).select(w,r,now)
            if d.query_id is None: return None
            return Decision(next(o for o in waiting if o.occurrence_id==d.query_id),prediction=d.predicted_runtime,reason=d.reason)
        if self.name=='rr':
            classes=state.setdefault('classes',{})
            for o in waiting:
                if o.occurrence_id not in classes:
                    classes[o.occurrence_id]=predictor.predict(o,running,now)
            queues=[[o for o in waiting if classes[o.occurrence_id]==c] for c in (0,1)]
            turn=state.get('turn',0)
            c=turn if queues[turn] else 1-turn
            if all(queues): state['turn']=1-turn
            o=queues[c][0]
            return Decision(o,prediction=c,reason='class_round_robin')
        physical=Counter(r.slot for r in running)
        for o in waiting:
            joint=None
            if self.name.endswith('reeval'):
                joint=predictor.predict_joint(o,running,now)
                c=joint[o.occurrence_id]
            else: c=predictor.predict(o,running,now)
            validate_predictions({o.occurrence_id:c},[o.occurrence_id],'classification')
            if self.name.startswith('rtq'):
                virtual=Counter(joint[r.occurrence.occurrence_id] for r in running) if joint else physical
                for slot in range(int(c),2):
                    if physical[slot]<5 and virtual[slot]<5:
                        return Decision(o,slot=slot,prediction=c,reason='class_capacity')
            else:
                used=sum(1+2*joint[r.occurrence.occurrence_id] for r in running) if joint else sum(r.cost for r in running)
                cost=1+2*c
                if used+cost<=self.budget:
                    return Decision(o,cost=cost,prediction=c,reason='budget')
        return None
