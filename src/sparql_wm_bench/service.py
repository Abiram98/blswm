"""Optional length-prefixed JSON inference protocol; errors never fall back."""
import json
import socket
import socketserver
import struct
from .schema import validate_predictions

def receive(sock):
    def exact(n):
        parts=b''
        while len(parts)<n:
            part=sock.recv(n-len(parts))
            if not part: raise ConnectionError('Truncated frame')
            parts+=part
        return parts
    n=struct.unpack('!I',exact(4))[0]
    if n>16*1024*1024: raise ValueError('Frame too large')
    return json.loads(exact(n))

def send(sock, payload):
    data=json.dumps(payload,allow_nan=False).encode(); sock.sendall(struct.pack('!I',len(data))+data)

def handle(model, request):
    if request.get('schema_version')!=1 or request.get('model_type')!=model.config['model_type'] or request.get('thresholds')!=model.meta.get('thresholds',[120]):
        raise ValueError('Request metadata mismatch')
    items=request['S']
    keys=[x['occurrence_id'] for x in items]
    if not items or len(keys)!=len(set(keys)) or any(not x.get('query_text') for x in items): raise ValueError('Invalid state')
    values=model.state(items)
    task='regression' if model.config['model_type'].endswith('_reg') else 'classification'
    validate_predictions(values,keys,task)
    return dict(schema_version=1,model_type=model.config['model_type'],units=model.meta['units'],thresholds=model.meta.get('thresholds',[120]),semantics='total_execution_runtime',predictions=values)

def serve(model, host='127.0.0.1', port=19022):
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.settimeout(60)
            try: response=handle(model,receive(self.request))
            except Exception as e: response={'error':f'{type(e).__name__}: {e}'}
            send(self.request,response)
    with socketserver.TCPServer((host,port),Handler) as server: server.serve_forever()

class Remote:
    def __init__(self, host, port, model_type, catalog, timeout=60):
        self.host,self.port,self.model_type,self.timeout=host,port,model_type,timeout
        self.texts={r['query_id']:r['query_text'] for r in catalog}
    def predict_joint(self,candidate,running,now):
        items=[dict(occurrence_id=r.occurrence.occurrence_id,query_text=self.texts[r.occurrence.query_id],admission_s=r.admission_s) for r in running]
        items.append(dict(occurrence_id=candidate.occurrence_id,query_text=self.texts[candidate.query_id],admission_s=now))
        with socket.create_connection((self.host,self.port),timeout=self.timeout) as sock:
            send(sock,dict(schema_version=1,model_type=self.model_type,thresholds=[120],S=items)); result=receive(sock)
        if 'error' in result: raise RuntimeError(result['error'])
        task='regression' if self.model_type.endswith('_reg') else 'classification'
        if result.get('schema_version')!=1 or result.get('thresholds')!=[120] or result.get('semantics')!='total_execution_runtime': raise ValueError('Response schema mismatch')
        if result.get('model_type')!=self.model_type or result.get('units')!=('seconds' if task=='regression' else 'class'): raise ValueError('Response metadata mismatch')
        return validate_predictions(result['predictions'],[x['occurrence_id'] for x in items],task)
    def predict(self,candidate,running,now):
        if self.model_type=='planrgcn_class': running=()
        return self.predict_joint(candidate,running,now)[candidate.occurrence_id]
