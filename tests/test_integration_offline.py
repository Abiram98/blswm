"""Integration tests requiring no external service, GPU or Java."""
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import pytest
from sparql_wm_bench.service import send,receive,handle
from sparql_wm_bench.runtime import HTTPExecutor,replay
from sparql_wm_bench.policies import Policy
from sparql_wm_bench.schema import FakePredictor

def test_frames():
    a,b=socket.socketpair()
    try:
        send(a,{'S':[{'occurrence_id':'x'}]}); assert receive(b)=={'S':[{'occurrence_id':'x'}]}
    finally: a.close(); b.close()

def test_validation():
    class Model:
        config={'model_type':'bi_scqpp_class'}
        meta={'units':'class'}
        def state(self,s): return {x['occurrence_id']:0 for x in s}
    request=dict(schema_version=1,model_type='bi_scqpp_class',thresholds=[120],S=[dict(occurrence_id='a',query_text='select')])
    assert handle(Model(),request)['predictions']=={'a':0}
    request['thresholds']=[60]
    with pytest.raises(ValueError): handle(Model(),request)

def test_http_full_response_and_failures():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200); self.end_headers(); self.wfile.write(b'first'); self.wfile.flush()
            time.sleep(.04); self.wfile.write(b'last')
        def log_message(self,*a): pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        trace=[dict(occurrence_id='o',query_id='q',arrival_s=0,trace_index=0)]
        catalog=[dict(query_id='q',query_text='SELECT * WHERE {?s ?p ?o}')]
        run=replay(trace,catalog,Policy('fifo'),FakePredictor(),executor=HTTPExecutor(f'http://127.0.0.1:{server.server_port}'))
        assert run['complete']; assert run['records'][0]['execution_s']>=.04
        class Broken:
            def execute(self,text): raise RuntimeError('failure')
        run=replay(trace,catalog,Policy('fifo'),FakePredictor(),executor=Broken())
        assert not run['complete']; assert run['records'][0]['status']=='failed'
    finally: server.shutdown(); server.server_close();thread.join()
