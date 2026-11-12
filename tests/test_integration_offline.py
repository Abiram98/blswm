"""Integration tests requiring no external service, GPU or Java."""
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from policies.runtime import HTTPExecutor,replay
from policies.scheduling import Policy
from policies.schema import FakePredictor



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
