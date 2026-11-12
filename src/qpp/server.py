"""Threaded TCP inference server; inference is serialized around model state."""
import socketserver
from .protocol import receive, send


def server_for(predictor, host='127.0.0.1', port=9010):
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.settimeout(60)
            while True:
                try:
                    request = receive(self.request)
                except (EOFError, OSError):
                    return
                except (ValueError, UnicodeError) as exc:
                    send(self.request, dict(status='error', error=str(exc)))
                    return
                try:
                    if request.get('type') == 'info':
                        response = predictor.info()
                    elif request.get('type') == 'predict':
                        response = predictor.predict(request)
                    else:
                        raise ValueError('Request type must be info or predict')
                except Exception as exc:
                    response = dict(status='error', error=f'{type(exc).__name__}: {exc}')
                try:
                    send(self.request, response)
                except OSError:
                    return
    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
    return Server((host, port), Handler)
