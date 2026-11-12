"""Length-prefixed JSON query-set protocol (four-byte network-order length)."""
import json
import math
import socket
import struct

MAX_FRAME = 10 * 1024 * 1024
VERSION = 'queryset-json-v1'
ROLES = {'candidate': 0, 'running': 1, 'unknown': 2}


def receive(sock):
    def exact(size):
        result = bytearray()
        while len(result) < size:
            chunk = sock.recv(size - len(result))
            if not chunk:
                raise EOFError('Truncated frame')
            result.extend(chunk)
        return result
    size = struct.unpack('!I', exact(4))[0]
    if not 0 < size <= MAX_FRAME:
        raise ValueError('Invalid frame size')
    data = json.loads(exact(size))
    if not isinstance(data, dict):
        raise ValueError('Frame must contain a JSON object')
    return data


def send(sock, data):
    payload = json.dumps(data, allow_nan=False, separators=(',', ':')).encode()
    if len(payload) > MAX_FRAME:
        raise ValueError('Frame too large')
    sock.sendall(struct.pack('!I', len(payload)) + payload)


def normalize(queries):
    if not isinstance(queries, list) or not queries:
        raise ValueError('A nonempty queries list is required')
    result = []
    for index, query in enumerate(queries):
        if not isinstance(query, dict) or not isinstance(query.get('query_text'), str) or not query['query_text'].strip():
            raise ValueError(f'Query {index} requires query_text')
        elapsed = float(query.get('elapsed_at_decision', 0))
        position = query.get('pos', index)
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError('pos must be an integer')
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('elapsed_at_decision must be nonnegative and finite')
        role = query.get('role', 'unknown')
        if role not in ROLES:
            raise ValueError(f'Unknown query role: {role}')
        result.append(dict(query, query_id=str(query.get('query_id', f'query_{index}')),
                           pos=position, elapsed_at_decision=elapsed, role=role, input_index=index))
    return sorted(result, key=lambda q: (q['pos'], q['input_index']))


def request(host, port, payload, timeout=60):
    with socket.create_connection((host, port), timeout=timeout) as sock:
        send(sock, payload)
        response = receive(sock)
    if response.get('status') != 'ok':
        raise RuntimeError(response.get('error', 'Invalid predictor response'))
    return response
