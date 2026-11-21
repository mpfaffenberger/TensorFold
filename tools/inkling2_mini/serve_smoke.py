"""Real native loopback server check: vary only pixels, then return to text."""
import argparse
import base64
import http.client
import json
import os
import struct
import zlib


def solid_png(rgb: bytes) -> str:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))

    size = 160
    header = struct.pack('!IIBBBBB', size, size, 8, 2, 0, 0, 0)
    raw = (b'\0' + rgb * size) * size
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(png).decode('ascii')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18465)
    parser.add_argument('--api-key', default=os.environ.get('TENSORFOLD_API_KEY', ''))
    args = parser.parse_args()
    headers = {'Content-Type': 'application/json'}
    if args.api_key:
        headers['Authorization'] = f'Bearer {args.api_key}'
    for name, rgb in [('red', bytes([240, 0, 0])), ('blue', bytes([0, 0, 240])), ('4', None)]:
        content = ('What is 2 + 2? Reply with just the number.' if rgb is None else [
            {'type': 'image_url', 'image_url': {'url': solid_png(rgb)}},
            {'type': 'text', 'text': 'What color is this image? Answer in one word.'},
        ])
        client = http.client.HTTPConnection('127.0.0.1', args.port, timeout=180)
        try:
            body = {'model': 'inkling', 'messages': [{'role': 'user', 'content': content}], 'max_tokens': 128}
            client.request('POST', '/v1/chat/completions', json.dumps(body), headers)
            response = client.getresponse()
            result = json.load(response)
            if response.status != 200 or result['choices'][0]['message']['content'].lower().strip(' .\n') != name:
                raise ValueError(f'Expected {name}, got {result}')
            print('PASS:', name)
        finally:
            client.close()


if __name__ == '__main__':
    main()
