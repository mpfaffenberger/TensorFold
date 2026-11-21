"""Time warm Inkling HTTP generation; --max-tokens 1 isolates first-token latency."""
import argparse
import http.client
import json
import time


def run(port, prompt, output_budget):
    client = http.client.HTTPConnection('127.0.0.1', port, timeout=3600)
    body = json.dumps({'model': 'inkling', 'messages': [{'role': 'user', 'content': prompt}],
                       'max_tokens': output_budget})
    try:
        start = time.monotonic()
        client.request('POST', '/v1/chat/completions', body, {'Content-Type': 'application/json'})
        response = client.getresponse()
        reply = json.load(response)
        elapsed = time.monotonic() - start
        if response.status != 200:
            raise RuntimeError(reply)
        choice = reply['choices'][0]
        return {'seconds': round(elapsed, 3), 'finish_reason': choice['finish_reason'],
                'answer': choice['message']['content']}
    finally:
        client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18465)
    parser.add_argument('--fill', type=int, default=8200)
    parser.add_argument('--max-tokens', type=int, default=128)
    args = parser.parse_args()
    run(args.port, 'Warm up.', 1)
    prompt = 'Remember the code word MARIGOLD.' + ' x' * args.fill + '\nWhat was the code word? Reply with just that word.'
    print(json.dumps(run(args.port, prompt, args.max_tokens)))
