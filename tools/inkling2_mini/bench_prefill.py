"""Time warm Inkling HTTP generation; --max-tokens 1 isolates first-token latency.

Use --document with real text. Synthetic filler repeats one token, which routes
to a handful of experts and makes the expert path look faster than it is.
"""
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


def build_prompt(args):
    if args.document:
        with open(args.document, encoding='utf-8') as handle:
            text = handle.read(args.characters)
        return f'{text}\n\n{args.question}'
    return ('Remember the code word MARIGOLD.' + ' x' * args.fill
            + '\nWhat was the code word? Reply with just that word.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18465)
    parser.add_argument('--document', help='UTF-8 text file used as the prompt body')
    parser.add_argument('--characters', type=int, default=30000, help='characters of --document to send')
    parser.add_argument('--question', default='In one sentence, what is the text above about?')
    parser.add_argument('--fill', type=int, default=8200, help='synthetic filler tokens when no --document')
    parser.add_argument('--max-tokens', type=int, default=128)
    args = parser.parse_args()
    run(args.port, 'Warm up.', 1)
    print(json.dumps(run(args.port, build_prompt(args), args.max_tokens)))
