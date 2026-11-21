"""Live checks of the Inkling server's chat behaviour: system prompts, history,
streaming, tolerated and refused options, and a client that hangs up mid-stream.

    python serve_chat_check.py --port 18465 [--api-key KEY]
"""
import argparse
import http.client
import json
import os
import time


class Server:
    def __init__(self, port, key, model):
        self.port, self.model = port, model
        self.headers = {'Content-Type': 'application/json'}
        if key:
            self.headers['Authorization'] = f'Bearer {key}'

    def post(self, messages, **options):
        client = http.client.HTTPConnection('127.0.0.1', self.port, timeout=600)
        client.request('POST', '/v1/chat/completions', json.dumps({'model': self.model, 'messages': messages, **options}), self.headers)
        return client, client.getresponse()

    def ask(self, messages, **options):
        client, response = self.post(messages, **options)
        try:
            return response.status, json.load(response)
        finally:
            client.close()

    def stream(self, messages, **options):
        """Returns (answer, reasoning, finish_reason, usage, saw_done, seconds_to_first_event)."""
        start = time.monotonic()
        client, response = self.post(messages, stream=True, **options)
        if response.status != 200:
            raise RuntimeError(f'{response.status}: {response.read()[:300]!r}')
        if 'text/event-stream' not in response.getheader('Content-Type', ''):
            raise RuntimeError('stream was not served as text/event-stream')
        answer, reasoning, finish, usage, done, first = [], [], None, None, False, None
        for raw in response:
            line = raw.decode('utf-8').strip()   # raises if a character was split across events
            if not line.startswith('data: '):
                continue
            first = first if first is not None else time.monotonic() - start
            if line == 'data: [DONE]':
                done = True
                continue
            event = json.loads(line[6:])
            usage = event.get('usage') or usage
            for choice in event.get('choices', []):
                answer.append(choice['delta'].get('content') or '')
                reasoning.append(choice['delta'].get('reasoning_content') or '')
                finish = choice.get('finish_reason') or finish
        client.close()
        return ''.join(answer), ''.join(reasoning), finish, usage, done, first


def expect(condition, description, detail=''):
    print(('PASS: ' if condition else 'FAIL: ') + description + (f' -> {detail}' if detail and not condition else ''), flush=True)
    if not condition:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18465)
    parser.add_argument('--api-key', default=os.environ.get('TENSORFOLD_API_KEY', ''))
    parser.add_argument('--model', default='inkling')
    args = parser.parse_args()
    server = Server(args.port, args.api_key, args.model)
    name = [{'role': 'user', 'content': 'My name is Mike. Remember it.'},
            {'role': 'assistant', 'content': 'Got it, Mike.'},
            {'role': 'user', 'content': 'What is my name? Reply with just the name.'}]

    status, reply = server.ask([{'role': 'system', 'content': 'Always answer in French, in one short sentence.'},
                                {'role': 'user', 'content': 'What colour is the sky on a clear day?'}], max_tokens=300, temperature=0.7, top_p=0.9)
    answer = reply['choices'][0]['message']['content'] if status == 200 else str(reply)
    expect(status == 200 and 'bleu' in answer.lower(), 'system prompt is followed (French answer), sampling options tolerated', answer)
    expect(reply['usage']['prompt_tokens'] > 20 and reply['usage']['completion_tokens'] > 0, 'usage is reported', str(reply.get('usage')))

    status, reply = server.ask(name, max_tokens=300)
    answer = reply['choices'][0]['message']['content'] if status == 200 else str(reply)
    expect(status == 200 and 'mike' in answer.lower(), 'earlier turns are remembered', answer)

    answer, reasoning, finish, usage, done, first = server.stream(name, max_tokens=300, stream_options={'include_usage': True})
    expect('mike' in answer.lower() and finish == 'stop' and done, 'streaming returns the same kind of answer and ends with [DONE]', f'{answer!r} {finish} {done}')
    expect(usage and usage['completion_tokens'] > 0, 'streaming reports usage when asked', str(usage))
    expect(first < 1.0, 'the stream opens before generation finishes', f'{first:.2f}s')

    answer, _, finish, _, _, _ = server.stream([{'role': 'user', 'content': 'Repeat exactly: prix 5 \u20ac, caf\u00e9, \u65e5\u672c\u8a9e, \U0001f600'}], max_tokens=300)
    expect(all(piece in answer for piece in ('\u20ac', 'caf\u00e9', '\u65e5\u672c\u8a9e', '\U0001f600')), 'multi-byte characters survive streaming', answer)

    _, _, finish, _, _, _ = server.stream([{'role': 'user', 'content': 'Count from 1 to 200.'}], max_tokens=5)
    expect(finish == 'length', 'a spent token budget finishes with "length"', str(finish))
    status, _ = server.ask([{'role': 'user', 'content': 'Hi'}], max_tokens=1000000)
    expect(status == 200, 'an oversized max_tokens is clamped, not refused', str(status))

    for label, options in (('tools', {'tools': [{'type': 'function', 'function': {'name': 'f', 'parameters': {}}}]}),
                           ('stop sequences', {'stop': ['END']}), ('JSON mode', {'response_format': {'type': 'json_object'}})):
        status, reply = server.ask([{'role': 'user', 'content': 'Hi'}], **options)
        expect(status == 400 and 'supported' in reply['error']['message'], f'{label}: refused with an explanation', str(reply))

    # A client that hangs up must not hold the single engine for the rest of its budget.
    client, response = server.post([{'role': 'user', 'content': 'Write a very long story about a lighthouse.'}], stream=True, max_tokens=4000)
    response.readline()
    time.sleep(2)
    response.close(); client.close()
    start = time.monotonic()
    status, _ = server.ask([{'role': 'user', 'content': 'Say OK.'}], max_tokens=20)
    waited = time.monotonic() - start
    expect(status == 200 and waited < 20, 'a dropped stream releases the engine promptly', f'{waited:.1f}s')


if __name__ == '__main__':
    main()
