"""Resuming a prompt must not change its result.

Starts the Inkling server several times with different settings and compares the
next-token logits it leaves after prefill (TF_INKLING_DUMP_LOGITS). A prompt
resumed from saved state must match, bit for bit, the same prompt evaluated from
scratch when both split the work at the same positions.

    python prefix_cache_check.py --dump /tmp/logits.f32 -- tf-inkling-hip serve /path/to/Inkling-2-Mini --port 18467

The server command inherits TF_INKLING_PREFILL_CHUNK, TF_INKLING_PREFIX_CACHE and
TF_INKLING_DUMP_LOGITS from this script. Under a container runtime, forward them
(for podman: -e NAME) and give --dump-as the path the server sees.
"""
import argparse
import hashlib
import http.client
import json
import os
import pathlib
import subprocess
import sys
import time


class Server:
    def __init__(self, command, port, dump, dump_as):
        self.command, self.port, self.dump, self.dump_as, self.process = command, port, pathlib.Path(dump), dump_as, None

    def start(self, chunk, cache):
        self.stop()
        environment = dict(os.environ, TF_INKLING_PREFILL_CHUNK=str(chunk), TF_INKLING_PREFIX_CACHE='1' if cache else '0',
                           TF_INKLING_DUMP_LOGITS=self.dump_as)
        self.process = subprocess.Popen(self.command, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(240):
            if self.process.poll() is not None:
                raise RuntimeError('the server exited while starting')
            try:
                client = http.client.HTTPConnection('127.0.0.1', self.port, timeout=2)
                client.request('GET', '/health')
                client.getresponse().read()
                return
            except OSError:
                time.sleep(0.5)
        raise RuntimeError('the server did not start')

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None

    def ask(self, messages, tokens=1):
        """Returns (prompt tokens, digest of the logits left after prefill)."""
        headers = {'Content-Type': 'application/json'}
        if os.environ.get('TENSORFOLD_API_KEY'):
            headers['Authorization'] = 'Bearer ' + os.environ['TENSORFOLD_API_KEY']
        client = http.client.HTTPConnection('127.0.0.1', self.port, timeout=900)
        client.request('POST', '/v1/chat/completions', json.dumps({'messages': messages, 'max_tokens': tokens}), headers)
        response = client.getresponse()
        reply = json.load(response)
        client.close()
        if response.status != 200:
            raise RuntimeError(reply)
        return reply['usage']['prompt_tokens'], hashlib.sha256(self.dump.read_bytes()).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port', type=int, default=18467)
    parser.add_argument('--dump', required=True, help='file the server writes its logits to, as this script sees it')
    parser.add_argument('--dump-as', help='the same file as the server sees it (default: --dump)')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='-- followed by the server command')
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('give the server command after --')
    server = Server(command, args.port, args.dump, args.dump_as or args.dump)

    # About a thousand tokens shared by every prompt, so that snapshots at 256, 512 and 768 exist.
    shared = ' '.join(f'Entry {index}: the {("red", "quiet", "northern", "seventh")[index % 4]} lighthouse logged {index * 37 % 101} ships.'
                      for index in range(70))
    system = {'role': 'system', 'content': shared}
    first = [system, {'role': 'user', 'content': 'Summarise the log in five words.'}]
    second = [system, {'role': 'user', 'content': 'Which entry mentions the most ships? Answer with its number.'}]
    extended = first + [{'role': 'assistant', 'content': 'Lighthouses logged many passing ships.'},
                        {'role': 'user', 'content': 'Now list the four kinds of lighthouse mentioned, one per line.'}]
    failures = 0

    def verdict(label, resumed, scratch):
        nonlocal failures
        failures += resumed != scratch
        print(f'{"PASS" if resumed == scratch else "FAIL"}: {label}: resumed {resumed}, from scratch {scratch}', flush=True)

    try:
        server.start(256, True)
        length, _ = server.ask(first)
        if length < 800:
            raise RuntimeError(f'the shared prompt is only {length} tokens; the check needs snapshots up to 768')
        _, resumed = server.ask(second)
        _, repeated = server.ask(second)
        server.start(256, False)
        _, scratch = server.ask(second)
        verdict('a new question after a shared system prompt', resumed, scratch)
        verdict('the same prompt sent twice', repeated, scratch)

        server.start(256, True)
        server.ask(first, 200)
        _, resumed = server.ask(second)
        verdict('with 200 tokens generated in between', resumed, scratch)

        # A continuing conversation resumes at the end of the previous prompt, so chunk there.
        server.start(length, True)
        server.ask(first)
        _, resumed = server.ask(extended)
        server.start(length, False)
        _, scratch = server.ask(extended)
        verdict('a conversation continued', resumed, scratch)
    finally:
        server.stop()
    return failures


if __name__ == '__main__':
    sys.exit(main())
