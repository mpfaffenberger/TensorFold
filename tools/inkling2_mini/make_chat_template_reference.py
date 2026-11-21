"""Record the token IDs the checkpoint's own chat template produces.

Run with the pinned Transformers build that carries Inkling 2 support:

    python make_chat_template_reference.py /path/to/Inkling-2-Mini > chat-template-reference.json

The native server's chat rendering is checked against this file by
zig/kernels/hip/inkling_mini/chat_template_test.cpp.
"""
import json
import sys

import transformers

IMAGE = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AQI='}}
CASES = [
    ('single user turn', None, [{'role': 'user', 'content': 'What is 2 + 2? Reply with just the number.'}]),
    ('system prompt then user', None, [{'role': 'system', 'content': 'You are terse.'},
                                       {'role': 'user', 'content': 'Name a prime.'}]),
    ('history with reasoning', None, [{'role': 'system', 'content': 'Be brief.'},
                                      {'role': 'user', 'content': 'Hi'},
                                      {'role': 'assistant', 'content': 'Hello.', 'reasoning_content': 'A greeting.'},
                                      {'role': 'user', 'content': 'And now?\nSecond line \u20ac \U0001f600'}]),
    ('assistant turn with reasoning and no answer', None, [{'role': 'user', 'content': 'Think.'},
                                                           {'role': 'assistant', 'content': '', 'reasoning_content': 'Done thinking.'},
                                                           {'role': 'user', 'content': 'Well?'}]),
    ('assistant turn without content', None, [{'role': 'user', 'content': 'a'},
                                              {'role': 'assistant', 'content': None},
                                              {'role': 'user', 'content': 'b'}]),
    ('text part list', None, [{'role': 'user', 'content': [{'type': 'text', 'text': 'first'}, {'type': 'text', 'text': ''},
                                                           {'type': 'text', 'text': 'third'}]}]),
    ('image then text', None, [{'role': 'user', 'content': [IMAGE, {'type': 'text', 'text': 'What colour is this?'}]}]),
    ('system message only', None, [{'role': 'system', 'content': 'Only instructions.'}]),
    ('two system messages', None, [{'role': 'system', 'content': 'One.'}, {'role': 'system', 'content': 'Two.'},
                                   {'role': 'user', 'content': 'Go.'}]),
    ('named effort', 'high', [{'role': 'user', 'content': 'Explain gravity.'}]),
    ('numeric effort', 0.35, [{'role': 'user', 'content': 'Explain gravity.'}]),
    ('minimal effort', 'minimal', [{'role': 'user', 'content': 'Explain gravity.'}]),
]


def main() -> None:
    tokenizer = transformers.AutoTokenizer.from_pretrained(sys.argv[1])
    reference = []
    for name, effort, messages in CASES:
        # The server's default is zero effort; the template's own default is 0.9.
        rendered = tokenizer.apply_chat_template(messages, reasoning_effort=0 if effort is None else effort,
                                                 add_generation_prompt=True, tokenize=True)
        token_ids = rendered['input_ids'] if hasattr(rendered, 'keys') else rendered
        request = {'messages': messages}
        if effort is not None:
            request['reasoning_effort'] = effort
        reference.append({'name': name, 'request': request, 'token_ids': [int(token) for token in token_ids]})
    json.dump({'transformers': transformers.__version__, 'cases': reference}, sys.stdout, indent=1)


if __name__ == '__main__':
    main()
