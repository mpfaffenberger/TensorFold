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


WEATHER = {'type': 'function', 'function': {'name': 'get_weather', 'description': 'Current weather for a city.',
           'parameters': {'type': 'object', 'required': ['city'], 'additionalProperties': False,
                          'properties': {'city': {'type': 'string', 'description': 'City name'},
                                         'unit': {'enum': ['c', 'f'], 'type': 'string', 'default': 'c'}}}}}
BARE = {'name': 'ping', 'parameters': {}}  # no wrapper, no description, empty parameters


def call(identifier, name, arguments):
    """An assistant tool call as OpenAI clients send it: arguments are a JSON string."""
    return {'id': identifier, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments, ensure_ascii=False)}}


# (name, reasoning_effort or None, messages, tools or None)
TOOL_CASES = [
    ('tools declared', None, [{'role': 'user', 'content': 'Weather in Paris?'}], [WEATHER, BARE]),
    ('tools with a system prompt', 'low', [{'role': 'system', 'content': 'Use tools.'}, {'role': 'user', 'content': 'Weather in Paris?'}], [WEATHER]),
    ('tool call and result by id', None, [
        {'role': 'user', 'content': 'Weather in Paris?'},
        {'role': 'assistant', 'content': None, 'tool_calls': [call('call_1', 'get_weather', {'unit': 'c', 'city': 'Paris'})]},
        {'role': 'tool', 'tool_call_id': 'call_1', 'content': '{"temperature": 21.5, "sky": "clear"}'},
        {'role': 'user', 'content': 'Thanks.'}], [WEATHER]),
    ('answer, reasoning and two calls', None, [
        {'role': 'user', 'content': 'Compare Paris and Oslo.'},
        {'role': 'assistant', 'content': 'Checking both.', 'reasoning_content': 'Two lookups.',
         'tool_calls': [call('a', 'get_weather', {'city': 'Paris'}), call('b', 'get_weather', {'city': 'Oslo', 'unit': 'f'})]},
        {'role': 'tool', 'tool_call_id': 'a', 'content': '21'},
        {'role': 'tool', 'tool_call_id': 'b', 'content': [{'type': 'text', 'text': '48'}]}], [WEATHER]),
    ('empty answer with a call, result by name', None, [
        {'role': 'user', 'content': 'Ping.'},
        {'role': 'assistant', 'content': '', 'tool_calls': [call('p', 'ping', {})]},
        {'role': 'tool', 'name': 'ping', 'content': ''}], [BARE]),
    ('nested, unicode and numeric arguments', None, [
        {'role': 'user', 'content': 'Go.'},
        {'role': 'assistant', 'content': None, 'tool_calls': [call('n', 'get_weather', {
            'zeta': {'b': 1, 'a': [True, None, 2.5, -3, 1e-05, 1e+20]}, 'city': 'Z\u00fcrich \u65e5\u672c "quoted" \\ back\nline',
            'Alpha': 1.0, 'count': 12345678901234})]},
        {'role': 'tool', 'tool_call_id': 'n', 'content': 'ok'}], [WEATHER]),
]


def main() -> None:
    tokenizer = transformers.AutoTokenizer.from_pretrained(sys.argv[1])
    reference = []
    for name, effort, messages, tools in [(*case, None) for case in CASES] + TOOL_CASES:
        # The template wants parsed argument objects; OpenAI clients send strings.
        rendered_messages = json.loads(json.dumps(messages))
        for message in rendered_messages:
            for made in message.get('tool_calls') or []:
                made['function']['arguments'] = json.loads(made['function']['arguments'])
        # The server's default is zero effort; the template's own default is 0.9.
        rendered = tokenizer.apply_chat_template(rendered_messages, tools=tools, reasoning_effort=0 if effort is None else effort,
                                                 add_generation_prompt=True, tokenize=True)
        token_ids = rendered['input_ids'] if hasattr(rendered, 'keys') else rendered
        request = {'messages': messages}
        if tools is not None:
            request['tools'] = tools
        if effort is not None:
            request['reasoning_effort'] = effort
        reference.append({'name': name, 'request': request, 'token_ids': [int(token) for token in token_ids]})
    json.dump({'transformers': transformers.__version__, 'cases': reference}, sys.stdout, indent=1)


if __name__ == '__main__':
    main()
