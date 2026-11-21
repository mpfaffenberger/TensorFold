# Building the Inkling-2-Mini HIP Engine

Include `engine.hpp` from a HIP C++17 translation unit. Use a 64-bit,
little-endian Linux host, ROCm HIP headers/compiler/runtime, and the public
[nlohmann/json](https://github.com/nlohmann/json) headers (3.11 or newer).
Install `nlohmann-json3-dev` on Debian/Ubuntu or `json-devel` on Fedora,
inside the build environment, or add `-I/path/to/include` where that directory
contains `nlohmann/json.hpp`. No JSON link flag, Torch, Python runtime, or
checkpoint conversion is needed.

From this directory, with ROCm on the compiler and runtime search paths:

```sh
hipcc -std=c++17 -O2 --offload-arch=gfx1151 engine_smoke.hip -o /tmp/engine-smoke
/tmp/engine-smoke "$HOME/models/Inkling-2-Mini"
```

Pass the original checkpoint directory containing `config.json`,
`model.safetensors.index.json` and its shards. Shards must remain unchanged
while mapped. Construction checks the model family and shard metadata but
does not upload weights. `raw` and `w` have independent caches: use `raw` for
NVFP4/MXFP8 packed tensors. `w` converts the stored scalar dtype (U8 means
one float per byte, **not** unpacked nibbles).

Use one Engine on one HIP device, from one host thread, on the default stream.
All pointers are borrowed from the Engine. Scratch contents persist between
calls. Adding other names never invalidates pointers. Growing the **same**
name preserves its prefix and zeroes the extension, but invalidates the old
pointer. `mv` enqueues work; output must not alias either input. Synchronize or
copy back before consuming results on the host. `INKLING_HIP_CHECK(call)`
throws on HIP errors; a subsequent sync may surface asynchronous failures.

## Single-turn text generation (experimental)

Build `main.hip` with the same settings as the smoke test and link the PCRE2-8, zlib and hipBLAS development libraries:

```sh
hipcc -std=c++17 -O2 --offload-arch=gfx1151 main.hip -lpcre2-8 -lz -lhipblas -o tf-inkling-hip
./tf-inkling-hip "$HOME/models/Inkling-2-Mini" 'What is the capital of Japan?' 96
```

Install the PCRE2 development headers in the build environment. A standalone
TensorFold build step is also provided: `zig build -Dhip=true tf-inkling-hip`.
The direct HIP build is the one exercised on the Radeon 8060S. The CLI performs
single-turn, batch-one greedy decoding and prints protocol tokens as generated.
Set `TF_INKLING_TOKEN_IDS=1` for decimal IDs. This binary runs independently of Python, PyTorch and vLLM.

## Chat server

```sh
zig build -Dhip=true native-hip tf-inkling-hip
zig-out/native-hip/bin/tensorfold-native serve /path/to/Inkling-2-Mini \
  --backend rocm --host 127.0.0.1 --port 18465 --name inkling \
  --context 16384 --max-tokens 4096 --vision
```

GET `/health` or `/v1/models`; POST `/v1/chat/completions`. One resident engine
serves one request at a time, in arrival order; a long request holds the server
until it finishes or its client hangs up.

**Where it listens.** On loopback the server is open and refuses `Host`
headers that name another machine. Any other `--host` (an IPv4 address, or
`0.0.0.0`) requires `TENSORFOLD_API_KEY` in the environment, and every route
except `/health` then needs `Authorization: Bearer <key>`. A reverse proxy
running in a container reaches the host through a non-loopback address, so it
needs this mode.

**Conversations** are rendered with the checkpoint's own chat template:
`system` (or `developer`), `user` and `assistant` turns, text parts, an earlier
turn's `reasoning_content`, and one image. `chat_template_test.cpp` checks the
rendering token for token against
`tools/inkling2_mini/chat-template-reference.json`, which
`make_chat_template_reference.py` records from the pinned Transformers
tokenizer. `reasoning_effort` takes `none`, `minimal`, `low`, `medium`, `high`,
`xhigh`, `max` or a number from 0 to 0.99. The server's default is `none`; the
template's own default is `high`, which spends far more tokens thinking.

**Responses.** `content` is the answer and `reasoning_content` the thinking.
`"stream": true` returns server-sent events, with a usage event when
`stream_options.include_usage` is set. `usage` is always in a plain reply.
`max_tokens` above the server's cap or the remaining context is clamped.

**Greedy only.** `temperature`, `top_p`, `seed` and other sampling options are
accepted and ignored, because ordinary clients always send them. Requests that
cannot be honoured are refused with 400: `tools` and tool messages, `stop`
sequences, JSON `response_format`, `n` above 1, a second image, and images that
are not inline PNG.

One optional image goes in a user message's content array:

```json
[{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}},
 {"type":"text","text":"What color is this image?"}]
```

Supply non-interlaced 8-bit PNG (gray, gray+alpha, RGB or RGBA), up to 256
merged visual tokens (`ceil(width/40) * ceil(height/40)`). The decoder does
not fetch remote URLs, accept JPEGs, or silently downsample oversized images.
Total context includes the template, visual tokens and requested output. The
limit is 16,384, **not** the checkpoint's advertised 1M. The JSON body cap is
8 MiB, including base64; message text is capped at 256 KiB.

## Speed, and what bounds it

Measured warm on a Radeon 8060S (gfx1151) through the loopback server, real
English text, one request at a time:

| Prompt tokens | Prefill | Rate |
|---|---|---|
| 151 | 1.7 s | 91 tok/s |
| 423 | 3.3 s | 130 tok/s |
| 1,416 | 7.8 s | 182 tok/s |
| 4,987 | 23.1 s | 216 tok/s |
| 9,590 | 43.8 s | 219 tok/s |

Decode is 17 tok/s at short context and 14 tok/s at 9.6K positions. The last
two prefill rows used `TF_INKLING_PREFILL_CHUNK=16384`; the default of 4096 is
about 10% slower beyond 4K tokens and needs a third of the scratch memory.

Every token passes 48 layer visits (the 12 middle layers run three times) and
eight experts per visit: about 4.3 billion weight multiplies per prompt token,
and 4.8 billion per generated token with the unembedding.

* **Prefill is bound by GEMM throughput.** Projections, expert rows and
  attention scores are hipBLAS SGEMMs. `rocBLAS` peaks near 2.7 TFLOP/s on
  this GPU and a token costs 8.7 GFLOP, a ceiling near 310 tok/s. Expert
  weights are dequantized to FP32 scratch once per chunk and layer visit,
  which is why short prompts are slower per token.
* **Decode is bound by memory traffic.** One token must read about 5.5 GB of
  weights: BF16 attention and unembedding matrices and packed experts, all in
  checkpoint form. The unembedding streams at roughly 175 GB/s, which would
  allow about 30 tok/s if every kernel streamed that well.

`TF_INKLING_TIMINGS=1` logs prefill and decode time per request (prefill
includes lazy weight uploads on a cold engine). `TF_INKLING_PROFILE=1` splits
prefill into attention, experts and the rest; it synchronizes the device at
stage boundaries, so a profiled run is slower.

## Numerical behaviour

`TF_INKLING_PREFILL_CHUNK=0` evaluates the prompt one token at a time with the
decode kernels. That path reproduces both traces in
`tools/inkling2_mini/gfx1151-reference.json` token for token and is the
oracle for the batched path.

The batched prefill keeps the same rounding points but its GEMMs sum in a
different order, so it is **not** bit-identical. This model rounds to BF16
after every operator and amplifies such differences: summing the scalar
matvec in 128 lanes instead of 256, a mathematically identical change, moves
the post-prefill logits by an RMS of 1.7 against a logit RMS of 17.9. The
batched path moves them by 1.5. Greedy output can therefore differ from the
reference traces wherever two tokens are nearly tied; one build of this path
diverged at the sixth generated token of the first trace and still answered
`4`. Decode attention sums cached keys in segments of 256, which is exact up
to 256 positions and reassociated beyond.

Checks, all on a real device with the original checkpoint:

```sh
hipcc -std=c++17 -O2 --offload-arch=gfx1151 batch_test.hip -lhipblas -o batch-test
./batch-test /path/to/Inkling-2-Mini     # batched operators vs scalar kernels
hipcc -std=c++17 -O2 --offload-arch=gfx1151 context_test.hip -o context-test
./context-test /path/to/Inkling-2-Mini   # cache windows, log scaling, admission
c++ -std=c++17 serve_test.cpp -o serve-test && ./serve-test   # request parsing, no GPU
c++ -std=c++17 chat_template_test.cpp -lpcre2-8 -o chat-template-test
./chat-template-test /path/to/Inkling-2-Mini ../../../../tools/inkling2_mini/chat-template-reference.json
python tools/inkling2_mini/serve_smoke.py --port 18465        # image and text answers
python tools/inkling2_mini/serve_chat_check.py --port 18465   # history, streaming, refusals
python tools/inkling2_mini/bench_prefill.py --port 18465 --document some.txt --max-tokens 1
```

The three Python checks read `TENSORFOLD_API_KEY` or take `--api-key`.

`batch-test` requires each batched operator to agree with its scalar kernel to
BF16 rounding noise (relative RMS under 2e-3) and scalar decode to continue
from the caches a batched prefill leaves behind.
