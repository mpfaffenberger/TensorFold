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

## Resident loopback server

```sh
zig build -Dhip=true native-hip tf-inkling-hip
zig-out/native-hip/bin/tensorfold-native serve /path/to/Inkling-2-Mini \
  --backend rocm --host 127.0.0.1 --port 18465 --name inkling \
  --context 16384 --max-tokens 512 --vision
```

GET `/health` or `/v1/models`; POST `/v1/chat/completions`. One resident engine
handles one single-user, non-streaming greedy request at a time. `content` is
the final answer and `reasoning_content` is separate. One optional image comes
first in the message content array, then text:

```json
[{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}},
 {"type":"text","text":"What color is this image?"}]
```

Supply non-interlaced 8-bit PNG (gray, gray+alpha, RGB or RGBA), up to 256
merged visual tokens (`ceil(width/40) * ceil(height/40)`). The decoder does
not fetch remote URLs, accept JPEGs, or silently downsample oversized images.
Total context includes the template, visual tokens and requested output. The
limit is 16,384, **not** the checkpoint's advertised 1M. Prefill processes 32-token
chunks through all 48 visits and projects only the final prompt position to logits.
Generation after the first token still uses the scalar decoder. Set
`TF_INKLING_TIMINGS=1` on the server process to log prefill time (including lazy
weight uploads on a cold engine) separately from subsequent-token decode time. The JSON body cap is 8 MiB, including base64; text is capped
at 32 KiB. Long requests occupy the server until completion.
