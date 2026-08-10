# Third-party notices

Siliang Engine additions are licensed under
[`licenses/SILIANG-ENGINE-MIT.txt`](licenses/SILIANG-ENGINE-MIT.txt). The
upstream source at the repository root remains under [`LICENSE`](LICENSE), and
bundled components retain their own copyright notices and license terms.

## llama.cpp and ggml

- Project: [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp)
- Upstream base: `07132750825a4f2d27a547cd9cdde1c6f6001885`
  (`b10270`)
- License: MIT
- Copyright: 2023-2026 The ggml authors
- Full text: [`LICENSE`](LICENSE)

The provenance of the Siliang delta on this fork is recorded in
[`docs/PROVENANCE.md`](docs/PROVENANCE.md).

## Windows release runtime

The Windows release archives include the OpenSSL 3 runtime DLLs selected by
the tagged CMake build so that `llama-server` TLS support is self-contained.
OpenSSL 3 is licensed under Apache License 2.0. Each release archive includes
the exact installed OpenSSL license text as `licenses/OPENSSL-Apache-2.0.txt`
and records the runtime version in `provenance/BUILD-INFO.txt`. OpenSSL source
and notices are available from the [OpenSSL project](https://www.openssl.org/source/).

## Bundled components with separate notices

The upstream tree includes, among other components:

- nlohmann/json, MIT, copyright 2013-2025 Niels Lohmann:
  [`licenses/LICENSE-jsonhpp`](licenses/LICENSE-jsonhpp)
- cpp-httplib, MIT, copyright 2017 yhirose:
  [`vendor/cpp-httplib/LICENSE`](vendor/cpp-httplib/LICENSE)
- `gguf-py`, MIT, copyright 2023 Georgi Gerganov:
  [`gguf-py/LICENSE`](gguf-py/LICENSE)
- nerdamer-prime, MIT, copyright 2023 together-science:
  [`tools/ui/src/lib/vendors/nerdamer-prime/LICENSE`](tools/ui/src/lib/vendors/nerdamer-prime/LICENSE)
- decimal.js, MIT, copyright 2025 Michael Mclaughlin:
  [`tools/ui/src/lib/vendors/decimal.js/LICENCE.md`](tools/ui/src/lib/vendors/decimal.js/LICENCE.md)
- big-integer, dedicated to the public domain under the terms in:
  [`tools/ui/src/lib/vendors/big-integer/LICENSE`](tools/ui/src/lib/vendors/big-integer/LICENSE)
- miniaudio, choice of public-domain dedication or the embedded "MIT No
  Attribution" terms, copyright 2026 David Reid; the alternatives are at the
  end of:
  [`vendor/miniaudio/miniaudio.h`](vendor/miniaudio/miniaudio.h)
- stb_image, choice of MIT or public-domain dedication, copyright 2017 Sean
  Barrett; the alternatives are embedded at the end of:
  [`vendor/stb/stb_image.h`](vendor/stb/stb_image.h)
- sheredom/subprocess.h, public-domain dedication under the Unlicense terms
  embedded in:
  [`vendor/sheredom/subprocess.h`](vendor/sheredom/subprocess.h)

## Apache, LLVM-exception, and BSD components

The complete terms used by the Apache-licensed families identified below are
bundled here:

- [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt), copied verbatim from
  the [Apache Software Foundation](https://www.apache.org/licenses/LICENSE-2.0.txt);
- [`licenses/LLVM-exception.txt`](licenses/LLVM-exception.txt), copied verbatim
  from the [SPDX License List v3.27](https://raw.githubusercontent.com/spdx/license-list-data/v3.27.0/text/LLVM-exception.txt).
  The identifier `Apache-2.0 WITH LLVM-exception` means the Apache 2.0 terms
  together with this exception; LLVM publishes the combined terms in its
  official [`LICENSE.TXT`](https://github.com/llvm/llvm-project/blob/main/LICENSE.TXT).

Known affected families at the recorded upstream base include:

- portions of the SYCL backend derived from the LLVM Project are licensed
  `Apache-2.0 WITH LLVM-exception`; the files also carry Intel MIT notices where
  applicable. See the SPDX header in
  [`ggml/src/ggml-sycl/common.hpp`](ggml/src/ggml-sycl/common.hpp)
  and preserve the individual headers throughout
  [`ggml/src/ggml-sycl/`](ggml/src/ggml-sycl/);
- the C++11 compatibility implementation embedded in nlohmann/json is derived
  from Google Abseil and identified there as Apache License 2.0:
  [`vendor/nlohmann/json.hpp`](vendor/nlohmann/json.hpp);
- the bundled OpenVINO frontend adapter headers are `Apache-2.0`, copyright
  Intel Corporation:
  [`ggml/src/ggml-openvino/openvino/frontend.h`](ggml/src/ggml-openvino/openvino/frontend.h)
  and
  [`ggml/src/ggml-openvino/openvino/rt_info/weightless_caching_attributes.hpp`](ggml/src/ggml-openvino/openvino/rt_info/weightless_caching_attributes.hpp);
- the legacy MiniCPM-V image-encoder converter carries an Apache License 2.0
  header, copyright 2024 Google AI and The HuggingFace Team:
  [`tools/mtmd/legacy-models/minicpmv-convert-image-encoder-to-gguf.py`](tools/mtmd/legacy-models/minicpmv-convert-image-encoder-to-gguf.py);
- the SmolLM3 and Devstral chat templates carry Apache License 2.0 notices from
  the Unsloth team:
  [`models/templates/HuggingFaceTB-SmolLM3-3B.jinja`](models/templates/HuggingFaceTB-SmolLM3-3B.jinja)
  and
  [`models/templates/unsloth-mistral-Devstral-Small-2507.jinja`](models/templates/unsloth-mistral-Devstral-Small-2507.jinja);
- the Android example's Gradle startup script carries an Apache License 2.0
  header, copyright 2015 its original authors:
  [`examples/llama.android/gradlew`](examples/llama.android/gradlew);
- xxHash, bundled for the `gguf-hash` example, is BSD-2-Clause, copyright
  2012-2023 Yann Collet; the complete notice is embedded in:
  [`examples/gguf-hash/deps/xxhash/xxhash.h`](examples/gguf-hash/deps/xxhash/xxhash.h).

Source-file headers and license files in the repository are authoritative for
their respective components. Redistribution must preserve those notices. This
inventory is informational, is not necessarily exhaustive, is not legal advice,
and does not replace or narrow any bundled license text.
