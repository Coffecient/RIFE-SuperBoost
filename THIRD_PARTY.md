# Third-party notices

## dev7 completion of M3

`tools/optimize/tensorrt_network.py` derives its neural modules from the audited
Practical-RIFE reference and retains the original MIT attribution. The new
`native/fusion_plugin.cpp` and `native/fusion.cu` independently express sparse
sampling, FP32 flow updates, state resizing, HWC8 packing and blending using
NVIDIA's public IPluginV3/CUDA interfaces. The workspace's independently written
native-lab kernels were reviewed for sampling/layout design; no community
exporter, vs-mlrt implementation or legacy ONNX is used to generate these assets.
The standalone Win32 progress observer is project code using system APIs.

Legacy ONNX/plans are read-only diagnostic references in the player investigation.
The user video, decoded frames and derived comparisons remain private evaluation
assets and are excluded from wheels and source distributions. Their rights are
not changed by the project's MIT license. NVIDIA/CUDA notices below still apply.

## Practical-RIFE

`tools/export/reference.py` adapts the inference network and warp implementation
from [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE), including the
network definitions shipped with the four official model archives listed in
`models/manifest.json`. Copyright (c) 2021 hzwer. Licensed under MIT; the complete
notice is retained in [models/licenses/practical-rife/LICENSE](models/licenses/practical-rife/LICENSE).

The adaptation removes training helpers and the process-global warp-grid cache,
accepts two padded RGB tensors and a runtime scalar time, selects the official
five-stage scales explicitly, and constructs the same sampling grid dynamically.
Original weights, residual beta parameters, channel order, stage widths, and
sampling rules are retained. No project optimization pass is applied to M1 graphs.

The original files under `models/assets/` are pinned by size and SHA-256. The
development-only oracle executes verified network/warp definitions and the
original `Model.inference` method in private namespaces without training imports.
No upstream training or video-entry script is run as a program.

Official checkpoints and generated reference ONNX retain the original model's MIT
terms. Their metadata and provenance are recorded separately from the new code's
copyright notice. Distribution must include the upstream MIT notice.

## Development dependencies

PyTorch, ONNX, ONNX Runtime, NumPy and their dependencies are installed into the
ignored development environment. Their own licenses apply. They are not vendored
into this source tree or a playback wheel at M1. The exact local environment is
recorded alongside validation results and in `tools/export/requirements.lock.txt`.

The HolyWu license snapshot is provenance research material. M1 does not copy its
implementation. No vs-mlrt source, exported graph, or runtime is used by the M1
export path.

## M2 native runtime and packages

The new core, backend, builder, CUDA transfer kernels, C ABI and build/validation
tools are project code under MIT. Utility and lifecycle patterns reviewed from
this workspace's independently written `native-lab` are identified in
`docs/evidence/m2-source-review.json`; that review does not relicense the whole lab.
No vs-mlrt/vstrt, community TensorRT backend, or VS implementation source is copied
or linked into the native binaries. The Python core uses VapourSynth's public API.

TensorRT 10.16.1.11 headers/import libraries and external NVIDIA runtime DLLs come
from the user-supplied SDK archive pinned in
`docs/evidence/tensorrt-10.16.1.11-archive.json`. Its runtime binaries are not included
in the M2 backend wheel; the selected directory and exact loaded identities are
recorded by the backend. NVIDIA SDK and runtime terms apply independently of MIT.

CUDA Toolkit 13.2.86 is an external development toolchain. Its static CUDA runtime
is linked into the two native binaries, avoiding a separately searched cudart DLL.
The backend wheel retains the supplied CUDA license/EULA in
`rife_superboost_tensorrt/licenses/`. These NVIDIA components do not acquire the
project's MIT license. Final dependency/redistribution packaging is a M5 decision.

MSVC 19.44, Windows SDK, CMake and NMake are development tools; no Microsoft runtime
installer or SDK is vendored. The native binaries use the static MSVC runtime.
Python/VS/NumPy/ONNX Runtime used by verification retain their respective licenses;
NumPy and ONNX Runtime are not playback dependencies of the new packages.

## M3 optimized graphs and cache

`tools/optimize/network.py` derives its network from the strict M1 reference and
therefore retains the Practical-RIFE MIT attribution. The new sparse-warp lattice,
paired sampling, dead-state selection, coefficient rearrangement and low-resolution
state paths are project transformations, independently expressed over the official
profile rather than copied from a community exporter or applied to its ONNX.
The phase mapping and low-resolution-state conditions were also reviewed against
this workspace's own `investigation/repack_lowres_probe.py`; no old graph asset,
GPL source or old deployment module is imported. Native caching is a fresh bounded
implementation over M2's C ABI, with lifecycle/cache patterns reviewed from the
independent `native-lab`. It does not copy its custom plugin implementations.

Generated full and split ONNX assets remain derivatives of the official MIT model.
Keep the Practical-RIFE copyright and license alongside them when distributing.
M3 wheels contain only project runtime/builder code and the already documented
CUDA static runtime notices; TensorRT DLL distribution remains a M5 decision.

## M4 precision assets and public API

`tools/optimize/export_mixed.py` and the mixed paths in `network.py` derive FP16
convolution weights from the same strict official checkpoint parameters. Learned
residual beta parameters retain their original FP32 values; residual accumulation,
geometry, sampling and blend remain FP32. New precision bundles copy the original
upstream MIT notice and provenance. No new upstream source or community model
asset is introduced. Public CFR/property handling, the reserved color stub, and
the native per-request bypass are project code under MIT.

M3's canonical version is corrected to `0.1.0.dev3` by metadata-only wheel
repackaging with regenerated RECORDs and identical native payloads. Original
observed artifacts/evidence are retained separately; this is not a relicensing
or modification of NVIDIA binaries. M4 uses `0.1.0.dev4` and the same independent
NVIDIA/CUDA redistribution boundary documented above.

## M5 distribution

Each independent model distribution contains the original Practical-RIFE MIT
notice, project MIT notice, checkpoint/source provenance and six pre-generated
graphs. Packaging uses standard ONNX external_data to share identical weight
bytes. All tensor bytes and remaining protobuf fields reconstruct exactly, and
all corresponding CPU outputs match bit-for-bit. No additional network source,
community graph, GPL exporter or inference implementation is introduced.

The TensorRT backend depends on NVIDIA's unmodified
`tensorrt-cu13-libs==10.16.1.11` Windows wheel. It does not redistribute TensorRT
DLLs itself. The exact license from that wheel, rather than a different `latest`
web page, is retained in [the license snapshot](docs/licenses/TensorRT-10.16.1.11-LICENSE.txt).
NVIDIA's library wheel and the user SDK have distinct binary hashes; M5 validates
the official dependency separately and keeps the actual identities in cache keys.

The backend still statically links CUDA runtime. Its complete CUDA notices and
an explicit NOTICE are included both as package data and wheel license files;
the wheel license expression is `MIT AND LicenseRef-NVIDIA-CUDA`. This does not
change the MIT license of project source. Static MSVC runtime redistribution
continues under its original terms. No SDK headers, import libraries, compiler,
`trtexec`, TensorRT Python bindings, or NVIDIA runtime DLLs are vendored in a
project wheel. The official dependency retains its own license and copyright.

The model providers, metadata-based runtime discovery, external weight identity,
standard build guards, child-process path handling and verification tools are new
project code. See [the source review](docs/evidence/m5-source-review.json) and
[distribution policy](docs/DISTRIBUTION.zh-CN.md). No upstream initializer code is
copied; discovery deliberately avoids executing it.
