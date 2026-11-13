# RIFE-SuperBoost TensorRT backend

Optional Windows x64 backend for RIFE-SuperBoost. Uses TensorRT **10.16.1.11**,
CUDA 13.2 static runtime, and a native out-of-process engine builder. Neither
TensorRT Python bindings nor a local compiler is required to use an installed
wheel. The native DLL and builder are private to this package.

Development revision `0.1.0.dev7` adds prepared sampling/HWC8 fusions, direct
cache-slot bindings, precision-matched RGB/features and a bounded Graph LRU.
RGBH selects Half RGB storage and RGBS retains Float storage automatically;
geometry and protected residual arithmetic remain Float. Set `fused=False`
for the prepared portable graph control. No ONNX is rewritten during playback.

Cold builds display an independent native progress window by default, including
actual TensorRT phases, elapsed time and the log path. Closing this window does
not cancel the host. Valid cache hits open no window. Use
`TensorRT(build_feedback="log")` for headless developer jobs.

The pinned dependency `tensorrt-cu13-libs==10.16.1.11` is published by NVIDIA at
<https://pypi.nvidia.com>. Its Windows wheel is approximately 1.91 GB compressed;
allow additional space for installation and per-device engine caches. Normal
pip installation may retrieve it through NVIDIA's PyPI stub. Adding NVIDIA's
index selects its prebuilt wheel directly. No NVIDIA DLLs are repackaged under
the project's MIT license.

```python
from rife_superboost_tensorrt import TensorRT

backend = TensorRT(precision="fp16")  # Locate the installed NVIDIA library wheel.
# Alternatively, supply an exact compatible NVIDIA runtime directory explicitly.
backend_fp32 = TensorRT(runtime_dir=r"C:\NVIDIA\TensorRT-10.16.1.11\bin", precision="fp32")
```

GPU DLLs are loaded lazily by absolute path, without importing `tensorrt_libs`
or changing global PATH. Same-name incompatible loaded DLLs raise an explicit
conflict. Restart the host after updating native libraries. NVIDIA CUDA notices
for the statically linked runtime are included in `licenses/`.

Validated device: RTX 4060 Laptop, SM89. Other SM75+ targets are build candidates,
not validated support claims. YUV conversion and additional backends are not
implemented. This local development version has not been published to PyPI.
