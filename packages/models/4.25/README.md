# RIFE-SuperBoost model 4.25

Optional pre-generated optimized RIFE model derived from hzwer/Practical-RIFE.
Original model: MIT, Copyright (c) 2021 hzwer. Original notice, fixed source
URLs, checkpoint hashes and packaging provenance are included.

Contains four portable aligned encoder/synthesis ONNX graphs for FP16 and FP32,
plus eight prepared TensorRT fusion graphs for explicit neural/RGB storage layouts.
Both policies share FP32 base weights through standard ONNX external_data.
Explicit constant Casts in the prepared FP16 graph select mixed arithmetic;
no separate FP16 weight copy is distributed. Full graphs are development assets. The installed
assets are never optimized, split or rewritten by the playback runtime.

Select with `interpolate(rgb, model="4.25", backend="tensorrt")`.
Requires the core, a working VapourSynth environment, and a separately installed
backend. Official right/bottom zero padding uses alignment 128; no added clamp
or color conversion. FP16 perceptual equivalence is not claimed.

This is local development version `0.1.0.dev9`, not a PyPI publication.
