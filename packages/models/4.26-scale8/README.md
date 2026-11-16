# RIFE-SuperBoost model 4.26-scale8

Optional tuned inference profile using the original RIFE 4.26 weights from
hzwer/Practical-RIFE (MIT, Copyright (c) 2021 hzwer).

The five stage scale denominators are **8/8/4/2/1**. This changes the first stage
from the standard profile's 16. Right/bottom zero padding to alignment 128 and
FP32 flow, sampling coordinates and residual accumulation are retained. This
is an inference configuration, with unchanged learned weights.

The package contains FP16/FP32 portable split graphs and TensorRT fusion graphs
for RGBH/RGBS storage, sharing one FP32 weight file. Graphs are prepared before
distribution. Playback only loads assets and builds a separate engine cache.
The standard `4.26` model can be installed alongside this profile.

```python
output = interpolate(rgb, model="4.26-scale8", backend="tensorrt", factor=2)
```

Install the corresponding local `rife-superboost-model-426-scale8` wheel,
or select the core extra `model-426-scale8`. Requires RIFE-SuperBoost
`0.1.0.dev9`, VapourSynth and a compatible inference backend.
The wheel includes upstream attribution, pinned checkpoint identity and the
explicit inference profile. Local development release; not uploaded to PyPI.
