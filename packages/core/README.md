# RIFE-SuperBoost

VapourSynth RIFE interpolation with optional backends and individually installed
models. An existing working VapourSynth environment is required.

```console
python -m pip install --find-links dist --extra-index-url https://pypi.nvidia.com "RIFE-SuperBoost[tensorrt,model-425]==0.1.0.dev9"
```

This development snapshot is built locally; it has not been uploaded to PyPI.
Use the corresponding local wheels until a release is published.

```python
from rife_superboost import interpolate

# rgb is an existing constant-format VapourSynth RGBH or RGBS VideoNode.
output = interpolate(rgb, model="4.25", backend="tensorrt", factor=2)
output.set_output()
```

The core requires `VapourSynth>=72`, uses its public VideoNode API, and does not
load GPU libraries during import. There is no `vs` extra. Model extras are
`model-425`, `model-425-lite`, `model-426`, `model-426-large`,
`model-425-scale8`, and `model-426-scale8`.
All models contain pre-generated FP16/FP32 optimized graphs and upstream MIT
notices. Runtime graph rewriting and automatic model downloads are not used.

The optional `4.25-scale8` and `4.26-scale8` models use unchanged base weights
with stage scale denominators 8/8/4/2/1, alignment 128 with zero padding, and
FP32 geometry. Select them with `model="4.25-scale8"` or
`model="4.26-scale8"`. Their model IDs and engine caches are separate from
the standard `4.25` and `4.26` profiles.

FP16 uses FP32 residual accumulation and sampling/geometry protection; it is not
claimed to match FP32 perceptually. YUV conversion is reserved and raises
`NotImplementedError`. Dev6 includes real-frame numerical checks and a ten-minute
paced VS pipeline run. Human viewing, native HDR and display/player acceptance
remain pending. Project code is MIT; NVIDIA libraries retain their own license.
