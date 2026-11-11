# Face landmark model (TFLite -> ONNX)

`assets/face_landmarks.onnx` is MediaPipe FaceMesh-V2 (`face_landmarks_detector.tflite`
from the official Face Landmarker bundle) converted for Windows ML (ORT 1.17: IR <= 9, opset <= 20).

- Source: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
- `.task` size 3,758,596 bytes; MD5 `b0e7274907a1644404fef66b28dd6d85`;
  SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`
- `face_landmarks_detector.tflite` (inside the zip) SHA-256
  `c7d54204ce0448474c7f3fa9af494787c0965cbdd6f20fc72867e43046bd43d5`
- `assets/face_landmarks.onnx` SHA-256 `014570104b501f29e44bfbcc98ed1f3c43bc491b7c0e18f3d964983ae2ca820a`
  (opset 13, IR 8, float32 weights embedded)

ONNX I/O: input `image` float32 [1,3,256,256] (NCHW, RGB, 0..1); outputs `landmarks`
[1,1434] (478 x xyz, pixel units of the 256x256 input) and `face_flag` [1,1] (raw logit, no sigmoid).

## Reproduce

Run from any directory other than the extracted/downloaded ones; use `-I` (isolated mode).

```
uv venv --python 3.13 <venv>
uv pip install --python <venv>/Scripts/python.exe --exclude-newer 2026-11-08T00:00:00Z -r tools/model/requirements.txt
curl -L -o face_landmarker.task <URL above>        # verify the MD5/SHA-256 above
<venv>/Scripts/python.exe -I -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extract('face_landmarks_detector.tflite', sys.argv[2])" face_landmarker.task <extract_dir>
<venv>/Scripts/python.exe -I tools/model/convert_face_landmarks.py <extract_dir>/face_landmarks_detector.tflite assets/face_landmarks.onnx
<venv>/Scripts/python.exe -I tools/model/validate_face_landmarks.py <extract_dir>/face_landmarks_detector.tflite assets/face_landmarks.onnx assets/reference
```

`validate_face_landmarks.py` compares ONNX (onnxruntime CPU) with TFLite (ai-edge-litert) on two
deterministic images (`lcg`, `ramp`), rewrites `assets/reference/{lcg,ramp}_output.bin`
(TFLite outputs: 1434 landmark floats + 1 face_flag float, little-endian float32) and exits non-zero
if max abs diff exceeds 1e-2 (landmarks) or 1e-3 (face_flag).
