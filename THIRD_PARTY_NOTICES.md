# Third-party notices

## Face landmark model (embedded in the executable)

- Name: MediaPipe Face Mesh V2 (`face_landmarks_detector.tflite` from the MediaPipe
  Face Landmarker bundle), by Geng Yan and Ivan Grishchenko, Google. Model date: 2022-09-15.
- License: Apache License, Version 2.0. Stated on the model card ("Licensed under
  Apache License, Version 2.0"):
  https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf
  The license text is in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).
- Source: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
  - `face_landmarker.task` SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`
  - `face_landmarks_detector.tflite` (inside it) SHA-256 `c7d54204ce0448474c7f3fa9af494787c0965cbdd6f20fc72867e43046bd43d5`
- File in this repository: `assets/face_landmarks.onnx`
  SHA-256 `014570104b501f29e44bfbcc98ed1f3c43bc491b7c0e18f3d964983ae2ca820a`
- Changes made to the original: converted from TensorFlow Lite to ONNX (opset 13) with
  `tools/model/convert_face_landmarks.py`. The float16 weights were widened to float32,
  the tensor layout changed from NHWC to NCHW, and the third output (a partial blendshape
  head) was removed. The network and its weights are otherwise unchanged; see
  `tools/model/README.md` for the reproduction steps and the numerical comparison.

## Rust crates (compiled into the executable)

All are licensed "MIT OR Apache-2.0" and are used here under the MIT License.

- `windows`, `windows-collections`, `windows-core`, `windows-future`, `windows-implement`,
  `windows-interface`, `windows-link`, `windows-numerics`, `windows-result`,
  `windows-strings`, `windows-threading` — Copyright (c) Microsoft Corporation.
  https://github.com/microsoft/windows-rs

Build-time only (procedural macros, not part of the executable): `proc-macro2`, `quote`,
`syn` (MIT OR Apache-2.0) and `unicode-ident` ((MIT OR Apache-2.0) AND Unicode-3.0).

### MIT License (windows-rs)

    MIT License

    Copyright (c) Microsoft Corporation.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE
