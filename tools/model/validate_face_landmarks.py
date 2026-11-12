#!/usr/bin/env python3
"""Validate face_landmarks.onnx against the original TFLite model.

Runs two deterministic images through the TFLite interpreter (NHWC) and onnxruntime
(NCHW), prints max abs differences, writes the TFLite outputs as reference .bin files
(1434 landmark floats + 1 face_flag float, little-endian float32), and exits non-zero
if a target is missed.

Usage: validate_face_landmarks.py MODEL.tflite MODEL.onnx REF_OUT_DIR
"""
import argparse
import os
import sys

import numpy as np
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter

SIZE = 256
LANDMARK_TOL = 1e-2  # landmark values are in 0..256 pixel units
FLAG_TOL = 1e-3
TFLITE_LANDMARKS = "Identity"
TFLITE_FACE_FLAG = "Identity_1"


def make_lcg():
    state = 12345
    out = np.empty(SIZE * SIZE * 3, dtype=np.float32)
    for k in range(out.size):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        out[k] = np.float32((state >> 8) / 16777216.0)
    return out.reshape(SIZE, SIZE, 3)  # k = (y*256 + x)*3 + c


def make_ramp():
    y, x, c = np.meshgrid(np.arange(SIZE), np.arange(SIZE), np.arange(3), indexing="ij")
    return (((x * 3 + y * 5 + c * 17) % 256) / 255.0).astype(np.float32)


def run_tflite(path, img):
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], img[None])
    it.invoke()
    by_name = {d["name"]: it.get_tensor(d["index"]) for d in it.get_output_details()}
    return by_name[TFLITE_LANDMARKS].reshape(-1), by_name[TFLITE_FACE_FLAG].reshape(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tflite")
    ap.add_argument("onnx")
    ap.add_argument("ref_dir", help="directory to (re)write lcg_output.bin and ramp_output.bin")
    a = ap.parse_args()

    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    os.makedirs(a.ref_dir, exist_ok=True)
    ok = True
    for name, img in (("lcg", make_lcg()), ("ramp", make_ramp())):
        lm_t, ff_t = run_tflite(a.tflite, img)
        assert lm_t.size == 1434 and ff_t.size == 1
        nchw = np.ascontiguousarray(img.transpose(2, 0, 1)[None])
        lm_o, ff_o = sess.run(["landmarks", "face_flag"], {"image": nchw})
        d_lm = float(np.max(np.abs(lm_o.reshape(-1) - lm_t)))
        d_ff = float(np.max(np.abs(ff_o.reshape(-1) - ff_t)))
        print(f"{name}: max|d landmarks|={d_lm:.3e} (tol {LANDMARK_TOL})  "
              f"max|d face_flag|={d_ff:.3e} (tol {FLAG_TOL})  tflite face_flag={ff_t[0]:.6f}")
        ok &= d_lm <= LANDMARK_TOL and d_ff <= FLAG_TOL
        ref = np.concatenate([lm_t, ff_t]).astype("<f4")
        ref.tofile(os.path.join(a.ref_dir, f"{name}_output.bin"))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
