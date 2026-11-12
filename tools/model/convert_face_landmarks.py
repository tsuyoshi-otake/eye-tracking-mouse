#!/usr/bin/env python3
"""Convert MediaPipe FaceMesh-V2 `face_landmarks_detector.tflite` to ONNX.

Hand-written converter (tflite2onnx 0.4.1 crashes on this model in its layout
propagation pass, and the model only uses 9 simple op types).

Final ONNX contract:
  input   image      float32 [1,3,256,256]  NCHW, RGB, values in [0,1]
  output  landmarks  float32 [1,1434]       478 x (x,y,z), same flat order as the
                                            TFLite output "Identity"
  output  face_flag  float32 [1,1]          raw TFLite output "Identity_1"
                                            (a logit; no sigmoid applied)
TFLite output "Identity_2" (a sigmoid of a separate head) is dropped.

Usage: convert_face_landmarks.py MODEL.tflite OUT.onnx
"""
import argparse
import math

import numpy as np
import onnx
import tflite
from onnx import TensorProto, helper, numpy_helper

OPSET = 13
IR_VERSION = 8
LANDMARKS_TFLITE_OUTPUT = "Identity"
FACE_FLAG_TFLITE_OUTPUT = "Identity_1"

# tflite TensorType -> numpy dtype (only the ones present in this model).
TENSOR_DTYPES = {0: np.float32, 1: np.float16, 2: np.int32}
PADDING_SAME, PADDING_VALID = 0, 1
BUILTIN_NAMES = {v: k for k, v in vars(tflite.BuiltinOperator).items() if not k.startswith("_")}


class Converter:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.model = tflite.Model.GetRootAsModel(f.read(), 0)
        self.sg = self.model.Subgraphs(0)
        self.consts = {}  # tflite tensor idx -> numpy array (constants incl. dequantized)
        self.nodes = []  # (onnx node, [tflite output idx])
        self.inits = {}  # onnx name -> numpy array

    # -- helpers ---------------------------------------------------------
    def tname(self, i):
        return self.sg.Tensors(i).Name().decode()

    def vname(self, i):
        return f"t{i}"

    def shape(self, i):
        return [int(d) for d in self.sg.Tensors(i).ShapeAsNumpy()]

    def const_array(self, i):
        if i in self.consts:
            return self.consts[i]
        t = self.sg.Tensors(i)
        buf = self.model.Buffers(t.Buffer()).DataAsNumpy()
        if isinstance(buf, int) or len(buf) == 0:
            return None
        arr = np.frombuffer(bytes(buf), dtype=TENSOR_DTYPES[t.Type()]).reshape(self.shape(i))
        self.consts[i] = arr
        return arr

    def add_init(self, name, arr):
        self.inits[name] = np.ascontiguousarray(arr)
        return name

    def operand(self, i, transform=None):
        """Name of an ONNX value for tflite tensor i; constants become float32 initializers."""
        arr = self.const_array(i)
        if arr is None:
            return self.vname(i)
        if arr.dtype == np.float16:
            arr = arr.astype(np.float32)
        if transform is not None:
            arr = transform(arr)
        return self.add_init(f"c{i}", arr)

    @staticmethod
    def same_pads(in_size, k, s):
        out = math.ceil(in_size / s)
        total = max((out - 1) * s + k - in_size, 0)
        begin = total // 2  # TFLite puts the extra pixel at the end
        return begin, total - begin

    @staticmethod
    def opts(op, cls):
        o = cls()
        o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
        return o

    # -- conversion ------------------------------------------------------
    def convert(self):
        sg = self.sg
        assert sg.InputsLength() == 1
        inp = int(sg.InputsAsNumpy()[0])
        assert self.shape(inp) == [1, 256, 256, 3] and sg.Tensors(inp).Type() == 0, "unexpected input"
        outs = {self.tname(int(o)): int(o) for o in sg.OutputsAsNumpy()}
        for i in range(sg.OperatorsLength()):
            op = sg.Operators(i)
            oc = self.model.OperatorCodes(op.OpcodeIndex())
            code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
            kind = BUILTIN_NAMES[code]
            ins = [int(x) for x in op.InputsAsNumpy()]
            o = [int(x) for x in op.OutputsAsNumpy()]
            handler = getattr(self, "op_" + kind, None)
            if handler is None:
                raise NotImplementedError(f"unsupported TFLite op {kind}")
            handler(op, ins, o)
        return inp, outs

    def op_DEQUANTIZE(self, op, ins, outs):
        arr = self.const_array(ins[0])
        if arr is None:
            raise NotImplementedError("DEQUANTIZE of a non-constant tensor")
        # float16 weights widen to float32 exactly.
        self.consts[outs[0]] = arr.astype(np.float32)

    def op_CONV_2D(self, op, ins, outs):
        o = self.opts(op, tflite.Conv2DOptions)
        assert o.FusedActivationFunction() == 0 and o.DilationHFactor() == 1 and o.DilationWFactor() == 1
        x, w, b = ins
        _, h, wd, _ = self.shape(x)
        kh, kw = self.shape(w)[1:3]
        sh, sw = o.StrideH(), o.StrideW()
        pads = self.pads(o.Padding(), h, wd, kh, kw, sh, sw)
        node = helper.make_node(
            "Conv",
            [self.vname(x), self.operand(w, lambda a: a.transpose(0, 3, 1, 2)), self.operand(b)],
            [self.vname(outs[0])], kernel_shape=[kh, kw], strides=[sh, sw], pads=pads,
        )
        self.nodes.append((node, outs))

    def op_DEPTHWISE_CONV_2D(self, op, ins, outs):
        o = self.opts(op, tflite.DepthwiseConv2DOptions)
        assert o.FusedActivationFunction() == 0 and o.DilationHFactor() == 1 and o.DilationWFactor() == 1
        x, w, b = ins
        _, h, wd, cin = self.shape(x)
        _, kh, kw, cout = self.shape(w)
        assert cout == cin * o.DepthMultiplier()
        sh, sw = o.StrideH(), o.StrideW()
        pads = self.pads(o.Padding(), h, wd, kh, kw, sh, sw)
        node = helper.make_node(
            "Conv",
            [self.vname(x), self.operand(w, lambda a: a.transpose(3, 0, 1, 2)), self.operand(b)],
            [self.vname(outs[0])], kernel_shape=[kh, kw], strides=[sh, sw], pads=pads, group=cin,
        )
        self.nodes.append((node, outs))

    def pads(self, padding, h, w, kh, kw, sh, sw):
        if padding == PADDING_VALID:
            return [0, 0, 0, 0]
        assert padding == PADDING_SAME
        top, bottom = self.same_pads(h, kh, sh)
        left, right = self.same_pads(w, kw, sw)
        return [top, left, bottom, right]

    def op_PRELU(self, op, ins, outs):
        x, alpha = ins
        # TFLite alpha [1,1,C] (NHWC broadcast) -> ONNX NCHW slope [C,1,1].
        slope = self.operand(alpha, lambda a: a.reshape(-1, 1, 1))
        self.nodes.append((helper.make_node("PRelu", [self.vname(x), slope], [self.vname(outs[0])]), outs))

    def op_ADD(self, op, ins, outs):
        assert self.opts(op, tflite.AddOptions).FusedActivationFunction() == 0
        assert all(self.const_array(i) is None for i in ins), "constant ADD operand not supported"
        self.nodes.append((helper.make_node("Add", [self.vname(i) for i in ins], [self.vname(outs[0])]), outs))

    def op_MAX_POOL_2D(self, op, ins, outs):
        o = self.opts(op, tflite.Pool2DOptions)
        assert o.FusedActivationFunction() == 0
        _, h, w, _ = self.shape(ins[0])
        kh, kw, sh, sw = o.FilterHeight(), o.FilterWidth(), o.StrideH(), o.StrideW()
        pads = self.pads(o.Padding(), h, w, kh, kw, sh, sw)
        node = helper.make_node("MaxPool", [self.vname(ins[0])], [self.vname(outs[0])],
                                kernel_shape=[kh, kw], strides=[sh, sw], pads=pads)
        self.nodes.append((node, outs))

    def op_PAD(self, op, ins, outs):
        p = self.const_array(ins[1]).astype(np.int64)  # [4,2] in NHWC
        p = p[[0, 3, 1, 2]]  # -> NCHW rows
        pads = self.add_init(f"c{ins[1]}", np.concatenate([p[:, 0], p[:, 1]]))
        self.nodes.append((helper.make_node("Pad", [self.vname(ins[0]), pads], [self.vname(outs[0])], mode="constant"), outs))

    def op_LOGISTIC(self, op, ins, outs):
        self.nodes.append((helper.make_node("Sigmoid", [self.vname(ins[0])], [self.vname(outs[0])]), outs))

    def op_RESHAPE(self, op, ins, outs):
        # Only used on the [1,1,1,1] -> [1,1] sigmoid head, which is pruned later.
        shp = self.add_init(f"c{ins[1]}", self.const_array(ins[1]).astype(np.int64))
        self.nodes.append((helper.make_node("Reshape", [self.vname(ins[0]), shp], [self.vname(outs[0])]), outs))


def build(path):
    c = Converter(path)
    inp, outs = c.convert()
    lm, ff = outs[LANDMARKS_TFLITE_OUTPUT], outs[FACE_FLAG_TFLITE_OUTPUT]
    # Final heads: [1,N,1,1] (NCHW, spatial 1x1 so flat order equals the NHWC flat order).
    assert c.shape(lm) == [1, 1, 1, 1434] and c.shape(ff) == [1, 1, 1, 1]
    c.add_init("shape_lm", np.array([1, 1434], dtype=np.int64))
    c.add_init("shape_ff", np.array([1, 1], dtype=np.int64))
    nodes = [n for n, _ in c.nodes]
    nodes.append(helper.make_node("Reshape", [c.vname(lm), "shape_lm"], ["landmarks"]))
    nodes.append(helper.make_node("Reshape", [c.vname(ff), "shape_ff"], ["face_flag"]))
    # Rename the graph input and prune nodes not needed by the two kept outputs.
    for n in nodes:
        n.input[:] = ["image" if x == c.vname(inp) else x for x in n.input]
    needed, keep = {"landmarks", "face_flag"}, []
    for n in reversed(nodes):
        if any(o in needed for o in n.output):
            keep.append(n)
            needed.update(n.input)
    keep.reverse()
    used = {x for n in keep for x in n.input}
    inits = [numpy_helper.from_array(a, name=k) for k, a in c.inits.items() if k in used]
    for i, n in enumerate(keep):
        n.name = f"n{i}_{n.op_type}"
    graph = helper.make_graph(
        keep, "face_landmarks",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 256, 256])],
        [helper.make_tensor_value_info("landmarks", TensorProto.FLOAT, [1, 1434]),
         helper.make_tensor_value_info("face_flag", TensorProto.FLOAT, [1, 1])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              producer_name="convert_face_landmarks.py")
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model, full_check=True)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tflite", help="path to face_landmarks_detector.tflite")
    ap.add_argument("onnx", help="output .onnx path")
    a = ap.parse_args()
    model = build(a.tflite)
    onnx.save(model, a.onnx)
    ops = sorted({n.op_type for n in model.graph.node})
    print(f"wrote {a.onnx}: {len(model.graph.node)} nodes, ops={ops}, opset={OPSET}, ir_version={IR_VERSION}")


if __name__ == "__main__":
    main()
