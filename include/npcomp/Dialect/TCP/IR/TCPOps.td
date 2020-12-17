//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TCP_OPS
#define TCP_OPS

include "npcomp/Dialect/TCP/IR/TCPBase.td"
include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

class TCP_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<TCP_Dialect, mnemonic, traits> {
}

def TCP_BroadcastToOp : TCP_Op<"broadcast_to"> {
  let summary = "Broadcasts an operand to a given shape.";
  let description = [{
Broadcasts `operand` to the shape `shape`.

It is undefined behavior if such a broadcast is not legal.
  }];
  let arguments = (ins AnyRankedTensor:$operand, Shape_ExtentTensorType:$shape);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$operand `,` $shape attr-dict `:` functional-type(operands, results)";
}

def TCP_SplattedOp : TCP_Op<"splatted"> {
  let summary = "Creates a tensor filled with a particular scalar value.";
  let description = [{
    Creates a tensor of shape `shape` with all elements filled with `splatVal`.

    This op is somewhat redundant with tcp.broadcast_to. However,
    tcp.broadcast_to handles degenerate "size-1" broadcasting which structurally
    cannot happen with this op. So to avoid losing that information, we keep
    this op separate.

    NOTE: The name "splatted" separates it from std.splat, which currently
    only handles statically shaped memrefs.

    TODO: Improve std.splat to take dynamic shapes.
  }];
  let arguments = (ins AnyType:$splatVal, Shape_ExtentTensorType:$shape);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$splatVal `,` $shape attr-dict `:` functional-type(operands, results)";
}

def TCP_PadOp : TCP_Op<"pad"> {
  let summary = "Pads a tensor with zeros";
  let description = [{
    Pads a tensor with zeros along the borders of the height and width dimensions. This op is inspired by the padding feature in PyTorch's Conv2d layer (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html).

    The tensors have dimensions:
    - in:     [N, Cin, H, W]
    - expand: [4] (Ptop, Pbottom, Pleft, Pright)
    - result: [N, Cin, Hout, Wout]
  }];
  let arguments = (ins 4DTensorOf<[F32]>:$in, 1DTensorOf<[I32,I32,I32,I32]>:$expand);
  let results = (outs 4DTensorOf<[F32]>:$result);

  let assemblyFormat = "$in `,` $expand attr-dict `:` functional-type(operands, results)";
}

#endif // TCP_OPS
