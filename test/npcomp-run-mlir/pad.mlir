// RUN: npcomp-opt -tcp-bufferize %s | npcomp-run-mlir %s \
// RUN:   -invoke splatted \
// RUN:   -arg-value="0.0 : f32" \
// RUN:   -arg-value="dense<[2, 2]> : tensor<2xindex>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

//   2x1     1x2       2x2
//  [ 1] + [3, 4] == [ 4,  5]
//  [10]          == [13, 14]

// CHECK: output #0: dense<0.000000e+00> : tensor<2x2xf32>
func @splatted(%arg0: f32, %arg1: tensor<?xindex>) -> tensor<?x?xf32> {
  %0 = tcp.splatted %arg0, %arg1 : (f32, tensor<?xindex>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
