// RUN: npcomp-run-mlir %s \
// RUN:   -invoke pad \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

//   2x1     1x2       2x2
//  [ 1] + [3, 4] == [ 4,  5]
//  [10]          == [13, 14]

// CHECK: output #0: dense<0.000000e+00> : tensor<1x1x4x5xf32>
func @pad() -> tensor<?x?x?x?xf32> {
  %cst = constant 3.000000e+00 : f32
  %fill = constant 1.000000e+00 : f32
  %s = shape.const_shape [1, 1, 2, 3] : tensor<4xindex>
  %e = shape.const_shape [1, 1, 1, 1] : tensor<4xindex>
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %8 = tcp.splatted %cst, %s : (f32, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  %9 = tcp.pad %8, %e, %fill : (tensor<?x?x?x?xf32>, tensor<4xindex>, f32) -> tensor<?x?x?x?xf32>
  return %9 : tensor<?x?x?x?xf32>
}
