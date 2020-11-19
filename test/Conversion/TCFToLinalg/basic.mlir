// RUN: npcomp-opt <%s -convert-tcf-to-linalg | FileCheck %s --dump-input=fail

// CHECK-LABEL:   func @tcf_matmul(
// CHECK-SAME:                     %[[LHS:.*]]: tensor<?x?xf32>,
// CHECK-SAME:                     %[[RHS:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:           %[[C0F32:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           %[[C1:.*]] = constant 1 : index
// CHECK:           %[[LHSK:.*]] = dim %[[LHS]], %[[C1]] : tensor<?x?xf32>
// CHECK:           %[[RHSK:.*]] = dim %[[RHS]], %[[C0]] : tensor<?x?xf32>
// CHECK:           %[[KEQUAL:.*]] = cmpi "eq", %[[LHSK]], %[[RHSK]] : index
// CHECK:           %[[WINESS:.*]] = shape.cstr_require %[[KEQUAL]], "mismatching contracting dimension for matmul"
// CHECK:           %[[RET:.*]] = shape.assuming %[[WINESS]] -> (tensor<?x?xf32>) {
// CHECK:             %[[LHSROWS:.*]] = dim %[[LHS]], %[[C0]] : tensor<?x?xf32>
// CHECK:             %[[RHSCOLS:.*]] = dim %[[RHS]], %[[C1]] : tensor<?x?xf32>
// CHECK:             %[[SHAPE:.*]] = tensor_from_elements %[[LHSROWS]], %[[RHSCOLS]] : tensor<2xindex>
// CHECK:             %[[INIT_TENSOR:.*]] = tcp.splatted %[[C0F32]], %[[SHAPE]] : (f32, tensor<2xindex>) -> tensor<?x?xf32>
// CHECK:             %[[MATMUL:.*]] = linalg.matmul ins(%[[LHS]], %[[RHS]] : tensor<?x?xf32>, tensor<?x?xf32>) init(%[[INIT_TENSOR]] : tensor<?x?xf32>)  -> tensor<?x?xf32>
// CHECK:             shape.assuming_yield %[[MATMUL]] : tensor<?x?xf32>
// CHECK:           }
// CHECK:           return %[[RET:.*]] : tensor<?x?xf32>
func @tcf_matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcf.matmul %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL:   func @tcf_conv_2d(
// CHECK-SAME:                     %[[IN:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
// CHECK-SAME:                     %[[FILTER:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
// CHECK-SAME:                     %[[BIAS:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
// CHECK:           %[[C0F32:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           %[[C1:.*]] = constant 1 : index
// CHECK:           %[[INK:.*]] = dim %[[IN]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:           %[[FILTERK:.*]] = dim %[[FILTER]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:           %[[KEQUAL:.*]] = cmpi "eq", %[[INK]], %[[FILTERK]] : index
// CHECK:           %[[WINESS:.*]] = shape.cstr_require %[[KEQUAL]], "mismatching contracting dimension for conv_2d"
// CHECK:           %[[RET:.*]] = shape.assuming %[[WINESS]] -> (tensor<?x?x?x?xf32>) {
// CHECK:             %[[INROWS:.*]] = dim %[[IN]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:             %[[FILTERCOLS:.*]] = dim %[[FILTER]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:             %[[SHAPE:.*]] = tensor_from_elements %[[INROWS]], %[[FILTERCOLS]] : tensor<2xindex>
// CHECK:             %[[INIT_TENSOR:.*]] = tcp.splatted %[[C0F32]], %[[SHAPE]] : (f32, tensor<2xindex>) -> tensor<?x?x?x?xf32>
// CHECK:             %[[CONV2D:.*]] = linalg.conv_2d_nchw ins(%[[IN]], %[[FILTER]] : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) init(%[[INIT_TENSOR]] : tensor<?x?x?x?xf32>)  -> tensor<?x?x?x?xf32>
// CHECK:             %[[CONV2DBIAS:.*]] = addf %[[CONV2D]], %[[BIAS]] : tensor<?x?x?x?xf32>
// CHECK:             shape.assuming_yield %[[CONV2DBIAS]] : tensor<?x?x?x?xf32>
// CHECK:           }
// CHECK:           return %[[RET:.*]] : tensor<?x?x?x?xf32>
func @tcf_conv_2d(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = tcf.conv_2d %arg0, %arg1, %arg2 : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
