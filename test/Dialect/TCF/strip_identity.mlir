// RUN: npcomp-opt --tcf-strip-identity <%s | FileCheck %s --dump-input=fail

// CHECK-NOT: tcf.identity
func @binary_elementwise(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) {
  %0 = tcf.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1 = tcf.identity %0 : tensor<?xf32>
  %2 = tcf.max %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %3 = tcf.exp %arg0 : tensor<?xf32>
  return
}