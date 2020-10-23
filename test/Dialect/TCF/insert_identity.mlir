// RUN: npcomp-opt --tcf-insert-identity <%s | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @binary_elementwise
func @binary_elementwise(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) {
  // CHECK: tcf.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  // CHECK: tcf.identity %{{.*}} : tensor<?xf32>
  // CHECK: tcf.max %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  // CHECK: tcf.exp %arg0 : tensor<?xf32>
  %0 = tcf.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1 = tcf.max %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %2 = tcf.exp %arg0 : tensor<?xf32>
  return
}

func @serial_binary_elementwise(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) {
  // CHECK: tcf.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  // CHECK: tcf.identity %0 : tensor<?xf32>
  // CHECK: tcf.identity %0 : tensor<?xf32>
  // CHECK: tcf.add %2, %arg0 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  // CHECK: tcf.identity %3 : tensor<?xf32>
  // CHECK: tcf.identity %3 : tensor<?xf32>
  // CHECK: tcf.add %5, %2 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  // CHECK: tcf.identity %6 : tensor<?xf32>
  // CHECK: tcf.max %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  // CHECK: tcf.exp %arg0 : tensor<?xf32>

  %0 = tcf.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %id_0 = tcf.identity %0 : tensor<?xf32>
  %1 = tcf.add %id_0, %arg0 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %id_1 = tcf.identity %1 : tensor<?xf32>
  %2 = tcf.add %id_1, %id_0 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %3 = tcf.max %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %4 = tcf.exp %arg0 : tensor<?xf32>
  return
}
