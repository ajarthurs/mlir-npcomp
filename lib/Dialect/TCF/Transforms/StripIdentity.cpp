#include "PassDetail.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "npcomp/Dialect/TCF/IR/TCFDialect.h"
#include "npcomp/Dialect/TCF/IR/TCFOps.h"
#include "npcomp/Dialect/TCF/Transforms/Passes.h"

namespace {

struct StripIdentityPattern : public mlir::OpRewritePattern<mlir::NPCOMP::tcf::IdentityOp> {
  StripIdentityPattern(mlir::MLIRContext *context)
      : OpRewritePattern<mlir::NPCOMP::tcf::IdentityOp>(context, /*benefit=*/1) {
    llvm::dbgs() << "StripIdentityPattern constructed" << "\n";
  }

  void rewrite(mlir::NPCOMP::tcf::IdentityOp op, mlir::PatternRewriter &rewriter) const override {
    llvm::dbgs() << "StripIdentityPattern.rewrite() start" << "\n";
    rewriter.eraseOp(op);
  }
};

class StripIdentityPass : public mlir::NPCOMP::tcf::TCFStripIdentityBase<StripIdentityPass> {
  void runOnOperation() override {
    llvm::dbgs() << "StripIdentityPass.runOnOperation() start" << "\n";
    StripIdentityPattern sip(&getContext());

    // Register rewrite patterns.
    mlir::OwningRewritePatternList patterns;
    patterns.insert<StripIdentityPattern>(sip);

    // Restrict dialects and their components affected by this pass.
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::NPCOMP::tcf::TCFDialect>();
    target.addLegalOp<mlir::FuncOp>();

    // Apply the rewrite patterns.
    auto func = getOperation();
    if(failed(applyPartialConversion(func, target, patterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
mlir::NPCOMP::tcf::createStripIdentityPass() {
  return std::make_unique<StripIdentityPass>();
}
