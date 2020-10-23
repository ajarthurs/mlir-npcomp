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

  mlir::LogicalResult matchAndRewrite(mlir::NPCOMP::tcf::IdentityOp op, mlir::PatternRewriter &rewriter) const override {
    llvm::dbgs() << "StripIdentityPattern.rewrite() start" << "\n";
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class StripIdentityPass : public mlir::NPCOMP::tcf::TCFStripIdentityBase<StripIdentityPass> {
  void runOnOperation() override {
    llvm::dbgs() << "StripIdentityPass.runOnOperation() start" << "\n";

    // Register rewrite patterns.
    mlir::OwningRewritePatternList patterns;
    patterns.insert<StripIdentityPattern>(&getContext());

    // Restrict dialects and their components affected by this pass.
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::NPCOMP::tcf::TCFDialect>();
    target.addLegalOp<mlir::FuncOp>();
    target.addIllegalOp<mlir::NPCOMP::tcf::IdentityOp>();

    // Apply the rewrite patterns.
    // NOTE(bryce): There are two ways of applying a list of RewritePatterns to a graph
    // 1) Conversion: Usually thought of as going FROM one Dialect(s) / Op(s) TO another Dialect(s) / Op(s)
    //    * Conversion / Partial conversions only run on Op(s) / Dialect(s) that are marked Illegal
    //        * Iterate over all Illegal Op(s) and apply RewritePatterns, then assert after all patterns have 
    //          Been applied that there are no remaining Illegal Op(s) / Dialect(s). For this case, our Rewrite
    //          Patterns were not being applied at all because we were not marking any Op's as Illegal during the conversion.
    // 2) ApplyPatternsAndFoldGreedily: Continuously apply each rewrite pattern (in order of insertion) until no work is left to be done
    //     * This method more fits our use case here, since we are performing a Transform (which implies we are operating only on Ops of the same Dialect)
    // auto func = getOperation();
    // assert(mlir::isa<mlir::FuncOp>(func) && "func is not a FuncOp");
    // llvm::dbgs() << "Attempting to a apply a partial conversion" << "\n";
    // if(failed(applyPartialConversion(func, target, patterns))) {
    //   signalPassFailure();
    // }

    // Here's an example that uses applyPatternsAndFoldGreedily()
    auto func = getOperation();
    applyPatternsAndFoldGreedily(func, patterns);
    llvm::dbgs() << "We didn't fail!" << "\n";
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
mlir::NPCOMP::tcf::createStripIdentityPass() {
  return std::make_unique<StripIdentityPass>();
}
