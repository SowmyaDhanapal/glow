#ifndef GLOW_BACKENDS_METAWARENN_METAWARENNFUNCTION_H
#define GLOW_BACKENDS_METAWARENN_METAWARENNFUNCTION_H

#include "glow/Backend/CompiledFunction.h"

#include "metawarenn_lib/metawarenn_graph.h"
#include "metawarenn_lib/metawarenn_utils.h"
#include "metawarenn_lib/optimizer/pass_manager.h"
#include "metawarenn_lib/optimizer/metawarenn_optimizer.h"

#define CHW_TO_HWC 0
#define HWC_TO_CHW 1
#define INVOKE_NNAC 1

namespace metawarenn {

class MetaWareNNFunction final : public CompiledFunction {
public:
  /// Constructor.
  MetaWareNNFunction(runtime::RuntimeBundle &&bundle, Function *F);
  /// @name CompiledFunction interface
  ///@{
  ~MetaWareNNFunction() override;
  std::string getCompileBackendName() const override { return "MetaWareNN"; }
  Error execute(ExecutionContext *context) override;
  ///@}
private:
  std::shared_ptr<::metawarenn::MWNNGraph> mwnn_graph_;
};

} // namespace metawarenn

#endif // GLOW_BACKENDS_METAWARENN_METAWARENNFUNCTION_H
