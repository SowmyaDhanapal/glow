#ifndef GLOW_BACKENDS_METAWARENN_METAWARENNFUNCTION_H
#define GLOW_BACKENDS_METAWARENN_METAWARENNFUNCTION_H

#include "glow/Backend/CompiledFunction.h"

#include "metawarenn_lib/metawarenn_graph.h"
#include "metawarenn_lib/metawarenn_utils.h"
#include "metawarenn_lib/optimizer/pass_manager.h"
#include "metawarenn_lib/optimizer/metawarenn_optimizer.h"
#include "metawarenn_lib/mwnnconvert/mwnn_protobuf/cpp_wrapper/MWNN.pb.h"

#define CHW_TO_HWC 0
#define HWC_TO_CHW 1
#define INVOKE_NNAC 1

namespace metawarenn {

static int graph_count;
class MetaWareNNFunction final : public CompiledFunction {
public:
  /// Constructor.
  MetaWareNNFunction(runtime::RuntimeBundle &&bundle, Function *F);
  /// @name CompiledFunction interface
  ///@{
  ~MetaWareNNFunction() override;
  std::string getCompileBackendName() const override { return "MetaWareNN"; }
  Error execute(ExecutionContext *context) override;
  const PlaceholderList &getInputs() const { return inputs_; }
  const PlaceholderList &getOutputs() const { return outputs_; }
  ///@}
private:
  /// Build the list of input and output placeholders.
  void findIOPlaceholders(Function *F);

  /// List of model input placeholders.
  PlaceholderList inputs_;

  /// List of model output placeholders.
  PlaceholderList outputs_;

  std::shared_ptr<::metawarenn::MWNNGraph> mwnn_graph_;
};

} // namespace metawarenn

#endif // GLOW_BACKENDS_METAWARENN_METAWARENNFUNCTION_H
