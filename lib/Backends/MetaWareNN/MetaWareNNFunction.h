#ifndef GLOW_BACKENDS_METAWARENN_METAWARENNFUNCTION_H
#define GLOW_BACKENDS_METAWARENN_METAWARENNFUNCTION_H

#include "glow/Graph/Utils.h"
#include "glow/Backend/BackendUtils.h"
#include "glow/Backend/CompiledFunction.h"
#include "glow/Graph/NodeValue.h"

#include "metawarenn_lib/metawarenn_graph.h"
#include "metawarenn_lib/metawarenn_utils.h"
#include "metawarenn_lib/metawarenn_element.h"
#include "metawarenn_lib/optimizer/pass_manager.h"
#include "metawarenn_lib/mwnnconvert/mwnn_protobuf/cpp_wrapper/MWNN.pb.h"
#include "metawarenn_lib/mwnn_inference_api/mwnn_inference_api.h"
#include "metawarenn_lib/executable_network/metawarenn_executable_graph.h"
#include "metawarenn_lib/mwnnconvert/mwnn_to_onnx_proto.h"

#include <fcntl.h>

#define CHW_TO_HWC 0
#define HWC_TO_CHW 1
#define INVOKE_NNAC 0

using namespace glow;
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

  static ElementType::element_type get_mwnn_type_glow(ElemKind glow_type) {
      switch (glow_type) {
          case ElemKind::BoolTy:
              return ElementType::element_type::boolean_;
          case ElemKind::Float16Ty:
              return ElementType::element_type::float16_;
          case ElemKind::FloatTy:
              return ElementType::element_type::float_;
          case ElemKind::Int8QTy:
              return ElementType::element_type::int8_;
          case ElemKind::Int16QTy:
              return ElementType::element_type::int16_;
          case ElemKind::Int32QTy:
              return ElementType::element_type::int32_;
          case ElemKind::Int64ITy:
              return ElementType::element_type::int64_;
          case ElemKind::UInt8QTy:
              return ElementType::element_type::uint8_;
          default:
              return ElementType::element_type::dynamic_;
      }
  }

private:
  /// Build the list of input and output placeholders.
  void findIOPlaceholders(Function *F);

  /// List of model input placeholders.
  PlaceholderList inputs_;

  /// List of model output placeholders.
  PlaceholderList outputs_;

  std::shared_ptr<metawarenn::Graph> graph_;
  #if EXECUTABLE_GRAPH_SERIALIZATION
  std::shared_ptr<metawarenn::ExecutableGraph> exe_graph_;
  #endif
};

} // namespace metawarenn

#endif // GLOW_BACKENDS_METAWARENN_METAWARENNFUNCTION_H
