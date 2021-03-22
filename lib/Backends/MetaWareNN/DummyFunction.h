#ifndef GLOW_BACKENDS_METAWARENN_METAWARENNFUNCTION_H
#define GLOW_BACKENDS_METAWARENN_METAWARENNFUNCTION_H

#include "glow/Backend/BackendUtils.h"
#include "glow/Backend/CompiledFunction.h"

using namespace glow;

namespace metawarenn
{
class MetaWareNNFunction final : public CompiledFunction {
public:
  MetaWareNNFunction();
  MetaWareNNFunction(Function *F);
  ~MetaWareNNFunction() override;
  std::string getCompileBackendName() const override { return "MetaWareNN"; }
  Error execute(ExecutionContext *context) override;
};

} //namespace metawarenn
#endif // GLOW_BACKENDS_HABANA_HABANAFUNCTION_H