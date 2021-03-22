#include "DummyFunction.h"

namespace metawarenn {

MetaWareNNFunction::MetaWareNNFunction(Function *F)
    : CompiledFunction(runtime::RuntimeBundle::create(*F)) {}

MetaWareNNFunction::~MetaWareNNFunction() {}

Error MetaWareNNFunction::execute(ExecutionContext *context){ 
    return Error::success();}

} //namespace metawarenn