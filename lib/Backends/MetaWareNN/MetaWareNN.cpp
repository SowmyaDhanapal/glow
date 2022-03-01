#include "MetaWareNN.h"

namespace metawarenn {

MetaWareNNBackend::MetaWareNNBackend() {
  LOG(INFO) << "MetaWareNNBackend constructor";
}

runtime::DeviceManager * MetaWareNNBackend::createDeviceManager(
    const runtime::DeviceConfig &deviceConfig) {
  return createMetaWareNNDeviceManager(deviceConfig);
}

Expected<std::unique_ptr<CompiledFunction>> MetaWareNNBackend::compile(
    Function *F, const BackendOptions &opts) const
{
  LOG(INFO) << "In MetaWareNNBackend::compile";

  std::unique_ptr<MetaWareNNFunction> compiledFunc =
      glow::make_unique<MetaWareNNFunction>(
          glow::runtime::RuntimeBundle::create(*F), F);
  return Expected<std::unique_ptr<CompiledFunction>>(std::move(compiledFunc));
}

bool MetaWareNNBackend::isOpSupported(const NodeInfo &NI) const
{
  switch (NI.getKind())
{
  case Kinded::Kind::ConvolutionNodeKind:
  case Kinded::Kind::AvgPoolNodeKind:
  case Kinded::Kind::ReluNodeKind:
  case Kinded::Kind::AddNodeKind:
  case Kinded::Kind::ReshapeNodeKind:
  case Kinded::Kind::SaveNodeKind:
  case Kinded::Kind::TransposeNodeKind:
  case Kinded::Kind::LocalResponseNormalizationNodeKind:
  case Kinded::Kind::MaxPoolNodeKind:
  case Kinded::Kind::GemmNodeKind:
  case Kinded::Kind::BatchNormalizationNodeKind:
  case Kinded::Kind::ConcatNodeKind:
  case Kinded::Kind::ChannelShuffleNodeKind:
  case Kinded::Kind::ClipNodeKind:
  case Kinded::Kind::FullyConnectedNodeKind:
  case Kinded::Kind::SoftMaxNodeKind:
  case Kinded::Kind::SliceNodeKind:
  case Kinded::Kind::MatMulNodeKind:
  case Kinded::Kind::LeakyReluNodeKind:
  case Kinded::Kind::MulNodeKind:
  case Kinded::Kind::QuantizeNodeKind:
  case Kinded::Kind::DequantizeNodeKind:
  case Kinded::Kind::RescaleQuantizedNodeKind:
      return true;
  default:
      return false;
  }
}

bool MetaWareNNBackend::shouldLower(const ::glow::Node *N) const
{
  return false;
}

unsigned MetaWareNNBackend::numDevices()
{
  return 1;
}

}  // namespace metawarenn
