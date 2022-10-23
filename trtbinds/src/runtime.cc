#include "../include/runtime.h"

std::unique_ptr<CudaEngineTRT> deserialize_cuda_engine(const std::unique_ptr<RuntimeTRT> &runtime, const std::unique_ptr<HostMemoryTRT> &host_memory)
{
    auto engine = runtime.get()->deserializeCudaEngine(host_memory->data(), host_memory->size());
    return std::unique_ptr<CudaEngineTRT>(engine);
}