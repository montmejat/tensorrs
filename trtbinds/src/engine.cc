#include "../include/engine.h"

std::unique_ptr<ExecutionContextTRT> create_execution_context(const std::unique_ptr<CudaEngineTRT> &engine)
{
    auto execution_context = engine.get()->createExecutionContext();
    return std::unique_ptr<ExecutionContextTRT>(execution_context);
}