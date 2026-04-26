#include "gpu_pipeline.h"

namespace crack {

GpuResult runGpuPipeline(const Image& inputBgr, const GpuOptions& options)
{
    GpuResult result;
    const CpuResult fallback = runCpuPipeline(inputBgr, CpuOptions{false, options.threshold});
    result.images = fallback.images;
    result.timing.h2dMs = -1.0;
    result.timing.kernelMs = -1.0;
    result.timing.d2hMs = -1.0;
    result.timing.totalMs = fallback.timing.totalMs;
    result.usedCpuFallback = true;
    result.note = "CUDA not available on this machine; GPU mode used the CPU fallback.";
    return result;
}

} // namespace crack