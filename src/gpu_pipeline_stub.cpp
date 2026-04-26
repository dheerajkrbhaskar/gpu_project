#include "gpu_pipeline.h"

using namespace std;

namespace crack {

GpuResult runGpuPipeline(const Image& inputBgr, const GpuOptions& options)
{
    GpuResult result;
    const CpuResult fallback = runCpuPipeline(inputBgr, CpuOptions{false, options.threshold});
    result.images = fallback.images;
    result.timing.grayscaleMs = fallback.timing.grayscaleMs;
    result.timing.sobelMs = fallback.timing.sobelMs;
    result.timing.thresholdMs = fallback.timing.thresholdMs;
    result.timing.totalMs = fallback.timing.totalMs;
    result.usedCpuFallback = true;
    result.note = "CUDA not available on this machine; GPU mode used the CPU fallback.";
    return result;
}

} // namespace crack