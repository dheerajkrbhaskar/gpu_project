#pragma once

#include "cpu_pipeline.h"

namespace crack {

struct GpuTiming {
    double h2dMs = 0.0;
    double kernelMs = 0.0;
    double d2hMs = 0.0;
    double totalMs = 0.0;
};

struct GpuResult {
    ProcessedImages images;
    GpuTiming timing;
    bool usedCpuFallback = false;
    std::string note;
};

struct GpuOptions {
    unsigned char threshold = 60;
};

GpuResult runGpuPipeline(const Image& inputBgr, const GpuOptions& options);

} // namespace crack
