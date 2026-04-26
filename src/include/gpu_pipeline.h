#pragma once

#include "cpu_pipeline.h"

namespace crack {

struct GpuTiming {
    double grayscaleMs = 0.0;
    double sobelMs = 0.0;
    double thresholdMs = 0.0;
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