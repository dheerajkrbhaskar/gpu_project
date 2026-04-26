#pragma once

#include "utils.h"

namespace crack {

struct CpuTiming {
    double grayscaleMs = 0.0;
    double sobelMs = 0.0;
    double thresholdMs = 0.0;
    double totalMs = 0.0;
};

struct CpuResult {
    ProcessedImages images;
    CpuTiming timing;
};

struct CpuOptions {
    bool useOpenMP = false;
    unsigned char threshold = 60;
};

CpuResult runCpuPipeline(const Image& inputBgr, const CpuOptions& options);

} // namespace crack