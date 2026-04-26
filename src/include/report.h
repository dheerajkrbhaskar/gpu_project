#pragma once

#include <array>
#include <filesystem>
#include <string>

#include "gpu_pipeline.h"

namespace crack {

struct DetectionSummary {
    bool present = false;
    double percent = 0.0;
    unsigned char edgeCutoff = 20;
    double thresholdPercent = 0.20;
};

struct ModeReportRow {
    std::string label;
    double grayMs = 0.0;
    double sobelMs = 0.0;
    double thresholdMs = 0.0;
    double totalMs = 0.0;
    double speedup = 0.0;
};

struct SavedArtifactPaths {
    std::filesystem::path grayscale;
    std::filesystem::path magnitude;
    std::filesystem::path binary;
};

std::string modeDisplayLabel(Mode mode);

DetectionSummary makeDetectionSummary(const Image& magnitude);

ModeReportRow makeModeReportRow(const std::string& label, const CpuTiming& timing, double speedup);
ModeReportRow makeModeReportRow(const std::string& label, const GpuTiming& timing, double speedup);

SavedArtifactPaths saveArtifacts(const ProcessedImages& images,
                                 const std::filesystem::path& inputPath,
                                 const std::string& label,
                                 const std::string& timestamp);

void printReportHeader(const std::filesystem::path& inputPath, const Image& inputImage, const AppConfig& config);
void printResultSection(const DetectionSummary& summary);
void printPerformanceSummary(const std::array<ModeReportRow, 3>& rows);
void printOutputFiles(const SavedArtifactPaths& paths);

} // namespace crack