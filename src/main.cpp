#include "gpu_pipeline.h"
#include "report.h"

#include <array>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>

using namespace std;

namespace crack {
namespace {

struct BenchSummary {
    std::string label;
    ProcessedImages images;
    CpuTiming cpuTiming;
    GpuTiming gpuTiming;
    bool usedCpuFallback = false;
    std::string note;
};

struct CpuBenchResult {
    BenchSummary summary;
    double averageTotalMs = 0.0;
};

struct GpuBenchResult {
    BenchSummary summary;
    double averageTotalMs = 0.0;
};

double computeSpeedup(double baselineMs, double candidateMs)
{
    if (baselineMs <= 0.0 || candidateMs <= 0.0) {
        return -1.0;
    }
    return baselineMs / candidateMs;
}

template <typename Timing>
void accumulateCpuTiming(CpuTiming& total, const Timing& sample)
{
    total.grayscaleMs += sample.grayscaleMs;
    total.sobelMs += sample.sobelMs;
    total.thresholdMs += sample.thresholdMs;
    total.totalMs += sample.totalMs;
}

CpuBenchResult benchmarkCpu(const Image& input, bool useOpenMP, int repeat, unsigned char threshold)
{
    CpuTiming totalTiming;
    CpuResult lastResult;

    for (int index = 0; index < repeat; ++index) {
        lastResult = runCpuPipeline(input, CpuOptions{useOpenMP, threshold});
        accumulateCpuTiming(totalTiming, lastResult.timing);
    }

    lastResult.timing.grayscaleMs = totalTiming.grayscaleMs / repeat;
    lastResult.timing.sobelMs = totalTiming.sobelMs / repeat;
    lastResult.timing.thresholdMs = totalTiming.thresholdMs / repeat;
    lastResult.timing.totalMs = totalTiming.totalMs / repeat;

    CpuBenchResult output;
    output.summary.label = useOpenMP ? "CPU (OpenMP)" : "CPU baseline";
    output.summary.images = std::move(lastResult.images);
    output.summary.cpuTiming = lastResult.timing;
    output.averageTotalMs = lastResult.timing.totalMs;
    return output;
}

GpuBenchResult benchmarkGpu(const Image& input, int repeat, unsigned char threshold)
{
    GpuTiming totalTiming;
    GpuResult lastResult;

    for (int index = 0; index < repeat; ++index) {
        lastResult = runGpuPipeline(input, GpuOptions{threshold});
        totalTiming.grayscaleMs += lastResult.timing.grayscaleMs;
        totalTiming.sobelMs += lastResult.timing.sobelMs;
        totalTiming.thresholdMs += lastResult.timing.thresholdMs;
        totalTiming.totalMs += lastResult.timing.totalMs;
    }

    lastResult.timing.grayscaleMs = totalTiming.grayscaleMs / repeat;
    lastResult.timing.sobelMs = totalTiming.sobelMs / repeat;
    lastResult.timing.thresholdMs = totalTiming.thresholdMs / repeat;
    lastResult.timing.totalMs = totalTiming.totalMs / repeat;

    GpuBenchResult output;
    output.summary.label = lastResult.usedCpuFallback ? "GPU (fallback)" : "GPU";
    output.summary.images = std::move(lastResult.images);
    output.summary.gpuTiming = lastResult.timing;
    output.summary.usedCpuFallback = lastResult.usedCpuFallback;
    output.summary.note = lastResult.note;
    output.averageTotalMs = lastResult.timing.totalMs;
    return output;
}

} // namespace
} // namespace crack

int main(int argc, char** argv)
{
    using namespace crack;

    AppConfig config;
    try {
        config = parseArguments(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n\n";
        printUsage(std::cerr, argv[0]);
        return 1;
    }

    if (config.showHelp) {
        printUsage(std::cout, argv[0]);
        return 0;
    }

    if (config.inputPath.empty()) {
        std::cerr << "Missing required --input argument\n\n";
        printUsage(std::cerr, argv[0]);
        return 1;
    }

    try {
        ensureOutputDirectory();

        const std::filesystem::path inputPath(config.inputPath);
        const Image inputImage = loadImageAsBgr(config.inputPath);
        const std::string timestamp = timestampString();

        const CpuBenchResult cpuBaseline = benchmarkCpu(inputImage, false, config.repeat, config.threshold);
        const CpuBenchResult openMpResult = benchmarkCpu(inputImage, true, config.repeat, config.threshold);
        const GpuBenchResult gpuResult = benchmarkGpu(inputImage, config.repeat, config.threshold);

        printReportHeader(inputPath, inputImage, config);

        const ProcessedImages* selectedImages = &cpuBaseline.summary.images;

        if (config.mode == Mode::Omp) {
            selectedImages = &openMpResult.summary.images;
        } else if (config.mode == Mode::Gpu) {
            selectedImages = &gpuResult.summary.images;
        }

        const DetectionSummary detection = makeDetectionSummary(selectedImages->binary);
        printResultSection(detection);

        printPerformanceSummary(std::array<ModeReportRow, 3>{
            makeModeReportRow("CPU", cpuBaseline.summary.cpuTiming, 1.00),
            makeModeReportRow("OpenMP", openMpResult.summary.cpuTiming, computeSpeedup(cpuBaseline.averageTotalMs, openMpResult.averageTotalMs)),
            makeModeReportRow("GPU", gpuResult.summary.gpuTiming, computeSpeedup(cpuBaseline.averageTotalMs, gpuResult.averageTotalMs)),
        });

        const std::string outputLabel = modeLabel(config.mode);
        const SavedArtifactPaths outputPaths = saveArtifacts(*selectedImages, inputPath, outputLabel, timestamp);
        printOutputFiles(outputPaths);

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
